import AVFoundation
import Combine
import Foundation

struct CapturedAudio {
    let url: URL
    let durationMilliseconds: Int
}

enum AudioRecorderError: LocalizedError {
    case permissionDenied
    case couldNotStart
    case notRecording
    case recordingActive

    var errorDescription: String? {
        switch self {
        case .permissionDenied: "Microphone access is off. Enable it in Settings to record."
        case .couldNotStart: "The microphone could not start recording."
        case .notRecording: "There is no active recording to stop."
        case .recordingActive: "Stop the recording before playing history audio."
        }
    }
}

/// Single owner of the process-wide `AVAudioSession`. Capture claims the
/// session exclusively: while a recording is active, playback can neither
/// reconfigure the category nor deactivate the session, and each component
/// deactivates only the activation it performed itself.
@MainActor
final class AudioSessionCoordinator {
    private enum Owner { case none, recorder, playback }

    private var owner: Owner = .none
    private let session: AVAudioSession

    init(session: AVAudioSession = AVAudioSession.sharedInstance()) {
        self.session = session
    }

    var isRecording: Bool { owner == .recorder }

    /// Claim the session for capture and configure it for recording.
    func beginRecording() throws {
        owner = .recorder
        do {
            try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
            try session.setActive(true)
        } catch {
            owner = .none
            throw error
        }
    }

    /// Release the recording claim. The system may already have deactivated
    /// the session (for example during an interruption), so a failing
    /// deactivation is ignored.
    func endRecording() {
        guard owner == .recorder else { return }
        owner = .none
        try? session.setActive(false, options: [.notifyOthersOnDeactivation])
    }

    /// Claim the session for playback. Refused while a recording owns it:
    /// switching the category to `.playback` would break active capture.
    func beginPlayback() throws {
        guard owner != .recorder else { throw AudioRecorderError.recordingActive }
        owner = .playback
        do {
            try session.setCategory(.playback, mode: .default)
            try session.setActive(true)
        } catch {
            owner = .none
            throw error
        }
    }

    /// A playback stop may only deactivate a session that playback activated
    /// itself, and never one a recording owns.
    func endPlayback() {
        guard owner == .playback else { return }
        owner = .none
        try? session.setActive(false, options: [.notifyOthersOnDeactivation])
    }
}

@MainActor
final class AudioRecorder: NSObject, ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var isStarting = false
    @Published private(set) var startedAt: Date?

    /// Called when capture ends without the stop button — an audio-session
    /// interruption such as a call or Siri, a lost microphone route, or a
    /// recorder error — with the audio captured up to that point.
    var onForcedStop: (@MainActor (CapturedAudio, String) -> Void)?

    private let coordinator: AudioSessionCoordinator
    private var recorder: AVAudioRecorder?
    private var interruptionObserver: (any NSObjectProtocol)?
    private var routeChangeObserver: (any NSObjectProtocol)?

    init(coordinator: AudioSessionCoordinator) {
        self.coordinator = coordinator
    }

    func start(at url: URL) async throws {
        guard !isRecording, !isStarting else { return }
        isStarting = true
        defer { isStarting = false }
        guard await AVAudioApplication.requestRecordPermission() else {
            throw AudioRecorderError.permissionDenied
        }

        do {
            try coordinator.beginRecording()
            let settings: [String: Any] = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 16_000.0,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsFloatKey: false,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
            ]
            let recorder = try AVAudioRecorder(url: url, settings: settings)
            recorder.delegate = self
            recorder.prepareToRecord()
            guard recorder.record() else { throw AudioRecorderError.couldNotStart }
            self.recorder = recorder
            startedAt = Date()
            isRecording = true
            observeSessionState()
        } catch {
            coordinator.endRecording()
            try? FileManager.default.removeItem(at: url)
            throw error
        }
    }

    func stop() throws -> CapturedAudio {
        guard isRecording, let capture = finishCapture() else {
            throw AudioRecorderError.notRecording
        }
        coordinator.endRecording()
        return capture
    }

    /// Stop the recorder and collect whatever was captured so far. Shared by
    /// the user-initiated stop and every forced stop.
    private func finishCapture() -> CapturedAudio? {
        guard let recorder else { return nil }
        let duration = max(0, Int((recorder.currentTime * 1_000).rounded()))
        let url = recorder.url
        recorder.stop()
        self.recorder = nil
        stopObservingSessionState()
        isRecording = false
        startedAt = nil
        return CapturedAudio(url: url, durationMilliseconds: duration)
    }

    /// Finalize capture that ended without the stop button. The recorder is
    /// already dead by then — its finish delegate is not called for
    /// interruptions — so the audio captured so far is saved and the reason
    /// reported. Capture is never resumed implicitly; the user restarts it
    /// deliberately.
    private func forcedStop(message: String) {
        guard isRecording else { return }
        coordinator.endRecording()
        guard let capture = finishCapture() else { return }
        onForcedStop?(capture, message)
    }

    private func observeSessionState() {
        interruptionObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            // Only the raw value crosses into the task; it is unconditionally
            // sendable.
            let rawType = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            Task { @MainActor [weak self] in
                self?.handleInterruption(rawType.flatMap(AVAudioSession.InterruptionType.init(rawValue:)))
            }
        }
        routeChangeObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.routeChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            let rawReason = notification.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt
            Task { @MainActor [weak self] in
                self?.handleRouteChange(rawReason.flatMap(AVAudioSession.RouteChangeReason.init(rawValue:)))
            }
        }
    }

    private func stopObservingSessionState() {
        if let interruptionObserver {
            NotificationCenter.default.removeObserver(interruptionObserver)
        }
        if let routeChangeObserver {
            NotificationCenter.default.removeObserver(routeChangeObserver)
        }
        interruptionObserver = nil
        routeChangeObserver = nil
    }

    private func handleInterruption(_ type: AVAudioSession.InterruptionType?) {
        guard isRecording, type == .began else { return }
        // The system has already silenced capture; .ended is ignored because
        // resuming after a call would record without an explicit user action.
        forcedStop(
            message: "Recording was interrupted by a call or another audio app. The audio captured so far was saved to history."
        )
    }

    private func handleRouteChange(_ reason: AVAudioSession.RouteChangeReason?) {
        guard isRecording else { return }
        guard reason == .oldDeviceUnavailable || reason == .mediaServicesWereReset else { return }
        forcedStop(
            message: "The microphone became unavailable. The audio captured so far was saved to history."
        )
    }
}

extension AudioRecorder: AVAudioRecorderDelegate {
    nonisolated func audioRecorderEncodeErrorDidOccur(
        _ recorder: AVAudioRecorder,
        error: (any Error)?
    ) {
        let reason = error?.localizedDescription ?? "the microphone stopped unexpectedly"
        Task { @MainActor [weak self] in
            self?.forcedStop(
                message: "Recording stopped: \(reason). The audio captured so far was saved to history."
            )
        }
    }
}

@MainActor
final class AudioPlayback: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published private(set) var playingID: UUID?
    private var player: AVAudioPlayer?
    private let coordinator: AudioSessionCoordinator

    init(coordinator: AudioSessionCoordinator) {
        self.coordinator = coordinator
    }

    func toggle(id: UUID, url: URL) throws {
        if playingID == id {
            stop()
            return
        }
        // Claims the shared session; refused while a recording is active so
        // playback cannot reconfigure the session mid-capture.
        try coordinator.beginPlayback()
        do {
            player?.stop()
            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = self
            guard player.play() else { throw AudioRecorderError.couldNotStart }
            self.player = player
            playingID = id
        } catch {
            coordinator.endPlayback()
            throw error
        }
    }

    func stop() {
        guard player != nil else {
            // Nothing is playing: never deactivate a session another
            // component (for example a live recording) owns.
            return
        }
        player?.stop()
        player = nil
        playingID = nil
        coordinator.endPlayback()
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in stop() }
    }
}
