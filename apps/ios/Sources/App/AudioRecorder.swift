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
    private var engine: AVAudioEngine?
    private var sink: CaptureSink?
    private var recordingURL: URL?
    /// The input port and rate the tap was installed with, so a later
    /// engine configuration change can tell "the tapped input is gone"
    /// (stop and salvage) from "only the outputs changed" (restart).
    private var activeInputUID: String?
    private var activeSampleRate: Double?
    private var interruptionObserver: (any NSObjectProtocol)?
    private var routeChangeObserver: (any NSObjectProtocol)?
    private var mediaServicesResetObserver: (any NSObjectProtocol)?
    private var engineChangeObserver: (any NSObjectProtocol)?

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
            let engine = AVAudioEngine()
            let input = engine.inputNode
            let format = input.outputFormat(forBus: 0)
            guard format.commonFormat == .pcmFormatFloat32,
                  !format.isInterleaved, format.channelCount > 0,
                  format.sampleRate.isFinite, format.sampleRate > 0
            else { throw AudioRecorderError.couldNotStart }
            let sink = try CaptureSink(url: url, sampleRate: format.sampleRate)
            activeInputUID = AVAudioSession.sharedInstance().currentRoute.inputs.first?.uid
            activeSampleRate = format.sampleRate
            input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
                sink.append(buffer)
            }
            engine.prepare()
            try engine.start()
            self.engine = engine
            self.sink = sink
            recordingURL = url
            startedAt = Date()
            isRecording = true
            observeSessionState()
        } catch {
            engine?.inputNode.removeTap(onBus: 0)
            engine?.stop()
            engine = nil
            sink = nil
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

    /// Returns ordered PCM16 frames that were also written to the staged WAV.
    func drainStreamChunks() -> [Data] {
        sink?.drain() ?? []
    }

    /// Stop the recorder and collect whatever was captured so far. Shared by
    /// the user-initiated stop and every forced stop.
    private func finishCapture() -> CapturedAudio? {
        guard let engine, let sink, let url = recordingURL else { return nil }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        let duration = sink.finish()
        self.engine = nil
        recordingURL = nil
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
        mediaServicesResetObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.mediaServicesWereResetNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleMediaServicesReset()
            }
        }
        engineChangeObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.isRecording, let engine = self.engine else { return }
                // This notification also fires for output-only route
                // changes (plugging in headphones, a Bluetooth output
                // connecting), which must not end capture — the
                // route-change handler below already ends it when the
                // input itself is lost. When the tapped input is still on
                // the route at the same format, the engine only needs a
                // restart; only its loss or a failed restart salvages the
                // take as a forced stop.
                let inputAlive = AVAudioSession.sharedInstance().currentRoute.inputs
                    .contains { $0.uid == self.activeInputUID }
                if inputAlive,
                   engine.inputNode.outputFormat(forBus: 0).sampleRate == self.activeSampleRate,
                   (try? engine.start()) != nil {
                    return
                }
                self.forcedStop(
                    message: "The microphone configuration changed. The audio captured so far was saved to history."
                )
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
        if let mediaServicesResetObserver {
            NotificationCenter.default.removeObserver(mediaServicesResetObserver)
        }
        if let engineChangeObserver {
            NotificationCenter.default.removeObserver(engineChangeObserver)
        }
        interruptionObserver = nil
        routeChangeObserver = nil
        mediaServicesResetObserver = nil
        engineChangeObserver = nil
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
        // Only the loss of the current input device ends capture; other route
        // changes (a category switch, new outputs) leave the recorder running.
        guard reason == .oldDeviceUnavailable else { return }
        forcedStop(
            message: "The microphone became unavailable. The audio captured so far was saved to history."
        )
    }

    /// A media services reset invalidates every audio object, including a
    /// running recorder whose delegate is not called, so capture is finalized
    /// with the audio gathered so far.
    private func handleMediaServicesReset() {
        guard isRecording else { return }
        forcedStop(
            message: "The audio system restarted. The audio captured so far was saved to history."
        )
    }
}

/// The tap owns one sink. The tap callback only copies the buffer's channel
/// memory (valid solely inside the callback); every lock, allocation,
/// resample, and file write happens on the sink's serial queue, off the
/// real-time audio thread. `stop` removes the tap before finalizing the WAV
/// header.
private final class CaptureSink: @unchecked Sendable {
    private let lock = NSLock()
    /// Serializes encode+write work moved off the render thread.
    private let queue = DispatchQueue(label: "dev.starling.capture.sink", qos: .userInitiated)
    private let file: FileHandle
    private let inputRate: Double
    private var inputIndex = 0
    private var outputIndex = 0
    private var lastSample: Float = 0
    private var frames: [Data] = []
    private var byteCount = 0
    private var finished = false
    /// Set by the first failed write; later appends are skipped so the file
    /// stays a valid (truncated) WAV instead of growing a hole.
    private var failed = false

    init(url: URL, sampleRate: Double) throws {
        inputRate = sampleRate
        guard FileManager.default.createFile(
            atPath: url.path, contents: Data(repeating: 0, count: 44)
        ) else {
            throw AudioRecorderError.couldNotStart
        }
        file = try FileHandle(forWritingTo: url)
        try file.seekToEnd()
    }

    deinit {
        // The start-failure path drops the sink without finish(); the file
        // descriptor must not leak. Pending queue work holds no strong
        // reference, so draining it first only orders the close.
        queue.sync {}
        if !finished {
            try? file.synchronize()
            try? file.close()
        }
    }

    /// Copies the tap buffer on the calling (render) thread and moves all
    /// heavy work to the sink queue.
    func append(_ buffer: AVAudioPCMBuffer) {
        guard let channels = buffer.floatChannelData, buffer.frameLength > 0 else { return }
        let count = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        var copies = [[Float]]()
        copies.reserveCapacity(channelCount)
        for channel in 0..<channelCount {
            copies.append(Array(UnsafeBufferPointer(start: channels[channel], count: count)))
        }
        queue.async { [weak self] in
            self?.encodeAndWrite(channels: copies, count: count)
        }
    }

    private func encodeAndWrite(channels: [[Float]], count: Int) {
        lock.lock()
        defer { lock.unlock() }
        guard !finished, !failed else { return }
        let channelCount = channels.count
        var mono = [Float](repeating: 0, count: count)
        for channel in 0..<channelCount {
            for i in 0..<count { mono[i] += channels[channel][i] / Float(channelCount) }
        }
        // Carry the output clock across tap boundaries; every source frame
        // contributes once, including when the hardware runs at 44.1 kHz.
        var bytes = Data()
        let end = inputIndex + count
        while Double(outputIndex) * inputRate / 16_000 < Double(end) {
            let source = Double(outputIndex) * inputRate / 16_000 - Double(inputIndex)
            let lower = Int(floor(source))
            let fraction = Float(source - Double(lower))
            let a = lower < 0 ? lastSample : mono[min(lower, count - 1)]
            let b = mono[min(max(lower + 1, 0), count - 1)]
            let value = max(-1, min(1, a + (b - a) * fraction))
            let sample = Int16(max(-32768, min(32767, Int((value * 32768).rounded()))))
            var little = sample.littleEndian
            withUnsafeBytes(of: &little) { bytes.append(contentsOf: $0) }
            outputIndex += 1
        }
        lastSample = mono[count - 1]
        inputIndex = end
        guard !bytes.isEmpty else { return }
        do {
            // The throwing API: the legacy `write(_:)` raises an ObjC
            // exception (crashing the app) on a full disk or vanished
            // file instead of returning an error.
            try file.write(contentsOf: bytes)
            byteCount += bytes.count
            frames.append(bytes)
        } catch {
            failed = true
        }
    }

    /// True once a write failed; capture produced a valid but truncated
    /// WAV and the caller may want to end it early.
    var writeFailed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return failed
    }

    func drain() -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        let result = frames
        frames.removeAll(keepingCapacity: true)
        return result
    }

    func finish() -> Int {
        // Wait out any queued copies first so the finalized header covers
        // every frame the tap produced.
        queue.sync {}
        lock.lock()
        defer { lock.unlock() }
        finished = true
        var header = Data("RIFF".utf8)
        var riffSize = UInt32(36 + byteCount).littleEndian
        withUnsafeBytes(of: &riffSize) { header.append(contentsOf: $0) }
        header.append(contentsOf: "WAVEfmt ".utf8)
        let formatValues: [UInt32] = [16, 0x00010001, 16_000, 32_000, 0x00100002]
        for value in formatValues {
            var little = value.littleEndian
            withUnsafeBytes(of: &little) { header.append(contentsOf: $0) }
        }
        header.append(contentsOf: "data".utf8)
        var length = UInt32(byteCount).littleEndian
        withUnsafeBytes(of: &length) { header.append(contentsOf: $0) }
        file.seek(toFileOffset: 0)
        try? file.write(contentsOf: header)
        try? file.synchronize()
        try? file.close()
        return byteCount * 1_000 / 32_000
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
