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

    var errorDescription: String? {
        switch self {
        case .permissionDenied: "Microphone access is off. Enable it in Settings to record."
        case .couldNotStart: "The microphone could not start recording."
        case .notRecording: "There is no active recording to stop."
        }
    }
}

@MainActor
final class AudioRecorder: ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var isStarting = false
    @Published private(set) var startedAt: Date?

    private var recorder: AVAudioRecorder?

    func start(at url: URL) async throws {
        guard !isRecording, !isStarting else { return }
        isStarting = true
        defer { isStarting = false }
        guard await AVAudioApplication.requestRecordPermission() else {
            throw AudioRecorderError.permissionDenied
        }

        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
            try audioSession.setActive(true)
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
            recorder.prepareToRecord()
            guard recorder.record() else { throw AudioRecorderError.couldNotStart }
            self.recorder = recorder
            startedAt = Date()
            isRecording = true
        } catch {
            try? audioSession.setActive(false)
            try? FileManager.default.removeItem(at: url)
            throw error
        }
    }

    func stop() throws -> CapturedAudio {
        guard let recorder, isRecording else { throw AudioRecorderError.notRecording }
        let duration = max(0, Int((recorder.currentTime * 1_000).rounded()))
        let url = recorder.url
        recorder.stop()
        self.recorder = nil
        isRecording = false
        startedAt = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        return CapturedAudio(url: url, durationMilliseconds: duration)
    }
}

@MainActor
final class AudioPlayback: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published private(set) var playingID: UUID?
    private var player: AVAudioPlayer?

    func toggle(id: UUID, url: URL) throws {
        if playingID == id {
            stop()
            return
        }
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playback, mode: .default)
        try audioSession.setActive(true)
        let player = try AVAudioPlayer(contentsOf: url)
        player.delegate = self
        guard player.play() else { throw AudioRecorderError.couldNotStart }
        self.player = player
        playingID = id
    }

    func stop() {
        player?.stop()
        player = nil
        playingID = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in stop() }
    }
}
