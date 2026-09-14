import Combine
import Foundation
import StarlingVoiceCore

@MainActor
final class AppModel: ObservableObject {
    @Published private(set) var sessions: [SessionRecord] = []
    @Published private(set) var isWorking = false
    @Published var selectedSession: SessionRecord?
    @Published var errorMessage: String?

    let recorder = AudioRecorder()
    let playback = AudioPlayback()
    private let repository: SessionRepository
    private(set) var audioURLs: [UUID: URL] = [:]

    init(repository: SessionRepository) {
        self.repository = repository
        Task { await recoverAndReload() }
    }

    func toggleRecording(configuration: ServerConfiguration) async {
        if recorder.isRecording {
            await finishRecording(configuration: configuration)
        } else {
            do {
                let stagingURL = try await repository.stagingRecordingURL()
                try await recorder.start(at: stagingURL)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func retry(_ record: SessionRecord, configuration: ServerConfiguration) async {
        await transcribe(record, configuration: configuration)
    }

    func delete(_ record: SessionRecord) async {
        do {
            playback.stop()
            try await repository.delete(record.id)
            selectedSession = nil
            await reload()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func togglePlayback(_ record: SessionRecord) {
        guard let url = audioURLs[record.id] else {
            errorMessage = "The saved recording is unavailable."
            return
        }
        do {
            try playback.toggle(id: record.id, url: url)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func transcriptExportURL(_ record: SessionRecord) async -> URL? {
        do {
            return try await repository.transcriptExportURL(for: record)
        } catch {
            errorMessage = error.localizedDescription
            return nil
        }
    }

    func reload() async {
        do {
            let loaded = try await repository.list()
            var urls: [UUID: URL] = [:]
            for record in loaded {
                urls[record.id] = await repository.recordingURL(for: record)
            }
            sessions = loaded
            audioURLs = urls
            if let selectedID = selectedSession?.id {
                selectedSession = loaded.first { $0.id == selectedID }
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func finishRecording(configuration: ServerConfiguration) async {
        do {
            let capture = try recorder.stop()
            // The source is already app-private. Promotion deletes it only
            // after the durable audio copy and manifest have both succeeded.
            let record = try await repository.commitStagedRecording(
                at: capture.url,
                durationMilliseconds: capture.durationMilliseconds
            )
            await reload()
            await transcribe(record, configuration: configuration)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func recoverAndReload() async {
        do {
            _ = try await repository.recoverPendingRecordings()
        } catch {
            errorMessage = "A pending recording could not be recovered: \(error.localizedDescription)"
        }
        await reload()
    }

    private func transcribe(_ record: SessionRecord, configuration: ServerConfiguration) async {
        guard !isWorking else { return }
        isWorking = true
        defer { isWorking = false }
        do {
            let attempting = try await repository.markAttempt(record.id)
            await reload()
            let recordingURL = await repository.recordingURL(for: attempting)
            let transcript = try await StarlingClient(configuration: configuration)
                .transcribe(recordingURL: recordingURL, requestID: UUID().uuidString)
            _ = try await repository.saveTranscript(transcript, for: attempting.id)
            await reload()
        } catch {
            do {
                _ = try await repository.saveFailure(error, for: record.id)
            } catch {
                errorMessage = error.localizedDescription
            }
            await reload()
            errorMessage = error.localizedDescription
        }
    }
}
