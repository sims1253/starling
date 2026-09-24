import Combine
import Foundation
import StarlingVoiceCore

@MainActor
final class AppModel: ObservableObject {
    @Published private(set) var sessions: [SessionRecord] = []
    @Published private(set) var unreadableSessionCount = 0
    @Published private(set) var isWorking = false
    /// True for the whole start-recording path (staging, permission, session
    /// claim), so history actions can stay locked before capture begins.
    @Published private(set) var isStartingCapture = false
    @Published var selectedSession: SessionRecord?
    @Published var errorMessage: String?
    @Published private(set) var livePartial = ""

    let recorder: AudioRecorder
    let playback: AudioPlayback
    private let repository: SessionRepository
    private(set) var audioURLs: [UUID: URL] = [:]
    private var liveStream: LiveTranscription?
    private var streamPump: Task<Void, Never>?

    init(repository: SessionRepository) {
        self.repository = repository
        let coordinator = AudioSessionCoordinator()
        recorder = AudioRecorder(coordinator: coordinator)
        playback = AudioPlayback(coordinator: coordinator)
        // Capture can end without the stop button (a call, Siri, a lost
        // microphone route). Keep the audio captured so far in history.
        recorder.onForcedStop = { [weak self] capture, message in
            self?.streamPump?.cancel()
            self?.liveStream?.close()
            self?.streamPump = nil
            self?.liveStream = nil
            self?.livePartial = ""
            Task { await self?.preserveInterruptedRecording(capture, message: message) }
        }
        Task { await recoverAndReload() }
    }

    func toggleRecording(configuration: ServerConfiguration) async {
        if recorder.isRecording {
            await finishRecording(configuration: configuration)
        } else {
            isStartingCapture = true
            defer { isStartingCapture = false }
            do {
                // Capture owns the shared audio session exclusively; playback
                // gives it up before recording starts.
                playback.stop()
                let stagingURL = try await repository.stagingRecordingURL()
                try await recorder.start(at: stagingURL)
                guard recorder.isRecording else { return }
                livePartial = ""
                do {
                    liveStream = try LiveTranscription(configuration: configuration) { [weak self] text in
                        self?.livePartial = text
                    }
                } catch {
                    // Stream setup failure is deterministic (an endpoint
                    // or scheme the WS URL builder rejects): the recording
                    // itself is fine and falls back to the full upload,
                    // but the missing partials must be explained instead
                    // of silently dropped.
                    liveStream = nil
                    errorMessage = "Live transcription is unavailable: \(error.localizedDescription)"
                }
                streamPump = Task { [weak self] in
                    while let self, self.recorder.isRecording, !Task.isCancelled {
                        let chunks = self.recorder.drainStreamChunks()
                        if let stream = self.liveStream {
                            do {
                                try await stream.send(chunks)
                            } catch {
                                stream.close()
                                self.liveStream = nil
                                // Stop showing a partial the stream can no
                                // longer update. Draining continues so the
                                // sink's frame buffer stays bounded for the
                                // rest of the take.
                                self.livePartial = ""
                            }
                        }
                        try? await Task.sleep(for: .milliseconds(100))
                    }
                }
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
            let listing = try await repository.list()
            var urls: [UUID: URL] = [:]
            for record in listing.sessions {
                urls[record.id] = await repository.recordingURL(for: record)
            }
            sessions = listing.sessions
            unreadableSessionCount = listing.damagedDirectories.count
            audioURLs = urls
            if let selectedID = selectedSession?.id {
                selectedSession = listing.sessions.first { $0.id == selectedID }
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Commit audio whose capture ended without the stop button so it stays
    /// reviewable and retryable, and tell the user what happened.
    private func preserveInterruptedRecording(_ capture: CapturedAudio, message: String) async {
        do {
            _ = try await repository.commitStagedRecording(
                at: capture.url,
                durationMilliseconds: capture.durationMilliseconds
            )
            await reload()
            errorMessage = message
        } catch {
            errorMessage = "Recording ended unexpectedly and its audio could not be saved: \(error.localizedDescription)"
        }
    }

    private func finishRecording(configuration: ServerConfiguration) async {
        var streamToClose: LiveTranscription?
        do {
            let capture = try recorder.stop()
            await streamPump?.value
            streamPump = nil
            let stream = liveStream
            streamToClose = stream
            liveStream = nil
            if let stream {
                do {
                    try await stream.send(recorder.drainStreamChunks())
                } catch {
                    stream.close()
                }
            }
            // The source is already app-private. Promotion deletes it only
            // after the durable audio copy and manifest have both succeeded.
            let record = try await repository.commitStagedRecording(
                at: capture.url,
                durationMilliseconds: capture.durationMilliseconds
            )
            await reload()
            await transcribe(record, configuration: configuration, stream: stream)
            streamToClose = nil
            livePartial = ""
        } catch {
            streamToClose?.close()
            livePartial = ""
            errorMessage = error.localizedDescription
        }
    }

    private func recoverAndReload() async {
        do {
            _ = try await repository.recoverPendingRecordings()
        } catch {
            errorMessage = error.localizedDescription
        }
        // A fresh process has no requests in flight; sessions still marked
        // transcribing belong to an interrupted attempt and become retryable.
        do {
            _ = try await repository.reconcileInterruptedTranscriptions()
        } catch {
            errorMessage = error.localizedDescription
        }
        await reload()
    }

    private func transcribe(
        _ record: SessionRecord,
        configuration: ServerConfiguration,
        stream: LiveTranscription? = nil
    ) async {
        guard !isWorking else {
            // A Retry on a history row won the race for the working slot:
            // close the stream we were handed rather than dropping the
            // reference and leaving the socket for the server's heartbeat
            // reaper to collect.
            stream?.close()
            return
        }
        isWorking = true
        defer { isWorking = false }
        // Why a stream-produced transcript was discarded, kept for the
        // failure path so the fallback stays diagnosable.
        var streamFailureNote: String?
        do {
            let attempting = try await repository.markAttempt(record.id)
            await reload()
            let recordingURL = await repository.recordingURL(for: attempting)
            let transcript: Transcript
            if let stream {
                do {
                    transcript = try await stream.finish()
                } catch {
                    // The stream already carried every frame and the
                    // commit; the full upload is the only way to a
                    // transcript now. Keep the reason — if the upload
                    // also fails, the user sees both.
                    streamFailureNote = "Live transcription failed (\(error.localizedDescription)); the recording was re-uploaded in full."
                    transcript = try await StarlingClient(configuration: configuration)
                        .transcribe(recordingURL: recordingURL, requestID: UUID().uuidString)
                }
            } else {
                transcript = try await StarlingClient(configuration: configuration)
                    .transcribe(recordingURL: recordingURL, requestID: UUID().uuidString)
            }
            _ = try await repository.saveTranscript(transcript, for: attempting.id)
            await reload()
        } catch {
            do {
                _ = try await repository.saveFailure(error, for: record.id)
            } catch {
                errorMessage = error.localizedDescription
            }
            await reload()
            errorMessage = streamFailureNote.map { "\($0) \(error.localizedDescription)" }
                ?? error.localizedDescription
        }
    }
}
