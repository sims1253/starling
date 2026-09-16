import Foundation
import XCTest
@testable import StarlingVoiceCore

final class SessionRepositoryTests: XCTestCase {
    func testRecordingSurvivesFailureRetryAndTranscriptPersistence() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let source = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: source)
        }
        let original = Data("RIFF-test-WAVE".utf8)
        try original.write(to: source)
        let repository = SessionRepository(rootURL: root)

        let captured = try await repository.createRecording(copying: source, durationMilliseconds: 50)
        _ = try await repository.markAttempt(captured.id)
        _ = try await repository.saveFailure(TestFailure(), for: captured.id)
        _ = try await repository.markAttempt(captured.id)
        let saved = try await repository.saveTranscript(
            Transcript(text: "  agreed  ", requestID: "request"),
            for: captured.id
        )

        XCTAssertEqual(saved.attemptCount, 2)
        XCTAssertEqual(saved.transcript?.text, "  agreed  ")
        let recordingURL = await repository.recordingURL(for: saved)
        XCTAssertEqual(try Data(contentsOf: recordingURL), original)

        let reopened = SessionRepository(rootURL: root)
        let restored = try await reopened.get(saved.id)
        XCTAssertEqual(restored.transcript?.text, "  agreed  ")
        let restoredRecordingURL = await reopened.recordingURL(for: restored)
        XCTAssertTrue(FileManager.default.fileExists(atPath: restoredRecordingURL.path))

        try await reopened.delete(saved.id)
        let afterDelete = try await reopened.list()
        XCTAssertTrue(afterDelete.sessions.isEmpty)
    }

    func testPromotionReclaimsCrashLeftoverEmptyDestinationDirectory() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = SessionRepository(rootURL: root)
        let id = UUID()
        let staged = try await repository.stagingRecordingURL(id: id)
        let audio = Data("leftover empty destination".utf8)
        try audio.write(to: staged)
        // A process exit between creating the destination directory and
        // copying the WAV leaves that destination empty.
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent(id.uuidString, isDirectory: true),
            withIntermediateDirectories: true
        )

        let record = try await repository.commitStagedRecording(at: staged)

        XCTAssertEqual(record.id, id)
        let durableURL = await repository.recordingURL(for: record)
        XCTAssertEqual(try Data(contentsOf: durableURL), audio)
        XCTAssertFalse(FileManager.default.fileExists(atPath: staged.path))
    }

    func testPromotionFailureLeavesStagedAudioRecoverable() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = SessionRepository(rootURL: root)
        let id = UUID()
        let staged = try await repository.stagingRecordingURL(id: id)
        let audio = Data("retained audio".utf8)
        try audio.write(to: staged)
        // Unrelated content in the destination must not be deleted, so it
        // stays an obstruction instead of being reclaimed.
        let obstruction = root.appendingPathComponent(id.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: obstruction, withIntermediateDirectories: true)
        let strayFile = obstruction.appendingPathComponent("notes.txt")
        try Data("not ours".utf8).write(to: strayFile)

        do {
            _ = try await repository.commitStagedRecording(at: staged)
            XCTFail("promotion should fail while the destination is obstructed")
        } catch {
            XCTAssertEqual(try Data(contentsOf: staged), audio)
            XCTAssertEqual(try Data(contentsOf: strayFile), Data("not ours".utf8))
        }
    }

    func testListSkipsDamagedDirectoriesInsteadOfFailing() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let source = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: source)
        }
        try Data("audio".utf8).write(to: source)
        let repository = SessionRepository(rootURL: root)
        let first = try await repository.createRecording(copying: source)
        let second = try await repository.createRecording(copying: source)

        // A directory without a manifest.
        let emptyID = UUID()
        let emptyDirectory = root.appendingPathComponent(emptyID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: emptyDirectory, withIntermediateDirectories: true)
        // A directory with an undecodable manifest.
        let corruptID = UUID()
        let corruptDirectory = root.appendingPathComponent(corruptID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: corruptDirectory, withIntermediateDirectories: true)
        try Data("{ not a manifest".utf8)
            .write(to: corruptDirectory.appendingPathComponent("manifest.json"))
        // A manifest whose audio file is gone.
        let missingAudio = try await repository.createRecording(copying: source)
        try FileManager.default.removeItem(at: await repository.recordingURL(for: missingAudio))

        let listing = try await repository.list()

        XCTAssertEqual(Set(listing.sessions.map(\.id)), Set([first.id, second.id]))
        XCTAssertEqual(
            Set(listing.damagedDirectories.map(\.name)),
            Set([emptyID.uuidString, corruptID.uuidString, missingAudio.id.uuidString])
        )
        XCTAssertTrue(listing.damagedDirectories.allSatisfy { !$0.reason.isEmpty })
        // Skipped directories and their files stay on disk.
        XCTAssertTrue(FileManager.default.fileExists(atPath: emptyDirectory.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: corruptDirectory.path))
        // Healthy sessions remain openable.
        let opened = try await repository.get(second.id)
        XCTAssertEqual(opened.id, second.id)
    }

    func testPendingRecoveryContinuesPastUnrecoverableFiles() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let first = SessionRepository(rootURL: root)
        let blocked = UUID()
        let blockedStaged = try await first.stagingRecordingURL(id: blocked)
        try Data("blocked".utf8).write(to: blockedStaged)
        // Unrelated destination content obstructs this one promotion forever.
        let obstruction = root.appendingPathComponent(blocked.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: obstruction, withIntermediateDirectories: true)
        try Data("not ours".utf8).write(to: obstruction.appendingPathComponent("notes.txt"))
        let healthy = UUID()
        let healthyStaged = try await first.stagingRecordingURL(id: healthy)
        let audio = Data("still promoted".utf8)
        try audio.write(to: healthyStaged)

        let relaunched = SessionRepository(rootURL: root)
        do {
            _ = try await relaunched.recoverPendingRecordings()
            XCTFail("the obstructed promotion should still be reported")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains(blockedStaged.lastPathComponent))
        }

        let listing = try await relaunched.list()
        XCTAssertEqual(listing.sessions.map(\.id), [healthy])
        let durableURL = await relaunched.recordingURL(for: healthy)
        XCTAssertEqual(try Data(contentsOf: durableURL), audio)
        // The blocked staged audio stays recoverable instead of vanishing.
        XCTAssertEqual(try Data(contentsOf: blockedStaged), Data("blocked".utf8))
    }

    func testRelaunchReconcilesInterruptedTranscribingSessions() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let source = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: source)
        }
        try Data("audio".utf8).write(to: source)
        let repository = SessionRepository(rootURL: root)
        // A request that never finished: .transcribing persisted before the exit.
        let interrupted = try await repository.createRecording(copying: source)
        _ = try await repository.markAttempt(interrupted.id)
        // A retry interrupted after an earlier success keeps its transcript.
        let retrying = try await repository.createRecording(copying: source)
        _ = try await repository.markAttempt(retrying.id)
        _ = try await repository.saveTranscript(Transcript(text: "first raw"), for: retrying.id)
        _ = try await repository.markAttempt(retrying.id)
        // Finished sessions are not reconciliation's business.
        let completed = try await repository.createRecording(copying: source)
        _ = try await repository.saveTranscript(Transcript(text: "done"), for: completed.id)

        // A fresh process has no request in flight for the interrupted ones.
        let relaunched = SessionRepository(rootURL: root)
        let reconciled = try await relaunched.reconcileInterruptedTranscriptions()
        XCTAssertEqual(Set(reconciled.map(\.id)), Set([interrupted.id, retrying.id]))

        let restored = try await relaunched.get(interrupted.id)
        XCTAssertEqual(restored.status, .failed)
        XCTAssertEqual(restored.attemptCount, 1)
        XCTAssertEqual(restored.lastError, "Interrupted before the server returned a transcript. Your audio is ready to retry.")
        let restoredRetrying = try await relaunched.get(retrying.id)
        XCTAssertEqual(restoredRetrying.status, .failed)
        XCTAssertEqual(restoredRetrying.attemptCount, 2)
        XCTAssertEqual(restoredRetrying.transcript?.text, "first raw")
        let unaffected = try await relaunched.get(completed.id)
        XCTAssertEqual(unaffected.status, .transcribed)

        // A second startup pass finds nothing left to reconcile.
        let again = SessionRepository(rootURL: root)
        let empty = try await again.reconcileInterruptedTranscriptions()
        XCTAssertTrue(empty.isEmpty)
    }

    func testRelaunchRecoversAfterAudioCopyBeforeManifest() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = SessionRepository(rootURL: root)
        let id = UUID()
        let staged = try await repository.stagingRecordingURL(id: id)
        let audio = Data("copied before manifest".utf8)
        try audio.write(to: staged)

        // Simulate a process exit after the destination audio copy succeeds.
        let orphanDirectory = root.appendingPathComponent(id.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: orphanDirectory, withIntermediateDirectories: true)
        try audio.write(to: orphanDirectory.appendingPathComponent("recording.wav"))

        let relaunched = SessionRepository(rootURL: root)
        let recovered = try await relaunched.recoverPendingRecordings()
        let record = try XCTUnwrap(recovered.first)
        XCTAssertEqual(record.id, id)
        let durableURL = await relaunched.recordingURL(for: record)
        XCTAssertEqual(try Data(contentsOf: durableURL), audio)
        XCTAssertFalse(FileManager.default.fileExists(atPath: staged.path))
    }

    func testSuccessfulRetryPreservesThePreviousRawTranscript() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let source = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: source)
        }
        try Data("audio".utf8).write(to: source)
        let repository = SessionRepository(rootURL: root)
        let record = try await repository.createRecording(copying: source)
        _ = try await repository.saveTranscript(Transcript(text: "first raw"), for: record.id)
        let retried = try await repository.saveTranscript(Transcript(text: "second raw"), for: record.id)

        XCTAssertEqual(retried.transcript?.text, "second raw")
        XCTAssertEqual(retried.transcriptHistory.map(\.text), ["first raw"])
        let reopened = SessionRepository(rootURL: root)
        let restored = try await reopened.get(record.id)
        XCTAssertEqual(restored.transcriptHistory.map(\.text), ["first raw"])
    }

    func testPendingRecordingIsRecoveredOnRelaunch() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let first = SessionRepository(rootURL: root)
        let staged = try await first.stagingRecordingURL()
        let audio = Data("closed wav bytes".utf8)
        try audio.write(to: staged)

        let relaunched = SessionRepository(rootURL: root)
        let recovered = try await relaunched.recoverPendingRecordings()
        XCTAssertEqual(recovered.count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: staged.path))
        let durableURL = await relaunched.recordingURL(for: try XCTUnwrap(recovered.first))
        XCTAssertEqual(try Data(contentsOf: durableURL), audio)
    }
}

private struct TestFailure: LocalizedError {
    var errorDescription: String? { "network unavailable" }
}
