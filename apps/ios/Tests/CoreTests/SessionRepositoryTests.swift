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
        XCTAssertTrue(afterDelete.isEmpty)
    }

    func testPromotionFailureLeavesStagedAudioRecoverable() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = SessionRepository(rootURL: root)
        let id = UUID()
        let staged = try await repository.stagingRecordingURL(id: id)
        let audio = Data("retained audio".utf8)
        try audio.write(to: staged)
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent(id.uuidString, isDirectory: true),
            withIntermediateDirectories: true
        )

        do {
            _ = try await repository.commitStagedRecording(at: staged)
            XCTFail("promotion should fail while the destination is obstructed")
        } catch {
            XCTAssertEqual(try Data(contentsOf: staged), audio)
        }
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
