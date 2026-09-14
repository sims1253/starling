import Foundation

public enum SessionRepositoryError: LocalizedError, Equatable {
    case missingRecording(UUID)
    case missingSession(UUID)
    case unsupportedSchema(Int)

    public var errorDescription: String? {
        switch self {
        case .missingRecording: "The saved recording is missing."
        case .missingSession: "The saved dictation session is missing."
        case let .unsupportedSchema(version): "Session schema version \(version) is not supported."
        }
    }
}

/// File-backed session history. Every mutating method returns only after its
/// manifest has been written atomically.
public actor SessionRepository {
    private let fileManager: FileManager
    public let rootURL: URL
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    public init(rootURL: URL? = nil, fileManager: FileManager = .default) {
        self.fileManager = fileManager
        if let rootURL {
            self.rootURL = rootURL
        } else {
            let applicationSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
                ?? fileManager.temporaryDirectory
            self.rootURL = applicationSupport
                .appendingPathComponent("StarlingVoice", isDirectory: true)
                .appendingPathComponent("Sessions", isDirectory: true)
        }
        encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        try? fileManager.createDirectory(at: self.rootURL, withIntermediateDirectories: true)
    }

    public func createRecording(
        copying sourceURL: URL,
        durationMilliseconds: Int? = nil,
        id: UUID = UUID(),
        createdAt: Date = Date()
    ) throws -> SessionRecord {
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw SessionRepositoryError.missingRecording(id)
        }
        try fileManager.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let directory = directoryURL(for: id)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: false)
        do {
            let record = SessionRecord(
                id: id,
                createdAt: createdAt,
                durationMilliseconds: durationMilliseconds
            )
            try fileManager.copyItem(at: sourceURL, to: recordingURL(for: record))
            try write(record)
#if canImport(Darwin)
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var mutableDirectory = directory
            try? mutableDirectory.setResourceValues(values)
#endif
#if os(iOS)
            try? fileManager.setAttributes(
                [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
                ofItemAtPath: directory.path
            )
#endif
            return record
        } catch {
            try? fileManager.removeItem(at: directory)
            throw error
        }
    }

    /// Allocate an app-private destination before AVAudioRecorder starts.
    public func stagingRecordingURL(id: UUID = UUID()) throws -> URL {
        let pending = pendingURL
        try fileManager.createDirectory(at: pending, withIntermediateDirectories: true)
        return pending.appendingPathComponent(id.uuidString + ".wav", isDirectory: false)
    }

    /// Promote a completed staged recording into durable history. The staged
    /// file is removed only after both the audio copy and manifest succeed.
    public func commitStagedRecording(
        at stagedURL: URL,
        durationMilliseconds: Int? = nil
    ) throws -> SessionRecord {
        let id = UUID(uuidString: stagedURL.deletingPathExtension().lastPathComponent) ?? UUID()
        let existingManifest = manifestURL(for: id)
        if fileManager.fileExists(atPath: existingManifest.path) {
            let existing = try get(id)
            try? fileManager.removeItem(at: stagedURL)
            return existing
        }
        guard fileManager.fileExists(atPath: stagedURL.path) else {
            throw SessionRepositoryError.missingRecording(id)
        }
        // A process can exit after copying the WAV into its destination
        // directory but before writing the manifest. Keep the staged source,
        // remove that incomplete destination, and repeat the promotion.
        // Empty or unrelated destination directories remain an obstruction so
        // an accidental collision does not remove user data.
        let destinationDirectory = directoryURL(for: id)
        let destinationAudio = recordingURL(for: SessionRecord(id: id))
        if fileManager.fileExists(atPath: destinationDirectory.path) {
            guard fileManager.fileExists(atPath: destinationAudio.path) else {
                throw CocoaError(.fileWriteFileExists, userInfo: [NSFilePathErrorKey: destinationDirectory.path])
            }
            try fileManager.removeItem(at: destinationDirectory)
        }
        let record = try createRecording(
            copying: stagedURL,
            durationMilliseconds: durationMilliseconds,
            id: id
        )
        // Failure to remove a now-duplicated staging file must not hide the
        // durable session or prevent transcription. Relaunch recovery retries.
        try? fileManager.removeItem(at: stagedURL)
        return record
    }

    /// Recover recordings that reached app-private storage before an app exit
    /// but were not yet promoted into history.
    @discardableResult
    public func recoverPendingRecordings() throws -> [SessionRecord] {
        guard fileManager.fileExists(atPath: pendingURL.path) else { return [] }
        let files = try fileManager.contentsOfDirectory(
            at: pendingURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        var recovered: [SessionRecord] = []
        for file in files where file.pathExtension.lowercased() == "wav" {
            recovered.append(try commitStagedRecording(at: file))
        }
        return recovered
    }

    public func list() throws -> [SessionRecord] {
        if !fileManager.fileExists(atPath: rootURL.path) {
            try fileManager.createDirectory(at: rootURL, withIntermediateDirectories: true)
            return []
        }
        let keys: [URLResourceKey] = [.isDirectoryKey]
        let directories = try fileManager.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        )
        return try directories.compactMap { directory in
            guard directory.lastPathComponent != "Pending" else { return nil }
            guard (try? directory.resourceValues(forKeys: Set(keys)).isDirectory) == true else { return nil }
            return try readManifest(in: directory)
        }
        .sorted { $0.updatedAt > $1.updatedAt }
    }

    public func get(_ id: UUID) throws -> SessionRecord {
        let manifest = manifestURL(for: id)
        guard fileManager.fileExists(atPath: manifest.path) else {
            throw SessionRepositoryError.missingSession(id)
        }
        return try readManifest(in: directoryURL(for: id))
    }

    @discardableResult
    public func markAttempt(_ id: UUID) throws -> SessionRecord {
        try update(id) { record in
            record.status = .transcribing
            record.attemptCount += 1
            record.lastError = nil
        }
    }

    @discardableResult
    public func saveTranscript(_ transcript: Transcript, for id: UUID) throws -> SessionRecord {
        try update(id) { record in
            record.status = .transcribed
            if let previous = record.transcript {
                record.transcriptHistory.append(previous)
            }
            record.transcript = transcript
            record.lastError = nil
        }
    }

    @discardableResult
    public func saveFailure(_ error: any Error, for id: UUID) throws -> SessionRecord {
        try update(id) { record in
            record.status = .failed
            record.lastError = error.localizedDescription
        }
    }

    public func delete(_ id: UUID) throws {
        let directory = directoryURL(for: id)
        guard fileManager.fileExists(atPath: directory.path) else { return }
        try fileManager.removeItem(at: directory)
    }

    public func recordingURL(for record: SessionRecord) -> URL {
        directoryURL(for: record.id).appendingPathComponent(record.audioFilename, isDirectory: false)
    }

    public func transcriptExportURL(for record: SessionRecord) throws -> URL? {
        guard let text = record.transcript?.text else { return nil }
        let url = directoryURL(for: record.id).appendingPathComponent("transcript.txt")
        try Data(text.utf8).write(to: url, options: .atomic)
        return url
    }

    private func update(_ id: UUID, mutation: (inout SessionRecord) -> Void) throws -> SessionRecord {
        var record = try get(id)
        mutation(&record)
        record.updatedAt = Date()
        try write(record)
        return record
    }

    private func readManifest(in directory: URL) throws -> SessionRecord {
        let data = try Data(contentsOf: directory.appendingPathComponent("manifest.json"))
        let record = try decoder.decode(SessionRecord.self, from: data)
        guard record.schemaVersion == 1 else {
            throw SessionRepositoryError.unsupportedSchema(record.schemaVersion)
        }
        guard fileManager.fileExists(atPath: recordingURL(for: record).path) else {
            throw SessionRepositoryError.missingRecording(record.id)
        }
        return record
    }

    private func write(_ record: SessionRecord) throws {
        let data = try encoder.encode(record)
        try data.write(to: manifestURL(for: record.id), options: .atomic)
    }

    private func directoryURL(for id: UUID) -> URL {
        rootURL.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    private func manifestURL(for id: UUID) -> URL {
        directoryURL(for: id).appendingPathComponent("manifest.json", isDirectory: false)
    }

    private var pendingURL: URL {
        rootURL.appendingPathComponent("Pending", isDirectory: true)
    }
}
