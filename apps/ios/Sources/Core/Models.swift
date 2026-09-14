import Foundation

public enum ServerProtocol: String, Codable, CaseIterable, Sendable {
    case openAI = "openai"
    case starling

    public var title: String {
        switch self {
        case .openAI: "OpenAI compatible"
        case .starling: "Starling legacy"
        }
    }

    public var route: String {
        switch self {
        case .openAI: "/v1/audio/transcriptions"
        case .starling: "/inference"
        }
    }
}

public struct ServerConfiguration: Codable, Equatable, Sendable {
    public var endpoint: String
    public var model: String
    public var apiProtocol: ServerProtocol
    public var allowsInsecureLocalHTTP: Bool

    public init(
        endpoint: String = "https://starling.local:8181",
        model: String = "parakeet",
        apiProtocol: ServerProtocol = .openAI,
        allowsInsecureLocalHTTP: Bool = false
    ) {
        self.endpoint = endpoint
        self.model = model
        self.apiProtocol = apiProtocol
        self.allowsInsecureLocalHTTP = allowsInsecureLocalHTTP
    }

    public func validatedBaseURL() throws -> URL {
        guard let components = URLComponents(string: endpoint),
              let scheme = components.scheme?.lowercased(),
              let host = components.host,
              !host.isEmpty,
              components.user == nil,
              components.password == nil,
              components.query == nil,
              components.fragment == nil
        else {
            throw ConfigurationError.invalidEndpoint
        }
        guard scheme == "https" || scheme == "http" else {
            throw ConfigurationError.unsupportedScheme
        }
        if scheme == "http" {
            guard allowsInsecureLocalHTTP else {
                throw ConfigurationError.insecureHTTPDisabled
            }
            guard Self.isLocalHost(host) else {
                throw ConfigurationError.insecureHTTPRequiresLocalHost
            }
        }
        let cleanModel = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard apiProtocol != .openAI || (!cleanModel.isEmpty && !model.contains("\r") && !model.contains("\n")) else {
            throw ConfigurationError.emptyModel
        }
        guard let url = components.url else { throw ConfigurationError.invalidEndpoint }
        return url
    }

    public func transcriptionURL() throws -> URL {
        let base = try validatedBaseURL()
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false) else {
            throw ConfigurationError.invalidEndpoint
        }
        var path = components.path
        while path.hasSuffix("/") { path.removeLast() }
        let normalizedPath = path.lowercased()

        switch apiProtocol {
        case .openAI:
            if normalizedPath.hasSuffix("/audio/transcriptions") {
                components.path = path
                guard let url = components.url else { throw ConfigurationError.invalidEndpoint }
                return url
            }
            if normalizedPath == "/v1" || normalizedPath.hasSuffix("/v1") {
                path += "/audio/transcriptions"
            } else {
                path += apiProtocol.route
            }
        case .starling:
            if normalizedPath.hasSuffix("/inference") || normalizedPath.hasSuffix("/transcribe") {
                components.path = path
                guard let url = components.url else { throw ConfigurationError.invalidEndpoint }
                return url
            }
            path += apiProtocol.route
        }

        components.path = path
        guard let url = components.url else { throw ConfigurationError.invalidEndpoint }
        return url
    }

    public static func isLocalHost(_ rawHost: String) -> Bool {
        let host = rawHost.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        if host == "localhost" || host == "::1" || host.hasSuffix(".local") {
            return true
        }
        let components = host.split(separator: ".", omittingEmptySubsequences: false)
        let octets = components.compactMap { Int($0) }
        if components.count == 4, octets.count == 4, octets.allSatisfy({ (0 ... 255).contains($0) }) {
            if octets[0] == 10 || (octets[0] == 127) || (octets[0] == 192 && octets[1] == 168) || (octets[0] == 169 && octets[1] == 254) {
                return true
            }
            if octets[0] == 172, (16 ... 31).contains(octets[1]) {
                return true
            }
        }
        return host.contains(":") && (host.hasPrefix("fc") || host.hasPrefix("fd") || host.hasPrefix("fe80:"))
    }
}

public enum ConfigurationError: LocalizedError, Equatable {
    case invalidEndpoint
    case unsupportedScheme
    case insecureHTTPDisabled
    case insecureHTTPRequiresLocalHost
    case emptyModel

    public var errorDescription: String? {
        switch self {
        case .invalidEndpoint: "Enter a complete server URL without credentials."
        case .unsupportedScheme: "The server URL must use HTTPS or HTTP."
        case .insecureHTTPDisabled: "Enable local HTTP explicitly or use HTTPS."
        case .insecureHTTPRequiresLocalHost: "Plain HTTP is limited to local and private network hosts."
        case .emptyModel: "Enter a model name without line breaks."
        }
    }
}

public struct TranscriptSegment: Codable, Equatable, Sendable {
    public let text: String
    public let startSeconds: Double
    public let endSeconds: Double

    public init(text: String, startSeconds: Double, endSeconds: Double) {
        self.text = text
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
    }
}

public struct Transcript: Codable, Equatable, Sendable {
    /// Exact server text. No trimming, filler removal, or normalization occurs.
    public let text: String
    public let segments: [TranscriptSegment]
    public let durationSeconds: Double?
    public let requestID: String?

    public init(
        text: String,
        segments: [TranscriptSegment] = [],
        durationSeconds: Double? = nil,
        requestID: String? = nil
    ) {
        self.text = text
        self.segments = segments
        self.durationSeconds = durationSeconds
        self.requestID = requestID
    }
}

public enum SessionStatus: String, Codable, Sendable {
    case captured
    case transcribing
    case transcribed
    case failed
}

public struct SessionRecord: Codable, Equatable, Identifiable, Sendable {
    public let schemaVersion: Int
    public let id: UUID
    public let createdAt: Date
    public var updatedAt: Date
    public var status: SessionStatus
    public let audioFilename: String
    public let durationMilliseconds: Int?
    public var attemptCount: Int
    public var transcript: Transcript?
    public var transcriptHistory: [Transcript]
    public var lastError: String?

    public init(
        id: UUID = UUID(),
        createdAt: Date = Date(),
        durationMilliseconds: Int? = nil
    ) {
        schemaVersion = 1
        self.id = id
        self.createdAt = createdAt
        updatedAt = createdAt
        status = .captured
        audioFilename = "recording.wav"
        self.durationMilliseconds = durationMilliseconds
        attemptCount = 0
        transcriptHistory = []
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion, id, createdAt, updatedAt, status, audioFilename
        case durationMilliseconds, attemptCount, transcript, transcriptHistory, lastError
    }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try values.decode(Int.self, forKey: .schemaVersion)
        id = try values.decode(UUID.self, forKey: .id)
        createdAt = try values.decode(Date.self, forKey: .createdAt)
        updatedAt = try values.decode(Date.self, forKey: .updatedAt)
        status = try values.decode(SessionStatus.self, forKey: .status)
        audioFilename = try values.decode(String.self, forKey: .audioFilename)
        durationMilliseconds = try values.decodeIfPresent(Int.self, forKey: .durationMilliseconds)
        attemptCount = try values.decode(Int.self, forKey: .attemptCount)
        transcript = try values.decodeIfPresent(Transcript.self, forKey: .transcript)
        transcriptHistory = try values.decodeIfPresent([Transcript].self, forKey: .transcriptHistory) ?? []
        lastError = try values.decodeIfPresent(String.self, forKey: .lastError)
    }
}
