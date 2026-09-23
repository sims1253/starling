import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public enum StarlingClientError: LocalizedError, Equatable {
    case invalidResponse
    case invalidPayload(String)
    case http(status: Int, message: String)

    public var errorDescription: String? {
        switch self {
        case .invalidResponse: "The server returned an invalid HTTP response."
        case let .invalidPayload(message): "The server response is invalid: \(message)"
        case let .http(status, message): "Server error \(status): \(message)"
        }
    }
}

public struct StarlingClient: Sendable {
    public let configuration: ServerConfiguration
    public let timeout: TimeInterval

    public init(configuration: ServerConfiguration, timeout: TimeInterval = 120) {
        self.configuration = configuration
        self.timeout = timeout
    }

    public func transcribe(recordingURL: URL, requestID: String = UUID().uuidString) async throws -> Transcript {
        let request = try makeRequest(recordingURL: recordingURL, requestID: requestID)
        let sessionConfiguration = URLSessionConfiguration.ephemeral
        sessionConfiguration.httpShouldSetCookies = false
        sessionConfiguration.httpCookieAcceptPolicy = .never
        let session = URLSession(
            configuration: sessionConfiguration,
            delegate: RedirectBlocker(),
            delegateQueue: nil
        )
        defer { session.finishTasksAndInvalidate() }
        let (data, response) = try await session.data(for: request)
        return try decode(data: data, response: response)
    }

    public func makeRequest(recordingURL: URL, requestID: String) throws -> URLRequest {
        let wav = try Data(contentsOf: recordingURL, options: [.mappedIfSafe])
        guard !wav.isEmpty else { throw StarlingClientError.invalidPayload("recording is empty") }
        do {
            try validateStarlingWAV(wav)
        } catch {
            throw StarlingClientError.invalidPayload(error.localizedDescription)
        }
        guard !requestID.isEmpty, !requestID.hasPrefix("#"), !requestID.contains("\r"), !requestID.contains("\n") else {
            throw StarlingClientError.invalidPayload("request ID is not accepted by the native server")
        }

        let boundary = "StarlingBoundary-\(UUID().uuidString)"
        var body = Data()
        body.appendMultipart(
            name: "file",
            filename: "recording.wav",
            contentType: "audio/wav",
            data: wav,
            boundary: boundary
        )
        body.appendMultipart(
            name: "model",
            value: configuration.model.trimmingCharacters(in: .whitespacesAndNewlines),
            boundary: boundary
        )
        body.appendMultipart(name: "response_format", value: "json", boundary: boundary)
        body.append(Data("--\(boundary)--\r\n".utf8))

        var request = URLRequest(url: try configuration.transcriptionURL())
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(requestID, forHTTPHeaderField: "X-Request-Id")
        request.httpBody = body
        return request
    }

    public func decode(data: Data, response: URLResponse) throws -> Transcript {
        guard let http = response as? HTTPURLResponse else { throw StarlingClientError.invalidResponse }
        guard (200 ..< 300).contains(http.statusCode) else {
            let serverError = try? JSONDecoder().decode(ServerErrorPayload.self, from: data)
            let body = String(data: data.prefix(4_096), encoding: .utf8)
            let message = serverError?.message ?? body ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            throw StarlingClientError.http(status: http.statusCode, message: message)
        }
        let payload: ResponsePayload
        do {
            payload = try JSONDecoder().decode(ResponsePayload.self, from: data)
        } catch {
            throw StarlingClientError.invalidPayload(error.localizedDescription)
        }
        let segments = try (payload.segments ?? []).map { segment in
            guard segment.start.isFinite, segment.end.isFinite,
                  segment.start >= 0, segment.end >= segment.start
            else { throw StarlingClientError.invalidPayload("segment timestamps are invalid") }
            return TranscriptSegment(
                text: segment.text,
                startSeconds: segment.start,
                endSeconds: segment.end
            )
        }
        let duration = payload.duration
        if let duration, (!duration.isFinite || duration < 0) {
            throw StarlingClientError.invalidPayload("duration is invalid")
        }
        let headerRequestID = http.value(forHTTPHeaderField: "X-Request-Id")
        return Transcript(
            text: payload.text,
            segments: segments,
            durationSeconds: duration,
            requestID: payload.requestID ?? headerRequestID
        )
    }
}

private struct ResponsePayload: Decodable {
    let text: String
    let segments: [SegmentPayload]?
    let duration: Double?
    let requestID: String?

    enum CodingKeys: String, CodingKey {
        case text, segments, duration
        case durationSeconds = "duration_s"
        case requestID = "request_id"
    }

    init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        text = try values.decode(String.self, forKey: .text)
        segments = try values.decodeIfPresent([SegmentPayload].self, forKey: .segments)
        duration = try values.decodeIfPresent(Double.self, forKey: .durationSeconds)
            ?? values.decodeIfPresent(Double.self, forKey: .duration)
        requestID = try values.decodeIfPresent(String.self, forKey: .requestID)
    }
}

private struct SegmentPayload: Decodable {
    let text: String
    let start: Double
    let end: Double

    enum CodingKeys: String, CodingKey {
        case text, start, end
        case startSeconds = "start_s"
        case endSeconds = "end_s"
    }

    init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        text = try values.decode(String.self, forKey: .text)
        if let seconds = try values.decodeIfPresent(Double.self, forKey: .startSeconds) {
            start = seconds
        } else {
            start = try values.decode(Double.self, forKey: .start)
        }
        if let seconds = try values.decodeIfPresent(Double.self, forKey: .endSeconds) {
            end = seconds
        } else {
            end = try values.decode(Double.self, forKey: .end)
        }
    }
}

private struct ServerErrorPayload: Decodable {
    let message: String?

    enum CodingKeys: String, CodingKey { case error, detail, message }

    init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        if let direct = try? values.decode(String.self, forKey: .error) {
            message = direct
        } else if let nested = try? values.decode(NestedError.self, forKey: .error) {
            message = nested.message
        } else {
            message = try values.decodeIfPresent(String.self, forKey: .detail)
                ?? values.decodeIfPresent(String.self, forKey: .message)
        }
    }

    private struct NestedError: Decodable { let message: String }
}

private final class RedirectBlocker: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

private extension Data {
    mutating func appendMultipart(name: String, value: String, boundary: String) {
        append(Data("--\(boundary)\r\n".utf8))
        append(Data("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".utf8))
        append(Data(value.utf8))
        append(Data("\r\n".utf8))
    }

    mutating func appendMultipart(
        name: String,
        filename: String,
        contentType: String,
        data: Data,
        boundary: String
    ) {
        append(Data("--\(boundary)\r\n".utf8))
        append(Data("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n".utf8))
        append(Data("Content-Type: \(contentType)\r\n\r\n".utf8))
        append(data)
        append(Data("\r\n".utf8))
    }
}
