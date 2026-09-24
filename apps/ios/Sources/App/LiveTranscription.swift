import Foundation
import StarlingVoiceCore

/// One WebSocket per recording. The reader stays active while the microphone
/// sends frames, so partials appear before the stop button is pressed.
@MainActor
final class LiveTranscription {
    private let session: URLSession
    private let socket: URLSessionWebSocketTask
    private var reader: Task<Transcript, Error>?

    init(configuration: ServerConfiguration, onPartial: @escaping @MainActor (String) -> Void) throws {
        let base = try configuration.validatedBaseURL()
        guard var url = URLComponents(url: base, resolvingAgainstBaseURL: false) else {
            throw ConfigurationError.invalidEndpoint
        }
        url.scheme = url.scheme?.lowercased() == "https" ? "wss" : "ws"
        var path = url.path
        while path.hasSuffix("/") { path.removeLast() }
        if path.lowercased().hasSuffix("/v1/audio/transcriptions") {
            path.removeLast("/v1/audio/transcriptions".count)
        } else if path.lowercased().hasSuffix("/v1") {
            path.removeLast("/v1".count)
        }
        url.path = path + "/stream"
        guard let endpoint = url.url else { throw ConfigurationError.invalidEndpoint }
        let options = URLSessionConfiguration.ephemeral
        options.httpShouldSetCookies = false
        options.httpCookieAcceptPolicy = .never
        session = URLSession(
            configuration: options,
            delegate: StreamRedirectBlocker(),
            delegateQueue: nil
        )
        socket = session.webSocketTask(with: endpoint)
        socket.resume()
        let socket = self.socket
        reader = Task {
            do {
                while true {
                    let message = try await socket.receive()
                    // A non-text frame, invalid JSON, or a message this
                    // build does not understand is skipped, not fatal:
                    // killing the reader over one malformed frame would
                    // silently end partials while the pump keeps feeding
                    // the socket. Only a "final" without text (the one
                    // payload we are waiting to complete on) stays an
                    // error.
                    guard case let .string(json) = message,
                          let data = json.data(using: .utf8),
                          let payload = try? JSONDecoder().decode(StreamMessage.self, from: data)
                    else { continue }
                    switch payload.type {
                    case "partial":
                        onPartial(payload.text ?? "")
                    case "final":
                        guard let text = payload.text else { throw StreamFailure.invalidResponse }
                        let segments = (payload.segments ?? []).map {
                            TranscriptSegment(text: $0.text, startSeconds: $0.start, endSeconds: $0.end)
                        }
                        return Transcript(text: text, segments: segments, durationSeconds: payload.duration)
                    case "error":
                        throw StreamFailure.server(payload.message ?? "Unknown stream error")
                    default:
                        break
                    }
                }
            } catch {
                // The reader is the only side that hears a dead stream.
                // Cancel the socket so the send path (and finish()) see
                // the failure on their next call instead of feeding a
                // connection nobody is reading.
                socket.cancel(with: .normalClosure, reason: nil)
                throw error
            }
        }
    }

    func send(_ frames: [Data]) async throws {
        for frame in frames where !frame.isEmpty {
            try await sendMessage(.data(frame))
        }
    }

    func finish() async throws -> Transcript {
        defer { close() }
        try await sendMessage(.string(#"{"type":"commit"}"#))
        guard let reader else { throw StreamFailure.invalidResponse }
        return try await withThrowingTaskGroup(of: Transcript.self) { group in
            group.addTask { try await reader.value }
            group.addTask {
                try await Task.sleep(for: .seconds(120))
                reader.cancel()
                throw StreamFailure.timeout
            }
            let value = try await group.next()!
            group.cancelAll()
            return value
        }
    }

    func close() {
        reader?.cancel()
        socket.cancel(with: .normalClosure, reason: nil)
        session.invalidateAndCancel()
    }

    private func sendMessage(_ message: URLSessionWebSocketTask.Message) async throws {
        let socket = self.socket
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try await socket.send(message) }
            group.addTask {
                try await Task.sleep(for: .seconds(10))
                socket.cancel(with: .normalClosure, reason: nil)
                throw StreamFailure.timeout
            }
            _ = try await group.next()
            group.cancelAll()
        }
    }
}

private final class StreamRedirectBlocker: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
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

private enum StreamFailure: LocalizedError {
    case invalidResponse
    case server(String)
    case timeout

    var errorDescription: String? {
        switch self {
        case .invalidResponse: "The stream returned an invalid response."
        case let .server(message): message
        case .timeout: "The stream did not finish in time."
        }
    }
}

private struct StreamMessage: Decodable {
    let type: String
    let text: String?
    let message: String?
    let segments: [StreamSegment]?
    let duration: Double?

    enum CodingKeys: String, CodingKey {
        case type, text, message, segments
        case duration = "duration_s"
    }
}

private struct StreamSegment: Decodable {
    let text: String
    let start: Double
    let end: Double

    enum CodingKeys: String, CodingKey {
        case text
        case start = "start_s"
        case end = "end_s"
    }
}
