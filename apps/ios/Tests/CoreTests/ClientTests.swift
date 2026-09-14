import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import XCTest
@testable import StarlingVoiceCore

final class ClientTests: XCTestCase {
    func testOpenAIRequestUsesStandardMultipartFields() throws {
        let file = try temporaryRecording()
        defer { try? FileManager.default.removeItem(at: file) }
        let client = StarlingClient(configuration: ServerConfiguration(
            endpoint: "https://voice.example.test",
            model: "parakeet",
            apiProtocol: .openAI
        ))
        let request = try client.makeRequest(recordingURL: file, requestID: "request-1")
        let body = String(decoding: request.httpBody ?? Data(), as: UTF8.self)

        XCTAssertEqual(request.url?.path, "/v1/audio/transcriptions")
        XCTAssertEqual(request.value(forHTTPHeaderField: "X-Request-Id"), "request-1")
        XCTAssertTrue(body.contains("name=\"file\"; filename=\"recording.wav\""))
        XCTAssertTrue(body.contains("name=\"model\"\r\n\r\nparakeet"))
        XCTAssertTrue(body.contains("name=\"response_format\"\r\n\r\njson"))
    }

    func testLegacyRequestOmitsOpenAIControlFields() throws {
        let file = try temporaryRecording()
        defer { try? FileManager.default.removeItem(at: file) }
        let client = StarlingClient(configuration: ServerConfiguration(
            endpoint: "https://voice.example.test",
            model: "ignored",
            apiProtocol: .starling
        ))
        let request = try client.makeRequest(recordingURL: file, requestID: "request-2")
        let body = String(decoding: request.httpBody ?? Data(), as: UTF8.self)

        XCTAssertEqual(request.url?.path, "/inference")
        XCTAssertFalse(body.contains("name=\"model\""))
        XCTAssertFalse(body.contains("name=\"response_format\""))
    }

    func testRawTextAndOptionalTimingAreDecodedWithoutRewriting() throws {
        let client = StarlingClient(configuration: ServerConfiguration())
        let data = Data(#"{"text":"  never remove like  ","segments":[{"text":"never","start_s":0,"end_s":0.2}],"duration_s":0.2}"#.utf8)
        let response = try XCTUnwrap(HTTPURLResponse(
            url: URL(string: "https://voice.example.test")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["X-Request-Id": "response-id"]
        ))
        let transcript = try client.decode(data: data, response: response)

        XCTAssertEqual(transcript.text, "  never remove like  ")
        XCTAssertEqual(transcript.segments, [.init(text: "never", startSeconds: 0, endSeconds: 0.2)])
        XCTAssertEqual(transcript.durationSeconds, 0.2)
        XCTAssertEqual(transcript.requestID, "response-id")
    }

    func testOpenAITextOnlyResponseDoesNotInventTiming() throws {
        let client = StarlingClient(configuration: ServerConfiguration())
        let response = try XCTUnwrap(HTTPURLResponse(
            url: URL(string: "https://voice.example.test")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: nil
        ))
        let transcript = try client.decode(data: Data(#"{"text":"agreed"}"#.utf8), response: response)
        XCTAssertEqual(transcript.text, "agreed")
        XCTAssertTrue(transcript.segments.isEmpty)
        XCTAssertNil(transcript.durationSeconds)
    }

    func testNestedOpenAIErrorMessageIsSurfaced() throws {
        let client = StarlingClient(configuration: ServerConfiguration())
        let response = try XCTUnwrap(HTTPURLResponse(
            url: URL(string: "https://voice.example.test")!,
            statusCode: 400,
            httpVersion: nil,
            headerFields: nil
        ))
        XCTAssertThrowsError(try client.decode(
            data: Data(#"{"error":{"message":"model is required"}}"#.utf8),
            response: response
        )) { error in
            XCTAssertEqual(error as? StarlingClientError, .http(status: 400, message: "model is required"))
        }
    }

    private func temporaryRecording() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        let wav = Data([
            82, 73, 70, 70, 38, 0, 0, 0, 87, 65, 86, 69,
            102, 109, 116, 32, 16, 0, 0, 0, 1, 0, 1, 0,
            128, 62, 0, 0, 0, 125, 0, 0, 2, 0, 16, 0,
            100, 97, 116, 97, 2, 0, 0, 0, 0, 0,
        ])
        try wav.write(to: url)
        return url
    }
}
