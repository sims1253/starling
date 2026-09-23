import XCTest
@testable import StarlingVoiceCore

final class ConfigurationTests: XCTestCase {
    func testHTTPSIsAcceptedByDefault() throws {
        let configuration = ServerConfiguration(endpoint: "https://voice.example.test/base")
        XCTAssertEqual(try configuration.transcriptionURL().absoluteString, "https://voice.example.test/base/v1/audio/transcriptions")
    }

    func testHTTPRequiresExplicitOptIn() {
        let configuration = ServerConfiguration(endpoint: "http://192.168.1.20:8181")
        XCTAssertThrowsError(try configuration.validatedBaseURL()) { error in
            XCTAssertEqual(error as? ConfigurationError, .insecureHTTPDisabled)
        }
    }

    func testHTTPOptInRemainsLimitedToLocalNetworks() throws {
        let local = ServerConfiguration(
            endpoint: "http://192.168.1.20:8181",
            allowsInsecureLocalHTTP: true
        )
        XCTAssertEqual(try local.validatedBaseURL().host, "192.168.1.20")

        let publicHost = ServerConfiguration(
            endpoint: "http://example.com:8181",
            allowsInsecureLocalHTTP: true
        )
        XCTAssertThrowsError(try publicHost.validatedBaseURL()) { error in
            XCTAssertEqual(error as? ConfigurationError, .insecureHTTPRequiresLocalHost)
        }

        let deceptiveHost = ServerConfiguration(
            endpoint: "http://192.168.1.20.example.com:8181",
            allowsInsecureLocalHTTP: true
        )
        XCTAssertThrowsError(try deceptiveHost.validatedBaseURL())
    }

    func testHTTPOptInAcceptsIPv4Loopback() throws {
        let local = ServerConfiguration(
            endpoint: "http://127.0.0.1:8181",
            allowsInsecureLocalHTTP: true
        )
        XCTAssertEqual(try local.validatedBaseURL().host, "127.0.0.1")
    }

    func testHTTPOptInRejectsBarePublicHostnames() {
        let configuration = ServerConfiguration(
            endpoint: "http://voice:8181",
            allowsInsecureLocalHTTP: true
        )
        XCTAssertThrowsError(try configuration.validatedBaseURL()) { error in
            XCTAssertEqual(error as? ConfigurationError, .insecureHTTPRequiresLocalHost)
        }
    }

    func testCredentialsCannotHideInQueryOrFragment() {
        let query = ServerConfiguration(endpoint: "https://voice.example.test:8181?token=secret")
        XCTAssertThrowsError(try query.validatedBaseURL()) { error in
            XCTAssertEqual(error as? ConfigurationError, .invalidEndpoint)
        }

        let fragment = ServerConfiguration(endpoint: "https://voice.example.test:8181/#token")
        XCTAssertThrowsError(try fragment.validatedBaseURL()) { error in
            XCTAssertEqual(error as? ConfigurationError, .invalidEndpoint)
        }
    }

    func testOpenAIBaseAndExactRoutesAreNotDuplicated() throws {
        let base = ServerConfiguration(endpoint: "https://voice.example.test/v1")
        XCTAssertEqual(try base.transcriptionURL().path, "/v1/audio/transcriptions")

        let exact = ServerConfiguration(endpoint: "https://voice.example.test/v1/audio/transcriptions")
        XCTAssertEqual(try exact.transcriptionURL().path, "/v1/audio/transcriptions")
    }

}
