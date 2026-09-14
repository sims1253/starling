import XCTest
@testable import StarlingVoiceCore

final class FidelityTests: XCTestCase {
    func testNonOneListStartIsPreserved() {
        let raw = "5. capture audio\n6. transcribe"
        let result = FidelityAnalyzer.analyze(raw)
        XCTAssertEqual(result.rawText, raw)
        XCTAssertTrue(result.warnings.contains { $0.code == .nonOneListStart })
    }

    func testUncommonExpectedVocabularyIsNeverGuessed() {
        let raw = "Add off to the proxy."
        let result = FidelityAnalyzer.analyze(raw, expectedTerms: ["auth"])
        XCTAssertEqual(result.rawText, raw)
        XCTAssertTrue(result.warnings.contains { $0.code == .expectedTermMissing })
    }

    func testSpokenCorrectionIsReviewOnly() {
        let raw = "I want orange, err, yellow."
        let result = FidelityAnalyzer.analyze(raw)
        XCTAssertEqual(result.rawText, raw)
        XCTAssertTrue(result.warnings.contains { $0.code == .possibleSelfCorrection })
    }

    func testNegationIsPreserved() {
        let raw = "Never merge this; I haven't approved it."
        let result = FidelityAnalyzer.analyze(raw)
        XCTAssertEqual(result.rawText, raw)
        XCTAssertTrue(result.warnings.contains { $0.code == .negationPresent })
    }

    func testDiscourseLikeIsPreserved() {
        let raw = "This was, like, easier."
        let result = FidelityAnalyzer.analyze(raw)
        XCTAssertEqual(result.rawText, raw)
        XCTAssertTrue(result.warnings.contains { $0.code == .discourseWordPreserved })
    }

    func testShortAnswersAreContent() {
        for raw in ["A", "agreed."] {
            let result = FidelityAnalyzer.analyze(raw)
            XCTAssertEqual(result.rawText, raw)
            XCTAssertTrue(result.warnings.contains { $0.code == .shortAnswer })
        }
    }

    func testMeasuredAudioGapDoesNotFabricateWords() {
        let raw = "First answer."
        let result = FidelityAnalyzer.analyze(
            raw,
            recordingDurationSeconds: 600,
            coveredDurationSeconds: 200
        )
        XCTAssertEqual(result.rawText, raw)
        XCTAssertTrue(result.warnings.contains { $0.code == .possibleAudioGap })
    }
}
