import Foundation

public enum FidelityWarningCode: String, Codable, Sendable {
    case nonOneListStart
    case possibleSelfCorrection
    case negationPresent
    case expectedTermMissing
    case shortAnswer
    case discourseWordPreserved
    case possibleAudioGap
}

public struct FidelityWarning: Identifiable, Equatable, Sendable {
    public let id: UUID
    public let code: FidelityWarningCode
    public let message: String

    public init(code: FidelityWarningCode, message: String) {
        id = UUID()
        self.code = code
        self.message = message
    }

    public static func == (lhs: FidelityWarning, rhs: FidelityWarning) -> Bool {
        lhs.code == rhs.code && lhs.message == rhs.message
    }
}

public struct FidelityAnalysis: Equatable, Sendable {
    /// The exact input. Warnings never rewrite it.
    public let rawText: String
    public let warnings: [FidelityWarning]
}

public enum FidelityAnalyzer {
    public static func analyze(
        _ rawText: String,
        expectedTerms: [String] = [],
        recordingDurationSeconds: Double? = nil,
        coveredDurationSeconds: Double? = nil
    ) -> FidelityAnalysis {
        var warnings: [FidelityWarning] = []
        let nsRange = NSRange(rawText.startIndex..., in: rawText)

        if let expression = try? NSRegularExpression(pattern: #"(?m)^[ \t]*(\d+)[.)](?=\s)"#),
           let match = expression.firstMatch(in: rawText, range: nsRange),
           let numberRange = Range(match.range(at: 1), in: rawText),
           let startNumber = Int(rawText[numberRange]),
           startNumber != 1 {
            warnings.append(.init(
                code: .nonOneListStart,
                message: "List starts at \(startNumber). Keep that number."
            ))
        }

        if rawText.range(of: #"\b(er+|uh+|um+)\b"#, options: [.regularExpression, .caseInsensitive]) != nil {
            warnings.append(.init(
                code: .possibleSelfCorrection,
                message: "Possible spoken correction. Check the nearby words before editing."
            ))
        }

        if rawText.range(
            of: #"\b(not|never|no|cannot|can't|won't|haven't|hasn't|don't|doesn't|didn't)\b"#,
            options: [.regularExpression, .caseInsensitive]
        ) != nil {
            warnings.append(.init(
                code: .negationPresent,
                message: "Keep negations when editing; removing them can change the meaning."
            ))
        }

        if rawText.range(of: #"\blike\b"#, options: [.regularExpression, .caseInsensitive]) != nil {
            warnings.append(.init(
                code: .discourseWordPreserved,
                message: "The transcript keeps the word \"like\"."
            ))
        }

        if rawText.range(
            of: #"^\s*([A-Za-z]|agreed|yes|no|okay|ok)\s*[.!]?\s*$"#,
            options: [.regularExpression, .caseInsensitive]
        ) != nil {
            warnings.append(.init(
                code: .shortAnswer,
                message: "Keep short answers, including a single letter."
            ))
        }

        let folded = rawText.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
        for term in expectedTerms where !term.isEmpty {
            let expected = term.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
            if !folded.contains(expected) {
                warnings.append(.init(
                    code: .expectedTermMissing,
                    message: "Expected term \"\(term)\" is missing. Check the saved audio."
                ))
            }
        }

        if let recordingDurationSeconds, let coveredDurationSeconds,
           recordingDurationSeconds > 1,
           coveredDurationSeconds >= 0,
           coveredDurationSeconds < recordingDurationSeconds * 0.8 {
            warnings.append(.init(
                code: .possibleAudioGap,
                message: "Timed segments cover less than 80% of the recording. Check the saved audio."
            ))
        }

        return FidelityAnalysis(rawText: rawText, warnings: warnings)
    }
}
