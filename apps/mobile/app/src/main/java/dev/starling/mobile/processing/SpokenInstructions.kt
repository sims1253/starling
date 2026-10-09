package dev.starling.mobile.processing

import org.json.JSONObject

/**
 * The split of one finalized take. Spans are Unicode code points, not UTF-16
 * units, because the contract (and every other platform) counts code points;
 * the payload and instruction strings are the exact substrings either way.
 * When nothing fired the payload is the whole text and both spans are null.
 */
data class Split(
    val matched: Boolean,
    val payload: String,
    val instruction: String,
    /** Code-point `[start, end)` of the delimiter token, or null. */
    val delimiterSpan: Pair<Int, Int>?,
    /** Code-point `[start, end)` of the instruction region (delimiter onward), or null. */
    val instructionSpan: Pair<Int, Int>?,
)

/**
 * Port of the trailing spoken instruction grammar (`tests/spoken_instructions.py`).
 * A delimiter spoken near the end of a finalized take splits it: the text after
 * it travels as the transform's instruction, never as input text.
 */
class SpokenInstructions(tableJson: String, private val commands: SpokenCommands) {
    /** The spelling the UI shows, e.g. "Starling,". */
    val canonical: String

    private val matchTokens: Set<String>
    private val windowWords: Int

    init {
        val delimiter = JSONObject(tableJson).getJSONObject("delimiter")
        canonical = delimiter.getString("canonical")
        val array = delimiter.getJSONArray("match_tokens")
        matchTokens = (0 until array.length()).map { array.getString(it) }.toSet()
        windowWords = delimiter.getInt("window_words")
    }

    /**
     * Splits at the last delimiter within the last [windowWords] tokens that is
     * not escaped by the language's literal word and has a letter or digit after it.
     */
    fun split(text: String, language: String?): Split {
        val spans = tokens(text)
        val cores = spans.map { core(text.substring(it.first, it.second)) }
        val literal = commands.literalWord(language)

        var found = -1
        for (i in maxOf(0, spans.size - windowWords) until spans.size) {
            if (cores[i] !in matchTokens) continue
            if (i > 0 && cores[i - 1] == literal) continue
            if (!hasPythonAlnumFrom(text, spans[i].second)) continue
            found = i
        }
        if (found < 0) {
            return Split(matched = false, payload = text, instruction = "", delimiterSpan = null, instructionSpan = null)
        }

        val (start, end) = spans[found]
        val delimiterStart = text.codePointCount(0, start)
        val delimiterEnd = text.codePointCount(0, end)
        return Split(
            matched = true,
            payload = text.substring(0, start),
            instruction = trimWhitespace(text.substring(end)),
            delimiterSpan = delimiterStart to delimiterEnd,
            instructionSpan = delimiterEnd to text.codePointCount(0, text.length),
        )
    }

    /**
     * Where live text must stop while a take is still being spoken: the
     * UTF-16 start of the first delimiter candidate in the trailing window
     * (whitespace before it included), or null. Unlike [split] it does not
     * wait for words after the delimiter, since they have not been spoken
     * yet; the final's [split] decides.
     */
    fun liveCut(text: String, language: String?): Int? {
        val spans = tokens(text)
        val literal = commands.literalWord(language)
        for (i in maxOf(0, spans.size - windowWords) until spans.size) {
            if (core(text.substring(spans[i].first, spans[i].second)) !in matchTokens) continue
            if (i > 0 && core(text.substring(spans[i - 1].first, spans[i - 1].second)) == literal) continue
            return text.substring(0, spans[i].first).trimEnd(' ', '\t', '\n', '\r', '\u000B', '\u000C').length
        }
        return null
    }

    /** Removes a leading delimiter token from an instruction region's text; anything else is returned unchanged. */
    fun stripDelimiter(instruction: String): String {
        val first = tokens(instruction).firstOrNull() ?: return instruction
        if (core(instruction.substring(first.first, first.second)) !in matchTokens) return instruction
        return trimWhitespace(instruction.substring(first.second))
    }

    private companion object {
        /**
         * Python's `str.isalnum()` per code point: letters, plus any character
         * with a numeric value (digits, letter-numbers and other-numbers, and
         * numeric ideographs). Java's `Character.getNumericValue` returns -1
         * only for characters without a numeric value.
         */
        fun hasPythonAlnumFrom(text: String, from: Int): Boolean {
            var i = from
            while (i < text.length) {
                val codePoint = text.codePointAt(i)
                if (Character.isLetter(codePoint) || Character.getNumericValue(codePoint) != -1) return true
                i += Character.charCount(codePoint)
            }
            return false
        }
    }
}
