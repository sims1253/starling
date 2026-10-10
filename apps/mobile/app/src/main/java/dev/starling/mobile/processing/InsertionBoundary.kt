package dev.starling.mobile.processing

import java.util.Locale

/**
 * Insertion-boundary formatting (#341): when dictated text lands in the
 * middle of existing text, fix the leading space and the case of the first
 * letter, and nothing else.
 *
 * The Kotlin port of `packages/contracts/insertion-boundary` (its README is
 * authoritative); the JVM tests replay the contract's fixture table, like the
 * Python oracle (`tests/insertion_boundary.py`) and the Rust port do.
 * Android-free. Text is walked by code point; character classes follow the
 * Rust port's Unicode properties (White_Space, Alphabetic, Numeric,
 * Lowercase, Uppercase) and case mappings are the full ones.
 */
object InsertionBoundary {
    /** A rule that fired; [kind] is the contract's name for it. */
    enum class Change(val kind: String) {
        /** One U+0020 prepended. */
        LEADING_SPACE("leading_space"),

        /** The first cased character was lowercased. */
        FIRST_LETTER_CASE("first_letter_case"),
    }

    /**
     * The text around the insertion point, as the field reported it. No rule
     * reads [after]: the end of the dictated text is never touched.
     */
    data class Context(
        /** Text immediately before the insertion point ("" at a field start). */
        val before: String,
        /** Text immediately after the insertion point. */
        val after: String,
        /**
         * The field shows only its placeholder: [before] and [after] are hint
         * text and the field is treated as empty.
         */
        val showingHint: Boolean = false,
    )

    data class Adjustment(
        /** The adjusted text; exactly the input when no rule fired. */
        val text: String,
        /** The rules that fired, in rule order. */
        val changes: List<Change>,
    ) {
        val unchanged: Boolean get() = changes.isEmpty()
    }

    /**
     * After these, dictated content is starting (no space, case kept).
     * Straight `"` and `'` are ambiguous at an insertion point; the contract
     * pins them as opening.
     */
    private val OPENING = "([{\"'“‘„«「『【（".codePoints().toArray().toSet()

    /**
     * Punctuation that continues a sentence (besides alphanumerics), incl.
     * the Arabic comma and semicolon. Sentence enders are deliberately
     * absent: the case is kept after them.
     */
    private val CONTINUING = ",;:)]}”’»」』】）》،؛".codePoints().toArray().toSet()

    /**
     * Han, kana and CJK punctuation: no space between two of them. Hangul is
     * absent on purpose (Korean separates words with spaces).
     */
    private val NO_SPACE_SCRIPTS = listOf(
        0x3000..0x30FF, // CJK symbols and punctuation, Hiragana, Katakana
        0x31F0..0x31FF, // Katakana phonetic extensions
        0x3400..0x4DBF, // CJK Extension A
        0x4E00..0x9FFF, // CJK Unified Ideographs
        0xF900..0xFAFF, // CJK Compatibility Ideographs
        0xFF01..0xFF0F, // fullwidth punctuation
        0xFF1A..0xFF20, // fullwidth punctuation
        0xFF5B..0xFF9F, // fullwidth punctuation, halfwidth Katakana
        0x20000..0x3FFFF, // CJK Extensions B and later
    )

    fun adjust(raw: String, context: Context, verbatim: Boolean): Adjustment {
        if (verbatim || raw.isEmpty()) return Adjustment(raw, emptyList())
        val before = if (context.showingHint) "" else context.before
        var text = raw
        val changes = mutableListOf<Change>()

        if (before.isNotEmpty()) {
            val lastBefore = before.codePointBefore(before.length)
            val firstRaw = raw.codePointAt(0)
            if (!isWhitespace(firstRaw) && !isWhitespace(lastBefore) && lastBefore !in OPENING &&
                !(noSpaceBoundary(lastBefore) && noSpaceBoundary(firstRaw))
            ) {
                text = " $text"
                changes += Change.LEADING_SPACE
            }
        }

        if (continuesSentence(before) && !firstTokenIsProtected(raw)) {
            // Only the first cased character is considered: if it is already
            // lowercase, later capitals are left alone.
            var index = 0
            while (index < text.length) {
                val cp = text.codePointAt(index)
                val width = Character.charCount(cp)
                if (isCased(cp)) {
                    val lowered = lower(cp)
                    if (lowered != String(Character.toChars(cp))) {
                        // The full mapping: 'İ' lowercases to two code points.
                        text = text.substring(0, index) + lowered + text.substring(index + width)
                        changes += Change.FIRST_LETTER_CASE
                    }
                    break
                }
                index += width
            }
        }

        return Adjustment(text, changes)
    }

    /**
     * Whether [before] ends mid-sentence: its last non-whitespace character
     * is alphanumeric or continuing punctuation, and the trailing whitespace
     * holds no line break (a new line behaves like a field start).
     */
    private fun continuesSentence(before: String): Boolean {
        var end = before.length
        while (end > 0) {
            val cp = before.codePointBefore(end)
            if (!isWhitespace(cp)) break
            end -= Character.charCount(cp)
        }
        if (end == 0) return false
        val trailing = before.substring(end)
        val last = before.codePointBefore(end)
        return '\n' !in trailing && '\r' !in trailing && (isAlphanumeric(last) || last in CONTINUING)
    }

    /**
     * Code-looking first tokens keep their case: paths, URLs, emails,
     * hashtags, snake_case, camel humps, ALL-CAPS (two or more letters), and
     * the pronoun `I` (also `I,` and `I'm`).
     */
    private fun firstTokenIsProtected(raw: String): Boolean {
        val token = firstToken(raw) ?: return false
        if (token.any { it in "_/\\@#" } || token.startsWith("www.")) return true
        val cps = token.codePoints().toArray()
        if (cps[0] == 'I'.code && (cps.size == 1 || !isAlphanumeric(cps[1]))) return true
        val hasHump = (0 until cps.size - 1).any { Character.isLowerCase(cps[it]) && Character.isUpperCase(cps[it + 1]) }
        val letters = cps.filter(Character::isAlphabetic)
        val allCaps = letters.size >= 2 && letters.all(Character::isUpperCase)
        return hasHump || allCaps
    }

    /** The first whitespace-delimited token, or null when [raw] is all whitespace. */
    private fun firstToken(raw: String): String? {
        fun skip(from: Int, whitespace: Boolean): Int {
            var index = from
            while (index < raw.length) {
                val cp = raw.codePointAt(index)
                if (isWhitespace(cp) != whitespace) break
                index += Character.charCount(cp)
            }
            return index
        }
        val start = skip(0, whitespace = true)
        return if (start == raw.length) null else raw.substring(start, skip(start, whitespace = false))
    }

    private fun noSpaceBoundary(cp: Int): Boolean = NO_SPACE_SCRIPTS.any { cp in it }

    /**
     * Unicode White_Space. `Character.isWhitespace` leaves out the no-break
     * spaces (which `isSpaceChar` has) and NEL, and counts the information
     * separators U+001C–U+001F, which are control characters.
     */
    private fun isWhitespace(cp: Int): Boolean =
        (Character.isWhitespace(cp) && cp !in 0x1C..0x1F) || Character.isSpaceChar(cp) || cp == 0x85

    /** Alphabetic or Numeric, like Rust's `char::is_alphanumeric`. */
    private fun isAlphanumeric(cp: Int): Boolean = Character.isAlphabetic(cp) || when (Character.getType(cp)) {
        Character.DECIMAL_DIGIT_NUMBER.toInt(), Character.LETTER_NUMBER.toInt(), Character.OTHER_NUMBER.toInt() -> true
        else -> false
    }

    private fun lower(cp: Int): String = String(Character.toChars(cp)).lowercase(Locale.ROOT)

    /** Python's notion of cased: some full case mapping changes the character. */
    private fun isCased(cp: Int): Boolean {
        val self = String(Character.toChars(cp))
        return self.lowercase(Locale.ROOT) != self || self.uppercase(Locale.ROOT) != self
    }
}
