package dev.starling.mobile.processing

import org.json.JSONObject

/**
 * Port of the deterministic processing step's spoken commands and mode
 * snippets (`tests/spoken_commands.py`, the semantics). The vocabulary is
 * contract data (`packages/contracts/mode-routing/spoken-commands.json`), so
 * every platform recognizes the same phrases. Offsets here are UTF-16 index
 * pairs; they never split a surrogate pair because the token boundaries are
 * ASCII whitespace.
 */

/** A mode snippet: when [spoken] is said, [expansion] takes its place. */
data class Snippet(val spoken: String, val expansion: String)

/** The whitespace set: exactly these six characters, so every platform splits tokens alike. */
internal const val WHITESPACE = " \t\n\r\u000B\u000C"

/** Punctuation a recognizer attaches to a word; it is ignored when matching a command. */
internal const val ATTACHED = ",.;:!?"

/** Trims [WHITESPACE] from both ends; Python's `strip(WHITESPACE)`. */
internal fun trimWhitespace(text: String): String = text.trim { it in WHITESPACE }

/** Maximal runs of non-whitespace as `[start, end)` UTF-16 index pairs. */
internal fun tokens(text: String): List<Pair<Int, Int>> {
    val spans = ArrayList<Pair<Int, Int>>()
    var i = 0
    while (i < text.length) {
        if (text[i] in WHITESPACE) {
            i++
            continue
        }
        val start = i
        while (i < text.length && text[i] !in WHITESPACE) i++
        spans.add(start to i)
    }
    return spans
}

/**
 * A token's core: the token without attached [ATTACHED] punctuation at either
 * end, lowercased. A quoted mention keeps its quote, so it never matches.
 */
internal fun core(token: String): String = token.trim { it in ATTACHED }.lowercase()

class SpokenCommands(tableJson: String) {
    private sealed class Action {
        class Punctuation(val value: String) : Action()
        object LineBreak : Action()
        object Paragraph : Action()
        object Bullet : Action()
        class Expand(val value: String) : Action()

        /** Kept rather than rejected at load: the Python oracle fails only when such a phrase matches. */
        class Unknown(val name: String) : Action()
    }

    /** [rank] orders ties: a built-in command beats a snippet with the same words. */
    private class Phrase(val words: List<String>, val rank: Int, val action: Action)

    private class Language(val literal: String, val commands: List<Phrase>)

    private val languages: Map<String, Language>

    init {
        val root = JSONObject(tableJson).getJSONObject("languages")
        val parsed = HashMap<String, Language>()
        for (key in root.keys()) {
            val lang = root.getJSONObject(key)
            val commands = lang.getJSONArray("commands")
            val phrases = (0 until commands.length()).map { index ->
                val command = commands.getJSONObject(index)
                val phrase = command.getString("phrase").lowercase()
                Phrase(
                    words = words(phrase),
                    rank = 0,
                    action = when (val name = command.getString("action")) {
                        "punctuation" -> Action.Punctuation(command.getString("value"))
                        "line_break" -> Action.LineBreak
                        "paragraph" -> Action.Paragraph
                        "bullet" -> Action.Bullet
                        else -> Action.Unknown(name)
                    },
                )
            }
            parsed[key] = Language(lang.getString("literal"), phrases)
        }
        languages = parsed
    }

    /**
     * The language's table. BCP 47 tags are case-insensitive and may carry a
     * region; an absent or empty tag means English, as in the oracle's
     * `language or "en"`.
     */
    private fun tableFor(language: String?): Language? {
        val tag = if (language.isNullOrEmpty()) "en" else language
        return languages[tag.substringBefore('-').lowercase()]
    }

    /** The escape word that makes the next phrase plain text; "literal" when the language has no table. */
    fun literalWord(language: String?): String = tableFor(language)?.literal ?: "literal"

    fun apply(
        text: String,
        language: String?,
        spokenCommands: Boolean,
        snippets: List<Snippet>,
    ): String {
        val lang = tableFor(language)
        val literal = lang?.literal ?: "literal"
        val phrases = ArrayList<Phrase>()
        if (spokenCommands && lang != null) phrases += lang.commands
        for (snippet in snippets) {
            phrases += Phrase(
                words = words(snippet.spoken.lowercase()),
                rank = 1,
                action = Action.Expand(snippet.expansion),
            )
        }
        // Longest words first, then longest characters (code points, as Python's len counts), commands before snippets.
        val ordered = phrases
            .filter { it.words.isNotEmpty() }
            .sortedWith(
                compareBy<Phrase>(
                    { -it.words.size },
                    { -it.words.joinToString(" ").let { joined -> joined.codePointCount(0, joined.length) } },
                    { it.rank },
                ),
            )

        val spans = tokens(text)
        val cores = spans.map { core(text.substring(it.first, it.second)) }

        fun matchAt(i: Int): Pair<Int, Action>? {
            for (phrase in ordered) {
                val n = phrase.words.size
                if (i + n <= spans.size && cores.subList(i, i + n) == phrase.words) {
                    return n to phrase.action
                }
            }
            return null
        }

        var out = ""
        var pos = 0
        var skipWs = false
        var i = 0
        while (i < spans.size) {
            val (start, end) = spans[i]
            val gap = if (skipWs) "" else text.substring(pos, start)
            if (cores[i] == literal) {
                val escaped = matchAt(i + 1)
                if (escaped != null) {
                    val n = escaped.first
                    out += gap + text.substring(spans[i + 1].first, spans[i + n].second)
                    pos = spans[i + n].second
                    i += n + 1
                    skipWs = false
                    continue
                }
            }
            val found = matchAt(i)
            if (found == null) {
                out += gap + text.substring(start, end)
                pos = end
                i += 1
                skipWs = false
                continue
            }
            val (n, action) = found
            when (action) {
                is Action.Punctuation -> {
                    // Replaces the punctuation the recognizer already put on the word, and
                    // keeps any line break the dropped whitespace held (a layout command before it).
                    val stripped = out.trimEnd { it in WHITESPACE }
                    val layout = "\n".repeat(out.substring(stripped.length).count { it == '\n' })
                    out = stripped
                    if (out.isNotEmpty() && out.last() in ATTACHED) out = out.dropLast(1)
                    out += action.value + layout
                    skipWs = layout.isNotEmpty()
                }
                Action.LineBreak -> {
                    out = out.trimEnd(' ', '\t') + "\n"
                    skipWs = true
                }
                Action.Paragraph -> {
                    out = out.trimEnd(' ', '\t') + "\n\n"
                    skipWs = true
                }
                Action.Bullet -> {
                    out = out.trimEnd(' ', '\t')
                    if (out.isNotEmpty() && !out.endsWith("\n")) out += "\n"
                    out += "- "
                    skipWs = true
                }
                is Action.Expand -> {
                    out += gap + action.value
                    skipWs = false
                }
                is Action.Unknown -> throw IllegalArgumentException("unknown action ${action.name}")
            }
            pos = spans[i + n - 1].second
            i += n
        }
        if (!skipWs) out += text.substring(pos)
        return out
    }

    private companion object {
        /** Tokenizes a phrase or snippet spelling into its words. */
        fun words(text: String): List<String> = tokens(text).map { text.substring(it.first, it.second) }
    }
}
