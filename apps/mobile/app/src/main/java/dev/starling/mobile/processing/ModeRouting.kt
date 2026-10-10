package dev.starling.mobile.processing

import java.util.regex.Pattern
import org.json.JSONArray
import org.json.JSONObject

/**
 * Kotlin port of the frozen mode-routing resolver. `tests/mode_routing.py`
 * (validate_config, prefix_match, resolve) is the contract; this file mirrors
 * it step for step so the shared fixtures replay identically on every port.
 *
 * Two JVM defaults are replaced on purpose, because the fixtures pin the
 * Python behavior:
 * - Whitespace follows Python's `str.isspace()` set, which is wider than
 *   `\s` in Java and includes NBSP and the 0x1C-0x1F separators. Matching is
 *   Unicode case-insensitive, so a dotless i or long s matches "i" and "s".
 * - Spans and alias lengths are Unicode code points. Java regex offsets and
 *   String.length count UTF-16 units, so they are converted at the boundary.
 *   The payload is still sliced in UTF-16 on the original string.
 */

data class ModeSnippet(val spoken: String, val expansion: String)

/** The mode fields the app routes and processes on. Unknown JSON fields are ignored. */
data class Mode(
    val id: String,
    val version: Int,
    val name: String,
    val description: String,
    val aliases: List<String>,
    val allowSpokenOverrides: Boolean,
    val snippets: List<ModeSnippet>,
    val asrRoute: String,
    val authoringRoute: String?,
    val localOnly: Boolean,
    /** "off", "reference" or "edit_target". */
    val selectedText: String,
    val selectionRequired: Boolean,
    val transformKinds: List<String>,
    val language: String?,
    val spokenCommands: Boolean,
    /** "direct" or "staged". */
    val processingDelivery: String,
    /** insert, replace_selection, preview, save_note, copy or insert_enter. */
    val delivery: String,
    val automaticDelivery: Boolean,
    /** faithful, verbatim, ...: verbatim returns raw text untouched. */
    val behavior: String,
)

/** A scoped routing rule. Exactly one of the scope fields is expected to be set. */
data class Rule(
    val id: String,
    val profileId: String,
    val priority: Int,
    val appId: String? = null,
    val site: String? = null,
    val projectId: String? = null,
)

/** A profiles resolution document: the default mode, the modes, and the scoped rules. */
class ProfilesDocument(
    val defaultProfile: String,
    val profiles: List<Mode>,
    val rules: List<Rule>,
) {
    /**
     * validate_config. Throws IllegalArgumentException with the oracle's
     * messages. Checks run in the oracle's order so the first violation matches.
     */
    fun validate() {
        val ids = profiles.map { it.id }
        require(ids.distinct().size == ids.size && defaultProfile in ids) {
            "Unique profiles and a valid default are required"
        }
        require("verbatim" in ids) { "The literal escape requires a verbatim profile" }
        require(rules.all { it.profileId in ids }) { "Rule references an unknown profile" }
        require(rules.map { it.id }.distinct().size == rules.size) { "Duplicate rule ID" }
        for (p in profiles) {
            require(!(p.localOnly && listOf(p.asrRoute, p.authoringRoute.orEmpty()).any { it.startsWith("remote-") })) {
                "A local-only profile cannot use a remote route"
            }
            require(!(p.selectionRequired && p.selectedText == "off")) { "Required selection cannot be disabled" }
            require(!(p.delivery == "replace_selection" && p.selectedText != "edit_target")) {
                "Replacement requires edit-target authority"
            }
        }
    }

    companion object {
        /** Parses a profiles document. It does not validate; call [validate] for that. */
        fun parse(json: String): ProfilesDocument {
            val root = JSONObject(json)
            return ProfilesDocument(
                defaultProfile = root.getString("default_profile"),
                profiles = root.getJSONArray("profiles").mapObjects(::parseMode),
                rules = root.optJSONArray("rules")?.mapObjects(::parseRule).orEmpty(),
            )
        }

        private fun parseMode(o: JSONObject) = Mode(
            id = o.getString("id"),
            version = o.getInt("version"),
            name = o.getString("name"),
            description = o.getString("description"),
            aliases = o.getJSONArray("aliases").strings(),
            allowSpokenOverrides = o.getBoolean("allow_spoken_overrides"),
            snippets = o.getJSONArray("snippets").mapObjects {
                ModeSnippet(it.getString("spoken"), it.getString("expansion"))
            },
            asrRoute = o.getString("asr_route"),
            authoringRoute = o.optNullableString("authoring_route"),
            localOnly = o.getBoolean("local_only"),
            selectedText = o.getString("selected_text"),
            selectionRequired = o.getBoolean("selection_required"),
            transformKinds = o.getJSONArray("transform_kinds").strings(),
            language = o.optNullableString("language"),
            spokenCommands = o.getBoolean("spoken_commands"),
            processingDelivery = o.getString("processing_delivery"),
            delivery = o.getString("delivery"),
            automaticDelivery = o.getBoolean("automatic_delivery"),
            behavior = o.getString("behavior"),
        )

        private fun parseRule(o: JSONObject) = Rule(
            id = o.getString("id"),
            profileId = o.getString("profile_id"),
            priority = o.getInt("priority"),
            appId = o.optNullableString("app_id"),
            site = o.optNullableString("site"),
            projectId = o.optNullableString("project_id"),
        )

        private fun JSONObject.optNullableString(key: String): String? = if (isNull(key)) null else getString(key)

        private fun JSONArray.strings(): List<String> = List(length()) { getString(it) }

        private fun <T> JSONArray.mapObjects(transform: (JSONObject) -> T): List<T> =
            List(length()) { transform(getJSONObject(it)) }
    }
}

/** One take's routing inputs. Defaults match the oracle's request.get(..., default). */
data class RouteRequest(
    val rawText: String,
    val manualMode: String? = null,
    val manualLocked: Boolean = true,
    val sessionAllowsAliases: Boolean = true,
    val secureField: Boolean = false,
    val projectId: String? = null,
    val site: String? = null,
    val appId: String? = null,
    val selectionAvailable: Boolean = false,
    val selectionGranted: Boolean = false,
)

/**
 * A route decision. The payload is a view: [prefixSpanCodepoints] records the
 * removed leading span as code points, and [rawText] is always kept. Delivery
 * fields are null on the early returns (blocked, conflicts), as in the oracle.
 */
data class RouteResult(
    val mode: String?,
    val source: String,
    val payload: String,
    val status: String,
    val rawText: String,
    val prefixSpanCodepoints: Pair<Int, Int>? = null,
    val delivery: String? = null,
    val selectedTextRole: String? = null,
    val localOnly: Boolean? = null,
)

/**
 * The oracle's `prefix_match`: a leading, whole-token match of [phrase] in
 * [text], case-insensitive, with an optional ": " or whitespace separator.
 * Returns the match end as a UTF-16 index for slicing, or null when there is none.
 */
internal fun prefixMatchEnd(text: String, phrase: String): Int? {
    val body = pySplit(phrase).joinToString("$PY_SPACE+") { Pattern.quote(it) }
    val pattern = "^$PY_SPACE*$body(?=\\z|$PY_SPACE|:)(?:$PY_SPACE*:$PY_SPACE*|$PY_SPACE+)?"
    val matcher = Pattern.compile(pattern, Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE).matcher(text)
    return if (matcher.lookingAt()) matcher.end() else null
}

/**
 * The oracle's `resolve`. Precedence, in order: secure-field policy, a locked
 * manual mode, the literal escape, a leading alias, scoped rules, then the
 * default profile. Ambiguous ties are needs_resolution, never a guess.
 */
fun resolve(doc: ProfilesDocument, request: RouteRequest): RouteResult {
    doc.validate()
    val raw = request.rawText
    if (request.secureField) {
        return RouteResult(null, "policy:secure-field", raw, "blocked", raw)
    }
    val profiles = doc.profiles.associateBy { it.id }
    val manual = request.manualMode
    require(manual == null || manual in profiles) { "Unknown manual mode" }

    var chosen = manual ?: doc.defaultProfile
    var source = if (manual != null) "manual" else "default"
    if (manual == null) {
        val matched = doc.rules.mapNotNull { rule -> ruleRank(rule, request)?.let { it to rule } }
        if (matched.isNotEmpty()) {
            val best = matched.maxOf { it.first }
            val winners = matched.filter { it.first == best }.map { it.second }
            if (winners.map { it.profileId }.distinct().size > 1) {
                return RouteResult(null, "rule_conflict", raw, "needs_resolution", raw)
            }
            val winner = winners.minWith(Comparator { a, b -> compareCodePoints(a.id, b.id) })
            chosen = winner.profileId
            source = "rule:${winner.id}"
        }
    }

    var payload = raw
    var span: Pair<Int, Int>? = null
    val locked = manual != null && request.manualLocked
    val aliasesAllowed = !locked && profiles.getValue(chosen).allowSpokenOverrides && request.sessionAllowsAliases
    if (aliasesAllowed) {
        val escapeEnd = prefixMatchEnd(raw, "literal")
        if (escapeEnd != null) {
            chosen = "verbatim"
            source = "escape:literal"
            payload = raw.substring(escapeEnd)
            span = Pair(0, raw.codePointCount(0, escapeEnd))
        } else {
            val matches = ArrayList<PhraseMatch>()
            for (profile in profiles.values) {
                for (alias in profile.aliases) {
                    val end = prefixMatchEnd(raw, alias) ?: continue
                    val stripped = pyStrip(alias)
                    matches += PhraseMatch(
                        alias = alias,
                        profileId = profile.id,
                        tokens = pySplit(alias).size,
                        characters = stripped.codePointCount(0, stripped.length),
                        end = end,
                    )
                }
            }
            if (matches.isNotEmpty()) {
                val best = matches.maxOf { Rank(it.tokens, it.characters) }
                val winners = matches.filter { Rank(it.tokens, it.characters) == best }
                if (winners.map { it.profileId }.distinct().size > 1) {
                    return RouteResult(null, "phrase_conflict", raw, "needs_resolution", raw)
                }
                // Python sorts by (alias, profile id); the winner is the first of that order.
                val winner = winners.minWith(Comparator { a, b ->
                    compareCodePoints(a.alias, b.alias).takeIf { it != 0 } ?: compareCodePoints(a.profileId, b.profileId)
                })
                chosen = winner.profileId
                source = "phrase:${winner.alias}"
                payload = raw.substring(winner.end)
                span = Pair(0, raw.codePointCount(0, winner.end))
            }
        }
    }

    val profile = profiles.getValue(chosen)
    val needsSelection = profile.selectionRequired &&
        !(request.selectionAvailable && request.selectionGranted)
    val status = if (pyStrip(payload).isEmpty() || needsSelection) "needs_input" else "ready"
    return RouteResult(
        mode = chosen,
        source = source,
        payload = payload,
        status = status,
        rawText = raw,
        prefixSpanCodepoints = span,
        delivery = profile.delivery,
        selectedTextRole = profile.selectedText,
        localOnly = profile.localOnly,
    )
}

/** A candidate leading alias, with its match end kept as a UTF-16 index for slicing. */
private class PhraseMatch(
    val alias: String,
    val profileId: String,
    val tokens: Int,
    val characters: Int,
    val end: Int,
)

/** Lexicographic rank, like Python tuple comparison: (specificity, priority) or (tokens, characters). */
private data class Rank(val major: Int, val minor: Int) : Comparable<Rank> {
    override fun compareTo(other: Rank): Int =
        if (major != other.major) major.compareTo(other.major) else minor.compareTo(other.minor)
}

/** The rule's rank when every scope it names matches the request. Project beats site beats app. */
private fun ruleRank(rule: Rule, request: RouteRequest): Rank? {
    val criteria = buildMap {
        rule.projectId?.let { put("project_id", it) }
        rule.site?.let { put("site", it) }
        rule.appId?.let { put("app_id", it) }
    }
    val requested = mapOf("project_id" to request.projectId, "site" to request.site, "app_id" to request.appId)
    if (criteria.isEmpty() || !criteria.all { (k, v) -> requested[k] == v }) return null
    val specificity = if ("project_id" in criteria) 3 else if ("site" in criteria) 2 else 1
    return Rank(specificity, rule.priority)
}

/** Python `str.isspace()`: Zs, plus the bidi B and S classes (which include 0x1C-0x1F and NEL). */
internal fun isPySpace(c: Char): Boolean =
    c in '\t'..'\r' || c in '\u001C'..' ' || c == '\u0085' || c == ' ' || c == ' ' ||
        c in ' '..' ' || c == ' ' || c == ' ' || c == ' ' || c == ' ' ||
        c == '　'

/** The same set as [isPySpace], as a regex class for the oracle's `\s`. */
private const val PY_SPACE =
    "[\\t\\n\\u000B\\f\\r\\u001C-\\u001F \\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000]"

private val SPACE_RUN = Regex("$PY_SPACE+")

/** Python `str.strip()`. */
internal fun pyStrip(s: String): String = s.trim { isPySpace(it) }

/** Python `str.split()` with no argument: runs of whitespace, no empty tokens. */
internal fun pySplit(s: String): List<String> {
    val trimmed = pyStrip(s)
    return if (trimmed.isEmpty()) emptyList() else trimmed.split(SPACE_RUN)
}

/** Python's str ordering: by code point, not UTF-16 unit, so tie-breaks match across ports. */
private fun compareCodePoints(a: String, b: String): Int {
    val x = a.codePoints().toArray()
    val y = b.codePoints().toArray()
    for (i in 0 until minOf(x.size, y.size)) {
        if (x[i] != y[i]) return x[i].compareTo(y[i])
    }
    return x.size.compareTo(y.size)
}
