package dev.starling.mobile.processing

import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Replays the shared routing fixtures (the oracle's corpus) through the Kotlin resolver. */
class ModeRoutingConformanceTest {
    private class Case(
        val file: String,
        val name: String,
        val profilesFile: String,
        val request: JSONObject,
        val expected: JSONObject,
    ) {
        val label: String get() = "$file/$name (profiles $profilesFile)"
    }

    private val documents = mutableMapOf<String, ProfilesDocument>()

    private fun document(file: String): ProfilesDocument =
        documents.getOrPut(file) { ProfilesDocument.parse(Contracts.fixture(file)) }

    private fun cases(): List<Case> = listOf("routing.json", "routing-variants.json").flatMap { file ->
        val array = JSONArray(Contracts.fixture(file))
        List(array.length()) { i ->
            val c = array.getJSONObject(i)
            // The oracle's routing_cases(): a case names its profiles document, else profiles.json.
            Case(
                file = file,
                name = c.getString("name"),
                profilesFile = c.optString("profiles", "profiles.json"),
                request = c.getJSONObject("request"),
                expected = c.getJSONObject("expected"),
            )
        }
    }

    private fun JSONObject.optNullableString(key: String): String? = if (isNull(key)) null else getString(key)

    private fun requestOf(json: JSONObject) = RouteRequest(
        rawText = json.getString("raw_text"),
        manualMode = json.optNullableString("manual_mode"),
        manualLocked = json.optBoolean("manual_locked", true),
        sessionAllowsAliases = json.optBoolean("session_allows_aliases", true),
        secureField = json.optBoolean("secure_field", false),
        projectId = json.optNullableString("project_id"),
        site = json.optNullableString("site"),
        appId = json.optNullableString("app_id"),
        selectionAvailable = json.optBoolean("selection_available", false),
        selectionGranted = json.optBoolean("selection_granted", false),
    )

    /** The value of each result field an expected key can name. Unknown keys fail the case. */
    private fun actualValue(result: RouteResult, key: String): Any? = when (key) {
        "mode" -> result.mode
        "source" -> result.source
        "payload" -> result.payload
        "status" -> result.status
        "raw_text" -> result.rawText
        "prefix_span_codepoints" -> result.prefixSpanCodepoints?.let { listOf(it.first, it.second) }
        "delivery" -> result.delivery
        "selected_text_role" -> result.selectedTextRole
        "local_only" -> result.localOnly
        else -> throw IllegalStateException("expected key '$key' has no Kotlin result field")
    }

    /** JSON null and arrays compared as Kotlin values, matching [actualValue]. */
    private fun expectedValue(raw: Any?): Any? = when (raw) {
        JSONObject.NULL -> null
        is JSONArray -> (0 until raw.length()).map { raw.get(it) }
        else -> raw
    }

    @Test
    fun everyRoutingFixtureCaseReplaysLikeTheOracle() {
        val all = cases()
        // Guard against a fixture edit silently dropping cases from the replay.
        assertEquals("routing.json case count", 25, all.count { it.file == "routing.json" })
        assertEquals("routing-variants.json case count", 10, all.count { it.file == "routing-variants.json" })

        val failures = mutableListOf<String>()
        for (case in all) {
            if (case.expected.length() == 0) {
                failures += "${case.label}: no expected keys, so nothing would be checked"
                continue
            }
            try {
                val result = resolve(document(case.profilesFile), requestOf(case.request))
                for (key in case.expected.keys().asSequence()) {
                    val want = expectedValue(case.expected.get(key))
                    val got = actualValue(result, key)
                    if (want != got) failures += "${case.label}: $key expected $want, got $got"
                }
            } catch (e: Exception) {
                failures += "${case.label}: ${e.javaClass.simpleName}: ${e.message}"
            }
        }
        assertTrue(
            "${failures.size} of ${all.size} routing cases diverged from the oracle:\n" + failures.joinToString("\n"),
            failures.isEmpty(),
        )
    }

    @Test
    fun everyProfilesDocumentValidates() {
        // The oracle's test_profiles_documents_conform sees these five documents: canonical plus four variants.
        val files = File(Contracts.dir, "mode-routing/fixtures")
            .listFiles { f -> f.name.startsWith("profiles") && f.name.endsWith(".json") }
            .orEmpty()
            .map { it.name }
            .sorted()
        assertEquals("profiles*.json fixtures: $files", 5, files.size)
        val failures = files.mapNotNull { name ->
            runCatching { document(name).validate() }.exceptionOrNull()?.let { "$name: ${it.message}" }
        }
        assertTrue("profiles documents rejected by validate():\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun validateRejectsWhatTheOracleRejects() {
        // Mutations from tests/test_mode_routing.py, applied to the canonical document.
        fun mutated(edit: (JSONObject) -> Unit): ProfilesDocument {
            val root = JSONObject(Contracts.fixture("profiles.json"))
            edit(root)
            return ProfilesDocument.parse(root.toString())
        }

        val localRemote = mutated { it.getJSONArray("profiles").getJSONObject(0).put("asr_route", "remote-asr") }
        assertEquals("A local-only profile cannot use a remote route", messageOf { localRemote.validate() })

        // code-guidance (index 2) has selected_text "reference"; replace_selection needs edit_target.
        val replace = mutated { it.getJSONArray("profiles").getJSONObject(2).put("delivery", "replace_selection") }
        assertEquals("Replacement requires edit-target authority", messageOf { replace.validate() })
    }

    /** The IllegalArgumentException message from [block], or null when it does not throw. */
    private fun messageOf(block: () -> Unit): String? = try {
        block()
        null
    } catch (e: IllegalArgumentException) {
        e.message
    }

    @Test
    fun unknownManualModeIsRejected() {
        assertEquals(
            "Unknown manual mode",
            messageOf { resolve(document("profiles.json"), RouteRequest("hello", manualMode = "missing")) },
        )
    }

    @Test
    fun prefixSpanIsCodePointsWhenAliasHoldsAnEmoji() {
        // The alias "🎯 focus" is 8 UTF-16 units but 7 code points, so the span must be counted in code points.
        val base = document("profiles.json")
        val doc = ProfilesDocument(
            base.defaultProfile,
            base.profiles.map { if (it.id == "code-guidance") it.copy(aliases = listOf("🎯 focus")) else it },
            base.rules,
        )
        val raw = "🎯 focus: ship it"
        val result = resolve(doc, RouteRequest(raw))
        assertEquals("code-guidance", result.mode)
        assertEquals("phrase:🎯 focus", result.source)
        // 🎯 (1) + space (1) + focus (5) + ':' (1) + space (1) = 9 code points; 10 UTF-16 units.
        assertEquals(Pair(0, 9), result.prefixSpanCodepoints)
        assertEquals("ship it", result.payload)
        assertEquals(result.payload, raw.substring(raw.offsetByCodePoints(0, 9)))
    }

    @Test
    fun emojiAfterAliasLeavesTheSpanAndPayloadIntact() {
        val raw = "code this: 😀 add logs"
        val result = resolve(document("profiles.json"), RouteRequest(raw))
        assertEquals("code-guidance", result.mode)
        // "code this: " is 11 characters with no surrogate pairs inside the span.
        assertEquals(Pair(0, 11), result.prefixSpanCodepoints)
        assertEquals("😀 add logs", result.payload)
        assertEquals(result.payload, raw.substring(raw.offsetByCodePoints(0, 11)))
    }

    @Test
    fun literalEscapeWithEmojiPayloadCountsCodePoints() {
        val raw = "literal 🎉 x"
        val result = resolve(document("profiles.json"), RouteRequest(raw))
        assertEquals("verbatim", result.mode)
        assertEquals(Pair(0, 8), result.prefixSpanCodepoints)
        assertEquals("🎉 x", result.payload)
    }
}
