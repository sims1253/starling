package dev.starling.mobile.processing

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Replays every case of `fixtures/spoken-commands.json` against the Kotlin port. */
class SpokenCommandsConformanceTest {
    private val commands = SpokenCommands(Contracts.modeRouting("spoken-commands.json"))

    @Test
    fun replaysEveryFixtureCase() {
        val cases = JSONArray(Contracts.fixture("spoken-commands.json"))
        assertTrue("the fixture has cases", cases.length() > 0)
        for (index in 0 until cases.length()) {
            val case = cases.getJSONObject(index)
            val name = case.getString("name")
            val language = if (case.isNull("language")) null else case.getString("language")
            val actual = commands.apply(
                text = case.getString("input"),
                language = language,
                spokenCommands = case.getBoolean("spoken_commands"),
                snippets = snippets(case.getJSONArray("snippets")),
            )
            assertEquals("case $name", case.getString("output"), actual)
        }
    }

    private fun snippets(array: JSONArray): List<Snippet> =
        (0 until array.length()).map { index ->
            val snippet: JSONObject = array.getJSONObject(index)
            Snippet(snippet.getString("spoken"), snippet.getString("expansion"))
        }

    @Test
    fun astralCharacterBeforeACommandKeepsTheWordIntact() {
        assertEquals(
            "🎉, ok",
            commands.apply("🎉 comma ok", language = "en", spokenCommands = true, snippets = emptyList()),
        )
    }
}
