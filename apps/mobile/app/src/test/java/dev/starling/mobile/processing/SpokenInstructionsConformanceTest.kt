package dev.starling.mobile.processing

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Replays every case of `fixtures/spoken-instructions.json` against the Kotlin port. */
class SpokenInstructionsConformanceTest {
    private val commands = SpokenCommands(Contracts.modeRouting("spoken-commands.json"))
    private val instructions = SpokenInstructions(
        Contracts.modeRouting("spoken-instructions.json"),
        commands,
    )

    @Test
    fun replaysEveryFixtureCase() {
        val cases = JSONArray(Contracts.fixture("spoken-instructions.json"))
        assertTrue("the fixture has cases", cases.length() > 0)
        for (index in 0 until cases.length()) {
            val case = cases.getJSONObject(index)
            val name = case.getString("name")
            val language = if (case.isNull("language")) null else case.getString("language")
            val expected = case.getJSONObject("expected")
            val actual = instructions.split(case.getString("input"), language)

            assertEquals("case $name: matched", expected.getBoolean("matched"), actual.matched)
            assertEquals("case $name: payload", expected.getString("payload"), actual.payload)
            assertEquals("case $name: instruction", expected.getString("instruction"), actual.instruction)
            assertEquals("case $name: delimiter_span", span(expected, "delimiter_span"), actual.delimiterSpan)
            assertEquals("case $name: instruction_span", span(expected, "instruction_span"), actual.instructionSpan)
        }
    }

    @Test
    fun delimiterAndInstructionSpansCountCodePointsAfterAnEmoji() {
        // The emoji is one code point but two UTF-16 units: UTF-16 offsets would be 8 and 17.
        val split = instructions.split("🎉 done Starling, make it formal", language = "en")
        assertTrue(split.matched)
        assertEquals("🎉 done ", split.payload)
        assertEquals("make it formal", split.instruction)
        assertEquals(7 to 16, split.delimiterSpan)
        assertEquals(16 to 31, split.instructionSpan)
    }

    @Test
    fun emojiInPayloadAndInstructionKeepExactSubstrings() {
        // The ZWJ sequence is three code points; the spans must count it that way.
        val text = "Send 👩‍💻 Starling, translate 😀 to German"
        val split = instructions.split(text, language = "en")
        assertTrue(split.matched)
        assertEquals("Send 👩‍💻 ", split.payload)
        assertEquals("translate 😀 to German", split.instruction)
        assertEquals(9 to 18, split.delimiterSpan)
        assertEquals(18 to 40, split.instructionSpan)
        // The spans address the same text the payload and instruction were cut from.
        val raw = text.codePoints().toArray()
        assertEquals(split.payload, String(raw, 0, split.delimiterSpan!!.first))
        assertEquals(
            "Starling, translate 😀 to German",
            String(raw, split.delimiterSpan!!.first, raw.size - split.delimiterSpan!!.first),
        )
    }

    @Test
    fun emojiOnlyTailIsNotAnInstruction() {
        // Python's isalnum is false for an emoji, so a punctuation-and-symbol tail never fires.
        val split = instructions.split("Send the report Starling, 🎉", language = "en")
        assertFalse(split.matched)
        assertEquals(null, split.delimiterSpan)
    }

    private fun span(expected: JSONObject, key: String): Pair<Int, Int>? {
        if (expected.isNull(key)) return null
        val pair = expected.getJSONArray(key)
        return pair.getInt(0) to pair.getInt(1)
    }
}
