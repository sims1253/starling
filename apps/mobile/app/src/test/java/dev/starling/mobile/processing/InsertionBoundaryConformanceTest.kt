package dev.starling.mobile.processing

import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Replays every case of `insertion-boundary/fixtures/boundary-cases.json` against the Kotlin port. */
class InsertionBoundaryConformanceTest {
    @Test
    fun replaysEveryFixtureCase() {
        val cases = JSONArray(Contracts.insertionBoundaryFixture("boundary-cases.json"))
        assertTrue("the fixture has cases", cases.length() > 0)
        for (index in 0 until cases.length()) {
            val case = cases.getJSONObject(index)
            val name = case.getString("case_id")
            val actual = InsertionBoundary.adjust(
                raw = case.getString("raw"),
                context = InsertionBoundary.Context(
                    before = case.getString("before"),
                    after = case.getString("after"),
                    showingHint = case.optBoolean("showing_hint", false),
                ),
                verbatim = case.getBoolean("verbatim"),
            )
            val expectedChanges = case.getJSONArray("expected_changes")
            assertEquals("case $name", case.getString("expected_text"), actual.text)
            assertEquals(
                "case $name changes",
                (0 until expectedChanges.length()).map { expectedChanges.getJSONObject(it).getString("kind") },
                actual.changes.map { it.kind },
            )
            assertEquals("case $name: unchanged means byte-for-byte raw", actual.unchanged, actual.text == case.getString("raw"))
        }
    }

    @Test
    fun aNoBreakSpaceCountsAsWhitespace() {
        val adjusted = InsertionBoundary.adjust("Next", InsertionBoundary.Context("word ", ""), verbatim = false)
        assertEquals("next", adjusted.text)
    }

    @Test
    fun anAstralFirstLetterIsLoweredWhole() {
        // U+10400 DESERET CAPITAL LONG I lowercases to U+10428.
        val adjusted = InsertionBoundary.adjust("𐐀x", InsertionBoundary.Context("word", ""), verbatim = false)
        assertEquals(" 𐐨x", adjusted.text)
    }
}
