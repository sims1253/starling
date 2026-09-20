package dev.starling.mobile.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ChunkedTranscriptionTest {
    private val sampleRate = 16_000

    @Test
    fun audioWithinOneWindowYieldsASingleShotWindow() {
        val short = ChunkedTranscription.planWindows(30 * sampleRate, sampleRate)
        assertEquals(listOf(ChunkedTranscription.Window(0, 30 * sampleRate)), short)

        // A window is chunk + overlap: still one call, its natural length.
        val exactlyWindow = ChunkedTranscription.planWindows(32 * sampleRate, sampleRate)
        assertEquals(listOf(ChunkedTranscription.Window(0, 32 * sampleRate)), exactlyWindow)
    }

    @Test
    fun longAudioUsesThirtySecondStepsWithTwoSecondOverlap() {
        val total = 1_000_000
        val windows = ChunkedTranscription.planWindows(total, sampleRate)

        assertEquals(
            listOf(
                ChunkedTranscription.Window(0, 32 * sampleRate),
                ChunkedTranscription.Window(30 * sampleRate, 32 * sampleRate),
                ChunkedTranscription.Window(60 * sampleRate, total - 60 * sampleRate),
            ),
            windows,
        )
        // Every engine call stays bounded; adjacent windows overlap by 2 s.
        assertTrue(windows.all { it.length <= 32 * sampleRate })
        for (index in 1 until windows.size) {
            assertEquals(2 * sampleRate, windows[index - 1].endExclusive - windows[index].start)
        }
        // The windows start at zero and end at the audio's end, with no gaps.
        assertEquals(0, windows.first().start)
        assertEquals(total, windows.last().endExclusive)
    }

    @Test
    fun boundaryWordsTranscribedTwiceAppearOnce() {
        val first = "the quick brown fox jumps over"
        val second = "jumps over the lazy dog today"
        val third = "dog today and keeps going"

        val joined = ChunkedTranscription.joinTexts(listOf(first, second, third))

        assertEquals("the quick brown fox jumps over the lazy dog today and keeps going", joined)
    }

    @Test
    fun punctuationAndCaseDoNotBlockOverlapMatching() {
        val joined = ChunkedTranscription.joinTexts(
            listOf("He said, \"Hello there!\"", "hello there, and waved back"),
        )

        assertEquals("He said, \"Hello there!\" and waved back", joined)
    }

    @Test
    fun disjointTextsAreConcatenatedNotDropped() {
        val joined = ChunkedTranscription.joinTexts(listOf("first chunk ends abruptly", "second chunk begins anew"))

        assertEquals("first chunk ends abruptly second chunk begins anew", joined)
    }

    @Test
    fun whitespaceIsCollapsedWhenJoining() {
        val joined = ChunkedTranscription.joinTexts(
            listOf("  spaced   out   ", "\n\nnoise\tthen words  "),
        )

        assertEquals("spaced out noise then words", joined)
    }

    @Test
    fun stitchingEmptySidesKeepsTheOther() {
        assertTrue(ChunkedTranscription.stitchWords(emptyList(), listOf("a", "b")) == listOf("a", "b"))
        assertTrue(ChunkedTranscription.stitchWords(listOf("a", "b"), emptyList()) == listOf("a", "b"))
        assertEquals("", ChunkedTranscription.joinTexts(listOf("   ", "")))
    }

    @Test
    fun aSingleRunOfOneWordIsNotTreatedAsOverlap() {
        // minMatch = 2: "once" matches once, which must not be deduplicated.
        val joined = ChunkedTranscription.joinTexts(
            listOf("I saw it once", "once more please"),
        )

        assertEquals("I saw it once once more please", joined)
    }
}
