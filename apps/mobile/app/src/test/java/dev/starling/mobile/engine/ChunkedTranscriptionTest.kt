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

    // ---- empty-key guard (issue #118 defense, ported for B07) ---------------
    // Words that normalize to "" (pure punctuation) must never participate in
    // an overlap match: runs of empty keys would satisfy minMatch=2 with no
    // shared lexical word and splice unrelated halves together, dropping
    // words on both sides. Shared fixture corpus with the Python reference
    // (tests/test_stream_chunk.py) and the C++ port
    // (cpp/tests/stream_session_test.cpp); keep the three in lockstep.

    @Test
    fun punctuationOnlyBoundaryNeverSplicesUnrelatedTexts() {
        // The B07 reproduction: "—" and "..." normalize to "", two empty
        // matches satisfied minMatch=2 and produced "alpha — — agree",
        // silently dropping the negation "never" and the word "beta".
        val joined = ChunkedTranscription.joinTexts(
            listOf("alpha — — never", "beta ... ... agree"),
        )

        assertEquals("alpha — — never beta ... ... agree", joined)
    }

    @Test
    fun emptyKeysSharedFixtureNeverMatch() {
        // Byte-mirror of the Python/C++ shared fixture test_stitch_empty_keys_never_match.
        val out = ChunkedTranscription.stitchWords(
            listOf("--", ";;"),
            listOf("..", ",,", "word"),
        )

        assertEquals(listOf("--", ";;", "..", ",,", "word"), out)
    }

    @Test
    fun unicodePunctuationNeverMatches() {
        // Unicode punctuation (guillemets, ellipsis, em dash, inverted
        // question mark) normalizes to "" exactly like ASCII punctuation.
        val out = ChunkedTranscription.stitchWords(
            listOf("end", "„", "…", "—"),
            listOf("—", "…", "¿", "start"),
        )

        assertEquals(listOf("end", "„", "…", "—", "—", "…", "¿", "start"), out)
    }

    @Test
    fun longRunsOfPunctuationNeverMatch() {
        val tail = listOf("a") + List(5) { "." }
        val head = List(5) { "." } + listOf("b")

        assertEquals(tail + head, ChunkedTranscription.stitchWords(tail, head))
    }

    @Test
    fun repeatedRealWordsStillDedupeAsGenuineOverlap() {
        // Repeated real words are a genuine lexical overlap and still dedupe.
        val out = ChunkedTranscription.stitchWords(
            listOf("I", "said", "the", "the"),
            listOf("the", "the", "end"),
        )

        assertEquals(listOf("I", "said", "the", "the", "end"), out)
    }

    @Test
    fun genuineOverlapAdjacentToEmptyKeysStillDedupes() {
        // The guard must not overreach: empty keys sit right next to a real
        // two-word overlap, which still has to dedupe.
        val out = ChunkedTranscription.stitchWords(
            listOf("never", "—", "—", "cross", "roads"),
            listOf("cross", "roads", "ahead"),
        )

        assertEquals(listOf("never", "—", "—", "cross", "roads", "ahead"), out)
    }

    @Test
    fun nonAsciiSharedFixturesSurviveAndDedupe() {
        // Byte-mirrors of the Python/C++ shared fixtures (issue #118): the C++
        // port regressed on exactly these before it became UTF-8 aware.
        // Kotlin's \p{L} is Unicode-aware, so these pin cross-language parity
        // rather than reproduce a desktop bug.
        assertEquals(
            listOf("привет", "мир", "совсем", "другое"),
            ChunkedTranscription.stitchWords(listOf("привет", "мир"), listOf("совсем", "другое")),
        )
        assertEquals(
            listOf("你好", "世界", "再见", "朋友"),
            ChunkedTranscription.stitchWords(listOf("你好", "世界"), listOf("再见", "朋友")),
        )
        assertEquals(
            listOf("привет", "мир", "тут", "ок"),
            ChunkedTranscription.stitchWords(listOf("привет", "мир", "тут"), listOf("мир", "тут", "ок")),
        )
        // Committed spelling is kept when the overlap matches after normalization.
        assertEquals(
            listOf("привет,", "мир.", "!"),
            ChunkedTranscription.stitchWords(listOf("привет,", "мир."), listOf("привет", "мир", "!")),
        )
    }
}
