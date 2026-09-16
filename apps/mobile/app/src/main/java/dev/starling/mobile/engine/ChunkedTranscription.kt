package dev.starling.mobile.engine

/**
 * Bounds each on-device engine call the way the serving layer bounds its
 * parakeet calls: audio longer than 30 s goes through overlapping windows
 * (a 30 s step with 2 s of right context, the geometry of
 * src/starling/parakeet/chunking.py) because one full-attention encoder pass
 * over an unbounded clip scales quadratically in length. The C API returns
 * text per call, so the duplicated overlap is deduplicated at the word level
 * like the serving layer's stitch_words (src/starling/stream_chunk.py).
 */
object ChunkedTranscription {
    /** One bounded engine call: [start] in samples and its [length] in samples. */
    data class Window(val start: Int, val length: Int) {
        val endExclusive: Int get() = start + length
    }

    const val CHUNK_SECONDS = 30.0
    const val OVERLAP_SECONDS = 2.0
    const val DEFAULT_SAMPLE_RATE = 16_000

    /**
     * Windows of [CHUNK_SECONDS] + [OVERLAP_SECONDS] of audio, stepped by
     * [CHUNK_SECONDS], covering [totalSamples] contiguously. Audio that fits
     * one window yields that single window (byte-exact single-shot path,
     * like the server); the final window may be shorter and nothing is padded.
     */
    fun planWindows(totalSamples: Int, sampleRate: Int = DEFAULT_SAMPLE_RATE): List<Window> {
        require(sampleRate > 0) { "sampleRate must be positive" }
        if (totalSamples <= 0) return emptyList()
        val window = Math.round((CHUNK_SECONDS + OVERLAP_SECONDS) * sampleRate).toInt()
        val step = Math.round(CHUNK_SECONDS * sampleRate).toInt()
        val windows = ArrayList<Window>()
        var start = 0
        while (start < totalSamples) {
            val end = minOf(start + window, totalSamples)
            windows.add(Window(start, end - start))
            if (end >= totalSamples) break
            start += step
        }
        return windows
    }

    /**
     * Joins per-window transcripts into one text: the last words of each
     * text are matched against the first words of the next (they transcribed
     * the same 2 s of audio) and the duplicate copy is dropped.
     */
    fun joinTexts(texts: List<String>): String {
        var committed: List<String> = emptyList()
        for (text in texts) {
            val words = text.trim().split(WHITESPACE).filter { it.isNotEmpty() }
            committed = stitchWords(committed, words)
        }
        return committed.joinToString(" ")
    }

    /**
     * Appends [newWords] to [committedWords], deduplicating the words their
     * overlapping regions share. Port of the serving layer's stitch_words:
     * the longest common word run between the tail and the head (compared
     * normalized; punctuation and case do not block a match) is kept once,
     * taken from the earlier window. Without a run of at least [minMatch]
     * words the two are simply concatenated — a duplicated word reads better
     * than a dropped one for dictation.
     */
    fun stitchWords(
        committedWords: List<String>,
        newWords: List<String>,
        maxOverlap: Int = 24,
        minMatch: Int = 2,
    ): List<String> {
        if (committedWords.isEmpty()) return newWords
        if (newWords.isEmpty()) return committedWords
        val tail = committedWords.takeLast(maxOverlap)
        val head = newWords.take(maxOverlap)
        val run = longestCommonRun(tail.map(::normalizeWord), head.map(::normalizeWord))
            ?: return committedWords + newWords
        if (run.length < minMatch) return committedWords + newWords
        // Keep committed up to the end of the shared run; take new after it.
        val keep = committedWords.size - tail.size + run.startInTail + run.length
        val startInNew = run.startInHead + run.length
        return committedWords.take(keep) + newWords.drop(startInNew)
    }

    private data class Run(val startInTail: Int, val startInHead: Int, val length: Int)

    /**
     * Longest contiguous run of equal words between [tail] and [head]; the
     * first such run wins (earliest in tail, then earliest in head), matching
     * difflib's find_longest_match tie-breaking used by the Python original.
     */
    private fun longestCommonRun(tail: List<String>, head: List<String>): Run? {
        var best: Run? = null
        var previous = IntArray(head.size)
        for (i in tail.indices) {
            val current = IntArray(head.size)
            for (j in head.indices) {
                if (tail[i] == head[j]) {
                    current[j] = (if (j > 0) previous[j - 1] else 0) + 1
                    val length = current[j]
                    if (best == null || length > best.length) {
                        best = Run(i - length + 1, j - length + 1, length)
                    }
                }
            }
            previous = current
        }
        return best
    }

    /** Lowercase, punctuation stripped — for overlap matching only. */
    private fun normalizeWord(word: String): String =
        WORD_NOISE.replace(word.lowercase(), "")

    private val WHITESPACE = Regex("\\s+")

    /** Python `\w` (Unicode letters, digits, underscore) plus apostrophes. */
    private val WORD_NOISE = Regex("[^\\p{L}\\p{N}_']+")
}
