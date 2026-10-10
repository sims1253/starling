package dev.starling.mobile.audio

import dev.starling.mobile.storage.DiskLevel
import dev.starling.mobile.storage.DiskPolicy
import dev.starling.mobile.storage.FreeSpaceProbe
import java.io.File
import java.io.IOException

/**
 * The free-space watch of one running take (#342): asked once a second by
 * a thread of its own (a stalled checkpoint fsync must not hold it up), it probes [directory] every
 * [intervalMillis] and says whether the take has to stop. A failing probe
 * never stops a take.
 */
internal class DiskWatch(
    private val probe: FreeSpaceProbe,
    private val directory: File,
    private val policy: DiskPolicy,
    private val intervalMillis: Long = DiskPolicy.IN_TAKE_INTERVAL_MILLIS,
    // Monotonic: a clock set back must not postpone the next probe.
    private val clock: () -> Long = { System.nanoTime() / 1_000_000 },
) {
    private var nextCheck = clock() + intervalMillis

    fun critical(): Boolean {
        val now = clock()
        if (now < nextCheck) return false
        nextCheck = now + intervalMillis
        return policy.check(probe, directory)?.level == DiskLevel.CRITICAL
    }

    companion object {
        /** Whether a write failed because the disk is full (ENOSPC, EDQUOT). */
        fun isOutOfSpace(exception: Throwable): Boolean =
            generateSequence(exception) { it.cause }.any { cause ->
                cause is IOException && cause.message.orEmpty().let {
                    it.contains("ENOSPC") || it.contains("No space left") ||
                        it.contains("EDQUOT") || it.contains("quota exceeded", ignoreCase = true)
                }
            }
    }
}
