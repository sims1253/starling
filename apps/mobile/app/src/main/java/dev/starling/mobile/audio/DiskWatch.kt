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

/**
 * The low-disk stop of whichever take is running (#342), tied to that take.
 * [begin] opens a take and returns its token; only a raise carrying the
 * current, still open token sets the flag. A watcher whose probe returns
 * after its take ended (and maybe after the next one began) is refused, so
 * a late reading can never stop a different recording.
 */
internal class LowDiskStop {
    private var current = 0L
    private var open = false

    @Volatile
    var raised = false
        private set

    @Synchronized
    fun begin(): Long {
        current++
        open = true
        raised = false
        return current
    }

    /** Ends the take [token]; its raises are refused from now on. */
    @Synchronized
    fun close(token: Long) {
        if (token == current) open = false
    }

    @Synchronized
    fun isOpen(token: Long): Boolean = open && token == current

    /** Raises the stop for [token]; false when that take is no longer the running one. */
    @Synchronized
    fun raise(token: Long): Boolean {
        if (!isOpen(token)) return false
        raised = true
        return true
    }
}

/**
 * Starts the watch thread of take [token]: every [tickMillis] it asks
 * [watch] and raises [stop] for that take on a critical reading. It ends
 * when the take closes, and at once when interrupted (the capture
 * interrupts it as the take ends); a probe still in flight then is refused
 * by [LowDiskStop.raise].
 */
internal fun startDiskWatch(
    watch: DiskWatch,
    stop: LowDiskStop,
    token: Long,
    tickMillis: Long,
): Thread = Thread({
    try {
        while (stop.isOpen(token)) {
            Thread.sleep(tickMillis)
            if (stop.isOpen(token) && watch.critical()) stop.raise(token)
        }
    } catch (_: InterruptedException) {
        // Cancelled with its take; nothing to clean up.
    }
}, "starling-disk-watch").apply { isDaemon = true; start() }
