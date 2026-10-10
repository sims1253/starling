package dev.starling.mobile.storage

import dev.starling.mobile.audio.WavWriter
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicBoolean

/**
 * History-audio upkeep (#342): settles takes left with both a WAV and a
 * FLAC ([RecordingStore.settleAtRest]), compresses finished takes to FLAC
 * and then applies the retention policy, one pass at a time on [executor]. The app
 * schedules a pass after start, every 15 minutes, after each transcription
 * attempt and after the storage settings change. A pass does not start
 * while a take records ([recording]: any capture of this process writing),
 * and one that runs stops before its next compression or removal when a
 * take starts. A take whose compression fails three times is not tried
 * again until the next launch.
 */
class AudioUpkeep(
    private val store: RecordingStore,
    private val policy: PolicyGate,
    private val executor: Executor,
    private val recording: () -> Boolean = WavWriter::anyOpen,
    private val onFailure: (Throwable) -> Unit = {},
) {
    /** What one pass did. */
    data class Report(
        val compressed: Int = 0,
        val savedBytes: Long = 0,
        /** Takes that could not be compressed; they stay WAV. */
        val failures: Int = 0,
        val retention: RetentionReport = RetentionReport(),
        /** A take started recording and the pass stopped early. */
        val paused: Boolean = false,
    ) {
        /** Nothing worth telling the user. */
        val isEmpty: Boolean
            get() = compressed == 0 && failures == 0 && retention.removed.isEmpty() &&
                retention.held.isEmpty() && retention.overLimit.isEmpty() && retention.failed.isEmpty()
    }

    /** The last pass that did or found something; null until one did. */
    @Volatile
    var lastReport: Report? = null
        private set

    /** Called on the upkeep thread after a pass with something to report. */
    @Volatile
    var onReport: ((Report) -> Unit)? = null

    private val queued = AtomicBoolean(false)

    // Passes run one at a time, whichever thread calls: a scheduled pass and
    // a direct [runPass] never overlap.
    private val passLock = Any()

    // Only touched under [passLock].
    private val compressionFailures = HashMap<String, Int>()

    /** Queues a pass unless one is already waiting. */
    fun schedule() {
        if (!queued.compareAndSet(false, true)) return
        executor.execute {
            queued.set(false)
            runCatching { runPass() }.onFailure(onFailure)
        }
    }

    /** One pass, on the calling thread; waits for a pass another thread runs. */
    fun runPass(): Report = synchronized(passLock) {
        if (recording()) return@synchronized Report(paused = true)
        var compressed = 0
        var saved = 0L
        var failures = 0
        var paused = false
        // Takes a crash (or a refused unlink) left with both files, settled
        // here rather than at open: the check decodes the whole take.
        store.settleAtRest(stop = recording)
        for (id in store.compressionCandidates()) {
            if ((compressionFailures[id] ?: 0) >= MAX_COMPRESSION_ATTEMPTS) continue
            if (recording()) {
                paused = true
                break
            }
            try {
                val outcome = store.compressAudio(id)
                if (outcome is RecordingStore.Compression.Compressed) {
                    compressed++
                    // A WAV storage kept saves nothing yet.
                    if (!outcome.wavKept) saved += outcome.wavBytes - outcome.flacBytes
                }
            } catch (exception: Exception) {
                compressionFailures[id] = (compressionFailures[id] ?: 0) + 1
                failures++
                onFailure(exception)
            }
        }
        val retention = if (paused) RetentionReport() else store.applyRetention(policy, stop = recording)
        // A changed policy is applied by a pass of its own.
        if (retention.policyChanged) schedule()
        val report = Report(compressed, saved, failures, retention, paused || retention.stopped)
        if (!report.isEmpty) {
            lastReport = report
            onReport?.invoke(report)
        }
        report
    }

    companion object {
        const val MAX_COMPRESSION_ATTEMPTS = 3
        const val INTERVAL_MILLIS = 15L * 60 * 1000
    }
}
