package dev.starling.mobile.storage

import dev.starling.mobile.audio.WavWriter
import java.io.File
import java.io.IOException

/**
 * Free space for recording (#342). Every entry point checks it before a
 * take starts ([DiskPolicy.check]) and AudioCapture re-checks it while one
 * runs, so a filling disk is warned about early and a take stops cleanly,
 * finalized and committed, while there is still room for that.
 */
fun interface FreeSpaceProbe {
    /** Bytes this app can still write on the file system holding [directory]. */
    @Throws(IOException::class)
    fun availableBytes(directory: File): Long

    companion object {
        /**
         * `statvfs`'s f_bavail (File.usableSpace) of the nearest existing
         * directory. File.usableSpace answers a failed statvfs with 0, so 0
         * reads as unmeasurable, not as a full disk: the take starts with the
         * "can't check" warning, and a disk that is really full stops it at
         * its first refused write (ENOSPC).
         */
        val SYSTEM = FreeSpaceProbe { directory -> measure(directory, File::getUsableSpace) }

        internal fun measure(directory: File, usableSpace: (File) -> Long): Long {
            val existing = generateSequence(directory.absoluteFile) { it.parentFile }.firstOrNull(File::isDirectory)
                ?: throw IOException("No directory to measure above ${directory.path}")
            return usableSpace(existing).takeIf { it > 0 }
                ?: throw IOException("Free space unavailable for ${existing.path}")
        }

        /**
         * Debug builds only (StarlingApplication sets it from
         * `files/debug/free-space-mb`): a stand-in for the real free space,
         * so a device test can drive the low-space paths. Null measures.
         */
        @Volatile
        var debugOverride: FreeSpaceProbe? = null

        /** [SYSTEM], unless a debug build stands a value in for it. */
        val current: FreeSpaceProbe
            get() = FreeSpaceProbe { directory -> (debugOverride ?: SYSTEM).availableBytes(directory) }
    }
}

enum class DiskLevel { OK, LOW, CRITICAL }

data class DiskReading(val availableBytes: Long, val level: DiskLevel)

/**
 * The desktop's thresholds: below [warnBelow] the user is warned, below
 * [stopBelow] no take starts and a running one is stopped cleanly. The
 * floor leaves room for the stop itself (the final header, the metadata
 * commit) and for the rest of the phone.
 */
data class DiskPolicy(
    val warnBelow: Long = 1024L * 1024 * 1024,
    val stopBelow: Long = 128L * 1024 * 1024,
) {
    fun assess(availableBytes: Long): DiskReading = DiskReading(
        availableBytes,
        when {
            availableBytes < stopBelow -> DiskLevel.CRITICAL
            availableBytes < warnBelow -> DiskLevel.LOW
            else -> DiskLevel.OK
        },
    )

    /**
     * Probes [directory]; null when the probe fails, which never stops a
     * take: an unanswerable question is not a full disk.
     */
    fun check(probe: FreeSpaceProbe, directory: File): DiskReading? =
        runCatching { assess(probe.availableBytes(directory)) }.getOrNull()

    /** Minutes of 16 kHz PCM16 that fit before the stop threshold: what a warning says. */
    fun minutesLeft(availableBytes: Long): Long =
        (availableBytes - stopBelow).coerceAtLeast(0) / BYTES_PER_MINUTE

    companion object {
        val DEFAULT = DiskPolicy()
        private const val BYTES_PER_MINUTE =
            WavWriter.SAMPLE_RATE.toLong() * WavWriter.CHANNELS * WavWriter.BYTES_PER_SAMPLE * 60

        /**
         * How often a running take re-checks the free space: a minute of
         * audio is ~1.9 MB, so this leaves the stop threshold untouched.
         */
        const val IN_TAKE_INTERVAL_MILLIS = 5_000L
    }
}
