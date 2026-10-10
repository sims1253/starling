package dev.starling.mobile.storage

import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.audio.CaptureStopPolicy
import dev.starling.mobile.audio.DiskWatch
import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.RecordingStatus
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile

/** #342: free-space thresholds, the in-take watch and a clean, recoverable low-disk stop. */
class DiskSpaceTest {
    @get:Rule
    val folder = TemporaryFolder()

    private val mib = 1024L * 1024

    @Test
    fun theDesktopThresholdsDecideTheLevel() {
        val policy = DiskPolicy.DEFAULT
        assertEquals(DiskLevel.CRITICAL, policy.assess(128 * mib - 1).level)
        assertEquals(DiskLevel.LOW, policy.assess(128 * mib).level)
        assertEquals(DiskLevel.LOW, policy.assess(1024 * mib - 1).level)
        assertEquals(DiskLevel.OK, policy.assess(1024 * mib).level)
        // 16 kHz PCM16 is 1.92 MB a minute.
        assertEquals(10L, policy.minutesLeft(128 * mib + 10 * 1_920_000L))
        assertEquals(0L, policy.minutesLeft(0))
    }

    @Test
    fun aFailingProbeNeverReadsAsAFullDisk() {
        val failing = FreeSpaceProbe { throw IOException("statvfs failed") }
        assertEquals(null, DiskPolicy.DEFAULT.check(failing, folder.root))
        assertTrue(FreeSpaceProbe.SYSTEM.availableBytes(File(folder.root, "not/yet/created")) > 0)
    }

    @Test
    fun theInTakeWatchProbesEveryIntervalAndStopsBelowTheFloor() {
        var clock = 0L
        var free = 2048 * mib
        var probes = 0
        val watch = DiskWatch(
            probe = { probes++; free },
            directory = folder.root,
            policy = DiskPolicy.DEFAULT,
            intervalMillis = 5_000,
            clock = { clock },
        )
        // Asked every second by the checkpoint thread; probed every five.
        for (second in 1..4) {
            clock = second * 1_000L
            assertFalse(watch.critical())
        }
        assertEquals(0, probes)
        clock = 5_000
        assertFalse(watch.critical())
        assertEquals(1, probes)
        // The disk fills: the next probe stops the take, not the one in between.
        free = 100 * mib
        clock = 9_000
        assertFalse(watch.critical())
        clock = 10_000
        assertTrue(watch.critical())
        assertEquals(2, probes)
        // A probe that fails keeps the take running.
        val failing = DiskWatch({ throw IOException("gone") }, folder.root, DiskPolicy.DEFAULT, 0, { clock })
        assertFalse(failing.critical())
    }

    @Test
    fun outOfSpaceWritesAreRecognized() {
        assertTrue(DiskWatch.isOutOfSpace(IOException("write failed: ENOSPC (No space left on device)")))
        assertTrue(DiskWatch.isOutOfSpace(IllegalStateException("wrapped", IOException("No space left on device"))))
        assertTrue(DiskWatch.isOutOfSpace(IOException("write failed: EDQUOT (Quota exceeded)")))
        assertFalse(DiskWatch.isOutOfSpace(IOException("EIO")))
        assertFalse(DiskWatch.isOutOfSpace(IllegalStateException("The microphone returned an audio read error")))
    }

    @Test
    fun aLowDiskStopSettlesAsACompletedTake() {
        val result = CaptureStopPolicy.settle(error = null, bytes = 64_000, cappedAtLimit = false, stoppedForLowDisk = true)
        assertEquals(CaptureResult.Completed(2.0, cappedAtLimit = false, stoppedForLowDisk = true), result)
        // A worker error still wins.
        assertEquals(CaptureResult.Failed("mic"), CaptureStopPolicy.settle("mic", 64_000, false, true))
    }

    /**
     * A write the full disk refused part-way leaves a torn chunk past the
     * counted payload; finish() trims it, so the take commits like any
     * other and is retryable with every whole chunk.
     */
    @Test
    fun aTakeStoppedByAFullDiskIsFinalizedAndRecoverable() {
        val store = RecordingStore(File(folder.root, "recordings"))
        val recording = store.create()
        val partial = store.partialFile(recording)
        val writer = WavWriter(partial)
        val chunk = ByteArray(32_000) { (it % 113).toByte() }
        writer.write(chunk, chunk.size)
        // Half a chunk reached the file before the write failed with ENOSPC.
        RandomAccessFile(partial, "rw").use { file ->
            file.seek(file.length())
            file.write(ByteArray(16_001) { 7 })
        }
        writer.finish()
        assertEquals(44L + chunk.size, partial.length())

        val committed = store.commitAudio(recording, chunk.size / 32_000.0)

        assertEquals(RecordingStatus.PENDING, committed.status)
        val audio = store.withRequestAudio(recording.id) { it.readBytes() }
        assertArrayEquals(WavWriter.header(chunk.size.toLong()) + chunk, audio)
    }
}
