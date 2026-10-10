package dev.starling.mobile.storage

import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.audio.CaptureStopPolicy
import dev.starling.mobile.audio.DiskWatch
import dev.starling.mobile.audio.LowDiskStop
import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.audio.startDiskWatch
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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

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

    @Test
    fun aDelayedProbeFromAnEndedTakeCannotStopTheNextOne() {
        val stop = LowDiskStop()
        val probing = CountDownLatch(1)
        val answer = CountDownLatch(1)
        // The first take's probe stalls until the next take is running, then reads a full disk.
        // Like statvfs it cannot be interrupted.
        val first = stop.begin()
        val stalled = DiskWatch(
            probe = {
                probing.countDown()
                var interrupted = false
                while (true) {
                    try {
                        answer.await()
                        break
                    } catch (_: InterruptedException) {
                        interrupted = true
                    }
                }
                if (interrupted) Thread.currentThread().interrupt()
                1 * mib
            },
            directory = folder.root,
            policy = DiskPolicy.DEFAULT,
            intervalMillis = 0,
        )
        val firstWatch = startDiskWatch(stalled, stop, first, tickMillis = 1)
        assertTrue(probing.await(5, TimeUnit.SECONDS))
        // The take ends (cancelling its watch) and the next one starts while the probe is out.
        stop.close(first)
        firstWatch.interrupt()
        val second = stop.begin()
        answer.countDown()
        firstWatch.join(5_000)
        assertFalse(firstWatch.isAlive)
        assertFalse(stop.raised)
        assertTrue(stop.isOpen(second))
        assertFalse(stop.raise(first))
        assertFalse(stop.raised)

        // The running take's own watch still stops it.
        val full = DiskWatch({ 1 * mib }, folder.root, DiskPolicy.DEFAULT, intervalMillis = 0)
        val secondWatch = startDiskWatch(full, stop, second, tickMillis = 1)
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
        while (!stop.raised && System.nanoTime() < deadline) Thread.sleep(1)
        assertTrue(stop.raised)
        // Ending the take ends its watch; the reason stays readable for the settlement.
        stop.close(second)
        secondWatch.join(5_000)
        assertFalse(secondWatch.isAlive)
        assertTrue(stop.raised)
        // The next take starts clear.
        stop.begin()
        assertFalse(stop.raised)
    }

    @Test
    fun aStopAfterTheWorkerDecidedToEndForLowDiskKeepsTheReason() {
        val stop = LowDiskStop()
        // The watch raised and the worker latched its exit before the user's stop.
        val watched = stop.begin()
        assertFalse(stop.endIfRaised(watched))
        assertTrue(stop.raise(watched))
        assertTrue(stop.endIfRaised(watched))
        stop.cancel(watched)
        assertTrue(stop.raised)
        // A write failed for want of space, then the stop came.
        val full = stop.begin()
        assertTrue(stop.endIfRaised(full, outOfSpace = true))
        stop.cancel(full)
        assertTrue(stop.raised)
        // The stop came first: the worker no longer ends for low disk.
        val stopped = stop.begin()
        assertTrue(stop.raise(stopped))
        stop.cancel(stopped)
        assertFalse(stop.endIfRaised(stopped))
        assertFalse(stop.endIfRaised(stopped, outOfSpace = true))
        assertFalse(stop.raised)
    }

    @Test
    fun anExplicitStopIsNeverALowDiskStop() {
        val stop = LowDiskStop()
        // The watch raised, and the user stopped before the take acted on it.
        val take = stop.begin()
        assertTrue(stop.raise(take))
        stop.cancel(take)
        assertFalse(stop.raised)
        assertFalse(stop.raise(take))
        // A take that already ended for low disk keeps the reason when its owner stops it to settle.
        val ended = stop.begin()
        assertTrue(stop.raise(ended))
        stop.close(ended)
        stop.cancel(ended)
        assertTrue(stop.raised)
        // A stop of an earlier take changes nothing for the running one.
        val running = stop.begin()
        assertTrue(stop.raise(running))
        stop.cancel(ended)
        assertTrue(stop.raised)
    }

    @Test
    fun aStatThatAnswersZeroIsUnmeasuredNotAFullDisk() {
        // File.usableSpace answers a failed statvfs with 0.
        val failed = FreeSpaceProbe { directory -> FreeSpaceProbe.measure(directory) { 0L } }
        assertEquals(null, DiskPolicy.DEFAULT.check(failed, folder.root))
        val measured = FreeSpaceProbe { directory -> FreeSpaceProbe.measure(directory) { 100 * mib } }
        assertEquals(DiskLevel.CRITICAL, DiskPolicy.DEFAULT.check(measured, folder.root)?.level)
        // In a take, an unmeasured disk keeps recording.
        val watch = DiskWatch(failed, folder.root, DiskPolicy.DEFAULT, intervalMillis = 0, clock = { 0L })
        assertFalse(watch.critical())
    }
}
