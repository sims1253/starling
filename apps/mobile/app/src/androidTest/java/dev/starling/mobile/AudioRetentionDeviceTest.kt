package dev.starling.mobile

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaPlayer
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.starling.mobile.audio.AudioCapture
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.data.RetentionClass
import dev.starling.mobile.engine.WavPcm
import dev.starling.mobile.network.TranscriptionEngine
import dev.starling.mobile.storage.ClassLimits
import dev.starling.mobile.storage.DiskLevel
import dev.starling.mobile.storage.FreeSpaceProbe
import dev.starling.mobile.storage.HoldReason
import dev.starling.mobile.storage.PolicyGate
import dev.starling.mobile.storage.RecordingStore
import dev.starling.mobile.storage.RetentionPolicy
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Named-device acceptance for #342, in the debug app's own process with the
 * debug test microphone (files/debug/test-mic.wav, looped at real-time pace):
 *
 *   compress   records a take, transcribes it on-device from the WAV, waits
 *              for the app's upkeep to make it FLAC, then checks the request
 *              audio is byte-identical, the platform's own FLAC decoder
 *              agrees sample for sample, playback works, and retries it
 *   retention  a store of its own (injected clock, under files/debug) with
 *              a 10-minute take: compression time, then an age and a size
 *              policy with held takes, and a reopen
 *   lowdisk    the free-space probe stands in low values: the start check
 *              refuses / warns, and a running take stops itself cleanly and
 *              commits as a recoverable take
 *
 * Opt-in: `am instrument -e retention <phase|all> [-e seconds N] ...`.
 * compress leaves files/debug/compress-id and files/debug/request-from-wav.wav
 * for a host-side check with the reference `flac` decoder.
 */
@RunWith(AndroidJUnit4::class)
class AudioRetentionDeviceTest {
    private val app = ApplicationProvider.getApplicationContext<StarlingApplication>()
    private val arguments = InstrumentationRegistry.getArguments()
    private val debugDir = File(app.filesDir, "debug")
    private val source by lazy {
        assertTrue("files/debug/test-mic.wav is required", File(debugDir, "test-mic.wav").isFile)
        WavPcm.decodePcm16(File(debugDir, "test-mic.wav"))!!.pcm
    }

    private fun phase(name: String) =
        assumeTrue("set -e retention $name to run", arguments.getString("retention") in setOf(name, "all"))

    @Test
    fun compress() {
        phase("compress")
        File(debugDir, "fail-transcription").delete()
        val seconds = arguments.getString("seconds")?.toDouble() ?: 30.0
        val recording = app.recordings.create()
        File(debugDir, "compress-id").writeText(recording.id)
        val capture = AudioCapture()
        assertEquals(null, capture.start(app, app.recordings.partialFile(recording)))
        Thread.sleep((seconds * 1000).toLong())
        val completed = stop(capture) as CaptureResult.Completed
        app.recordings.commitAudio(recording, completed.durationSeconds)

        val fromWav = app.recordings.withRequestAudio(recording.id) { it.readBytes() }
        assertSourceAudio(fromWav, "request audio from the WAV")
        File(debugDir, "request-from-wav.wav").writeBytes(fromWav)
        val first = transcribe(recording.id)
        assertEquals(RecordingStatus.TRANSCRIBED, first.status)

        // The attempt settled; the app's upkeep compresses the take on its own.
        val wav = File(app.filesDir, "recordings/${recording.id}.wav")
        val flac = File(app.filesDir, "recordings/${recording.id}.flac")
        val waited = System.nanoTime()
        while ((wav.exists() || !flac.isFile) && System.nanoTime() - waited < TimeUnit.SECONDS.toNanos(120)) Thread.sleep(200)
        assertTrue("upkeep did not compress the take", flac.isFile && !wav.exists())
        Log.i(
            TAG,
            "compress: ${fromWav.size} B WAV -> ${flac.length()} B FLAC " +
                "(${"%.1f".format(100.0 * flac.length() / fromWav.size)}%), compressed by upkeep " +
                "${(System.nanoTime() - waited) / 1_000_000} ms after the attempt settled",
        )

        val fromFlac = app.recordings.withRequestAudio(recording.id) { it.readBytes() }
        assertArrayEquals("request audio differs after compression", fromWav, fromFlac)
        Log.i(TAG, "compress: request audio from FLAC is byte-identical (${fromFlac.size} B)")
        val platform = platformDecode(flac)
        assertArrayEquals("the platform decoder disagrees", fromWav.copyOfRange(44, fromWav.size), platform)
        Log.i(TAG, "compress: the platform FLAC decoder returns the same ${platform.size} PCM bytes")
        play(flac)

        val retried = transcribe(recording.id)
        assertEquals(RecordingStatus.TRANSCRIBED, retried.status)
        Log.i(TAG, "compress: retry from FLAC; same text as the attempt from the WAV: ${first.rawTranscript == retried.rawTranscript}")
        Log.i(TAG, "compress: first  '${first.rawTranscript?.take(160)}'")
        Log.i(TAG, "compress: retry  '${retried.rawTranscript?.take(160)}'")
    }

    @Test
    fun retention() {
        phase("retention")
        val day = 24L * 60 * 60 * 1000
        var now = System.currentTimeMillis()
        val dir = File(debugDir, "retention-store").apply { deleteRecursively() }
        try {
            val store = RecordingStore(dir) { now }
            fun take(ageDays: Double, seconds: Int, transcribed: Boolean = true): Recording {
                val realNow = now
                now -= (ageDays * day).toLong()
                val recording = store.create()
                now = realNow
                val pcm = ByteArray(seconds * 32_000) { source[it % source.size] }
                WavWriter(store.partialFile(recording)).apply { write(pcm, pcm.size); finish() }
                store.commitAudio(recording, seconds.toDouble())
                return if (transcribed) store.markTranscribed(recording.id, "text") else store.get(recording.id)
            }
            val long = take(ageDays = 60.0, seconds = 600)
            val untranscribed = take(ageDays = 50.0, seconds = 60, transcribed = false)
            val archived = take(ageDays = 45.0, seconds = 60).let { store.setRetentionClass(it.id, RetentionClass.ARCHIVAL) }
            val recent = take(ageDays = 0.2, seconds = 60)
            val sized = (1..3).map { take(ageDays = 10.0 - it, seconds = 120) }

            for (id in store.compressionCandidates()) {
                val started = System.nanoTime()
                val outcome = store.compressAudio(id) as RecordingStore.Compression.Compressed
                Log.i(
                    TAG,
                    "retention: compressed ${outcome.wavBytes} -> ${outcome.flacBytes} B in " +
                        "${(System.nanoTime() - started) / 1_000_000} ms (encode, sync, decode-and-compare, publish)",
                )
            }
            val before = store.withRequestAudio(long.id) { it.readBytes() }
            assertSourceAudio(before, "10-minute take from FLAC")

            val age = store.applyRetention(gate(RetentionPolicy(mapOf(RetentionClass.STANDARD to ClassLimits(maxAgeDays = 30)))))
            Log.i(TAG, "retention: age 30 d -> $age")
            assertEquals(listOf(long.id), age.removed.map { it.id })
            assertEquals(listOf(untranscribed.id to HoldReason.UNTRANSCRIBED), age.held.map { it.id to it.reason })
            assertTrue(File(dir, "${archived.id}.flac").isFile)

            // A 1 MiB limit: every older transcribed take goes, newest first
            // kept; the recent and the untranscribed take are held.
            val size = store.applyRetention(gate(RetentionPolicy(mapOf(RetentionClass.STANDARD to ClassLimits(maxTotalMb = 1)))))
            Log.i(TAG, "retention: size 1 MiB -> $size")
            assertEquals(sized.map { it.id }.toSet(), size.removed.map { it.id }.toSet())
            assertTrue(size.held.any { it.id == untranscribed.id && it.reason == HoldReason.UNTRANSCRIBED })
            assertTrue(File(dir, "${recent.id}.flac").isFile && File(dir, "${untranscribed.id}.flac").isFile)

            val reopened = RecordingStore(dir) { now }
            val removed = reopened.list().filter { it.audioRemoved != null }.map { it.id }.toSet()
            assertEquals((age.removed + size.removed).map { it.id }.toSet(), removed)
            assertEquals("text", reopened.get(long.id).rawTranscript)
            Log.i(TAG, "retention: reopened, ${removed.size} takes listed with their audio removed, transcripts kept")
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun lowdisk() {
        phase("lowdisk")
        val mib = 1024L * 1024
        val saved = FreeSpaceProbe.debugOverride
        var free = 4096 * mib
        FreeSpaceProbe.debugOverride = FreeSpaceProbe { free }
        try {
            free = 100 * mib
            assertEquals(DiskLevel.CRITICAL, app.diskBeforeTake()!!.level)
            free = 500 * mib
            assertEquals(DiskLevel.LOW, app.diskBeforeTake()!!.level)
            free = 4096 * mib
            assertEquals(DiskLevel.OK, app.diskBeforeTake()!!.level)

            val recording = app.recordings.create()
            val ended = CountDownLatch(1)
            val capture = AudioCapture()
            assertEquals(null, capture.start(app, app.recordings.partialFile(recording), onEnded = { ended.countDown() }))
            Thread.sleep(8_000)
            free = 100 * mib
            val low = System.nanoTime()
            assertTrue("the take did not stop itself", ended.await(15, TimeUnit.SECONDS))
            Log.i(TAG, "lowdisk: the take ended itself ${(System.nanoTime() - low) / 1_000_000} ms after the disk ran low")
            val result = stop(capture) as CaptureResult.Completed
            assertTrue(result.stoppedForLowDisk)
            val committed = app.recordings.commitAudio(recording, result.durationSeconds)
            assertEquals(RecordingStatus.PENDING, committed.status)
            val audio = app.recordings.withRequestAudio(recording.id) { it.readBytes() }
            assertSourceAudio(audio, "low-disk take")
            assertTrue(result.durationSeconds >= 8.0)
            Log.i(TAG, "lowdisk: committed ${result.durationSeconds} s, PENDING, request audio intact")
            assertFalse(app.recordings.isPinned(recording.id))
        } finally {
            FreeSpaceProbe.debugOverride = saved
        }
    }

    private fun gate(policy: RetentionPolicy) = object : PolicyGate {
        override fun <T> withPolicy(block: (RetentionPolicy) -> T): T = block(policy)
    }

    /** The WAV's header is the app's own and every sample is the looped test microphone's. */
    private fun assertSourceAudio(wav: ByteArray, what: String) {
        assertArrayEquals(what, WavWriter.header(wav.size - 44L), wav.copyOfRange(0, 44))
        for (i in 44 until wav.size) {
            if (wav[i] != source[(i - 44) % source.size]) throw AssertionError("$what: byte $i differs from the source")
        }
    }

    /**
     * Decodes [file] with the platform: AOSP's FLAC extractor decodes with
     * libFLAC itself and hands out PCM (audio/raw); otherwise the FLAC
     * frames go through the platform's MediaCodec decoder.
     */
    private fun platformDecode(file: File): ByteArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(file.absolutePath)
        val format = extractor.getTrackFormat(0)
        val mime = format.getString(MediaFormat.KEY_MIME)
        Log.i(TAG, "compress: platform extractor track $format")
        extractor.selectTrack(0)
        val out = ByteArrayOutputStream()
        if (mime == MediaFormat.MIMETYPE_AUDIO_RAW) {
            val buffer = java.nio.ByteBuffer.allocate(1 shl 20)
            while (true) {
                val size = extractor.readSampleData(buffer, 0)
                if (size < 0) break
                val chunk = ByteArray(size)
                buffer.position(0)
                buffer.get(chunk)
                out.write(chunk)
                extractor.advance()
            }
            extractor.release()
            return out.toByteArray()
        }
        assertEquals(MediaFormat.MIMETYPE_AUDIO_FLAC, mime)
        val codec = MediaCodec.createDecoderByType(MediaFormat.MIMETYPE_AUDIO_FLAC)
        Log.i(TAG, "compress: platform decoder ${codec.name}")
        try {
            codec.configure(format, null, null, 0)
            codec.start()
            val info = MediaCodec.BufferInfo()
            var inputDone = false
            while (true) {
                if (!inputDone) {
                    val index = codec.dequeueInputBuffer(10_000)
                    if (index >= 0) {
                        val size = extractor.readSampleData(codec.getInputBuffer(index)!!, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(index, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val index = codec.dequeueOutputBuffer(info, 10_000)
                if (index >= 0) {
                    val buffer = codec.getOutputBuffer(index)!!
                    val chunk = ByteArray(info.size)
                    buffer.position(info.offset)
                    buffer.get(chunk)
                    out.write(chunk)
                    codec.releaseOutputBuffer(index, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
            }
        } finally {
            codec.release()
            extractor.release()
        }
        return out.toByteArray()
    }

    private fun stop(capture: AudioCapture): CaptureResult {
        val latch = CountDownLatch(1)
        var outcome: CaptureResult? = null
        capture.stop { outcome = it; latch.countDown() }
        assertTrue(latch.await(30, TimeUnit.SECONDS))
        return outcome!!
    }

    private fun transcribe(id: String): Recording {
        val latch = CountDownLatch(1)
        var settled: Recording? = null
        val config = app.backendSettings.load().copy(engine = TranscriptionEngine.ON_DEVICE)
        val started = System.nanoTime()
        assertTrue(app.transcription.transcribe(id, config) { settled = it; latch.countDown() })
        // Phase-gated (never in CI): an on-device run of a long take, model
        // load included, on a phone; the same bound as DurableAudioDeviceTest.
        assertTrue("the transcription did not settle within 30 minutes", latch.await(30, TimeUnit.MINUTES))
        val result = requireNotNull(settled) { "the transcription settled without a recording" }
        Log.i(TAG, "transcription ${result.status} in ${(System.nanoTime() - started) / 1_000_000} ms ${result.errorMessage ?: ""}")
        return result
    }

    private fun play(file: File) {
        val player = MediaPlayer()
        try {
            InstrumentationRegistry.getInstrumentation().runOnMainSync {
                player.setDataSource(file.absolutePath)
                player.prepare()
                player.setVolume(0f, 0f)
                player.start()
            }
            Thread.sleep(2_000)
            Log.i(TAG, "playback of the FLAC: duration ${player.duration} ms, at ${player.currentPosition} ms after 2 s")
            assertTrue(player.currentPosition > 500)
        } finally {
            player.release()
        }
    }

    companion object {
        private const val TAG = "StarlingRetention"
    }
}
