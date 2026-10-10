package dev.starling.mobile

import android.media.MediaPlayer
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.starling.mobile.audio.AudioCapture
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.engine.WavPcm
import dev.starling.mobile.network.TranscriptionEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Named-device acceptance for #356, in the debug app's own process: the
 * capture, store and transcription path every entry point shares, with the
 * debug test microphone (files/debug/test-mic.wav, looped at real-time pace)
 * and the injected transcription failure (files/debug/fail-transcription).
 * Phases run as separate instrumentations so the process really dies in
 * between; the host script drives them:
 *
 *   record     records `minutes`, stops, fails transcription (injected)
 *   interrupt  starts a take and blocks; the host kills the process
 *   verify     after relaunch: both takes restored with truthful status,
 *              played back, and retranscribed with each installed model
 *
 * Opt-in: `am instrument -e durable <phase> [-e minutes N]
 * [-e killedBytes B -e killedConfirmedBytes C] ...`, the last two (required
 * by verify) being the partial WAV's payload size and header data size the host read
 * after the kill.
 */
@RunWith(AndroidJUnit4::class)
class DurableAudioDeviceTest {
    private val app = ApplicationProvider.getApplicationContext<StarlingApplication>()
    private val arguments = InstrumentationRegistry.getArguments()
    private val debugDir = File(app.filesDir, "debug")

    private fun phase(name: String) =
        assumeTrue("set -e durable $name to run", arguments.getString("durable") == name)

    @Test
    fun record() {
        phase("record")
        assertTrue("files/debug/test-mic.wav is required", File(debugDir, "test-mic.wav").isFile)
        assertTrue("files/debug/fail-transcription is required", File(debugDir, "fail-transcription").exists())
        val minutes = arguments.getString("minutes")?.toDouble() ?: 10.0

        val recording = app.recordings.create()
        File(debugDir, "record-id").writeText(recording.id)
        val capture = AudioCapture()
        assertEquals(null, capture.start(app, app.recordings.partialFile(recording)))
        Thread.sleep((minutes * 60_000).toLong())
        val result = stop(capture)
        val completed = result as? CaptureResult.Completed ?: throw AssertionError("capture: $result")
        Log.i(TAG, "record: captured ${completed.durationSeconds} s")
        app.recordings.commitAudio(recording, completed.durationSeconds)

        val settled = transcribe(recording.id, null)
        Log.i(TAG, "record: after stop ${settled.status} — ${settled.errorMessage}")
        assertEquals(RecordingStatus.FAILED, settled.status)
        assertTrue(app.recordings.audioFile(settled).length() > 44 + 32_000L * 60 * minutes * 0.99)
    }

    @Test
    fun interrupt() {
        phase("interrupt")
        val recording = app.recordings.create()
        File(debugDir, "interrupt-id").writeText(recording.id)
        val capture = AudioCapture()
        assertEquals(null, capture.start(app, app.recordings.partialFile(recording)))
        Log.i(TAG, "interrupt: recording ${recording.id}; kill the process now")
        // The host kills the process while this take is recording.
        Thread.sleep(TimeUnit.HOURS.toMillis(1))
    }

    @Test
    fun verify() {
        phase("verify")
        File(debugDir, "fail-transcription").delete()
        val models = app.onDeviceEngine.installedModels().map { it.name }
        Log.i(TAG, "verify: installed models $models")
        // Retry with the same and with another model.
        assertTrue("install at least two on-device models", models.size >= 2)
        val source = WavPcm.decodePcm16(File(debugDir, "test-mic.wav"))!!.pcm

        for (key in listOf("record-id", "interrupt-id")) {
            val id = File(debugDir, key).readText()
            val restored = app.recordings.get(id)
            Log.i(
                TAG,
                "verify $key: status=${restored.status} duration=${restored.durationSeconds} " +
                    "error=${restored.errorMessage} recovery=${restored.recovery}",
            )
            val audio = app.recordings.audioFile(restored)
            val pcm = WavPcm.decodePcm16(audio)!!.pcm
            if (key == "record-id") {
                // Failed after Stop; an earlier verify run may have retried it.
                assertEquals(
                    if (restored.revisions.isEmpty()) RecordingStatus.FAILED else RecordingStatus.TRANSCRIBED,
                    restored.status,
                )
                assertTrue(restored.durationSeconds >= 600.0)
            } else {
                assertEquals(RecordingStatus.PENDING, restored.status)
                val recovery = restored.recovery!!
                assertEquals(pcm.size / 32_000.0, recovery.recoveredSeconds, 1e-6)
                assertTrue(recovery.confirmedSeconds <= recovery.recoveredSeconds)
                // The watermark trails the last write by about two checkpoints.
                assertTrue(recovery.recoveredSeconds - recovery.confirmedSeconds < 5.0)
                // Measured by the host on the partial WAV after the kill
                // (its size and header data size), so recovery is checked
                // against what the dead process left, not against itself.
                val killedBytes = arguments.getString("killedBytes")?.toInt()
                    ?: throw AssertionError("pass -e killedBytes (the partial WAV's size - 44 after the kill)")
                val killedConfirmedBytes = arguments.getString("killedConfirmedBytes")?.toInt()
                    ?: throw AssertionError("pass -e killedConfirmedBytes (its header's data size after the kill)")
                assertEquals(killedBytes, pcm.size)
                assertEquals(killedConfirmedBytes / 32_000.0, recovery.confirmedSeconds, 1e-6)
            }
            // Every recovered sample is the test microphone's, in order.
            assertEquals(restored.durationSeconds, pcm.size / 32_000.0, 1e-6)
            for (i in pcm.indices) {
                if (pcm[i] != source[i % source.size]) throw AssertionError("$key: sample byte $i differs from the source")
            }
            Log.i(TAG, "verify $key: ${pcm.size} PCM bytes match the looped source")
            assertTrue(restored in app.recordings.list())
            play(audio)

            var revisions = restored.revisions.size
            for (model in models) {
                val settled = transcribe(id, model)
                Log.i(TAG, "verify $key with $model: ${settled.status} ${settled.errorMessage ?: ""}")
                assertEquals(RecordingStatus.TRANSCRIBED, settled.status)
                revisions++
                assertEquals(revisions, settled.revisions.size)
                assertEquals(model, settled.revisions.last().model)
                Log.i(TAG, "verify $key with $model: ${settled.revisions.last().text.take(200)}")
            }
        }
    }

    private fun stop(capture: AudioCapture): CaptureResult {
        val latch = CountDownLatch(1)
        var outcome: CaptureResult? = null
        capture.stop { outcome = it; latch.countDown() }
        assertTrue(latch.await(30, TimeUnit.SECONDS))
        return outcome!!
    }

    private fun transcribe(id: String, model: String?): Recording {
        val latch = CountDownLatch(1)
        var settled: Recording? = null
        val config = app.backendSettings.load().copy(engine = TranscriptionEngine.ON_DEVICE, onDeviceModel = model)
        val started = System.nanoTime()
        assertTrue(app.transcription.transcribe(id, config) { settled = it; latch.countDown() })
        assertTrue(latch.await(30, TimeUnit.MINUTES))
        Log.i(TAG, "transcription took ${(System.nanoTime() - started) / 1_000_000} ms")
        return settled!!
    }

    /** Plays a few seconds of the saved audio and checks that playback advances. */
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
            val position = player.currentPosition
            Log.i(TAG, "playback: duration ${player.duration} ms, at $position ms after 2 s")
            assertTrue(position > 500)
            assertTrue(player.duration > 0)
        } finally {
            player.release()
        }
    }

    companion object {
        private const val TAG = "StarlingDurable"
    }
}
