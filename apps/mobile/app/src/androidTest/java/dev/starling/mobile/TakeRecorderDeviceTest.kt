package dev.starling.mobile

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.speech.RecognizerIntent
import android.widget.Button
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import dev.starling.mobile.audio.PcmSource
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.engine.OnDeviceBackend
import dev.starling.mobile.engine.WavPcm
import dev.starling.mobile.transcription.TakeRecorder
import dev.starling.mobile.transcription.TranscriptionCoordinator
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** The shared take module with the real capture, store and coordinator. */
@RunWith(AndroidJUnit4::class)
class TakeRecorderDeviceTest {
    @get:Rule
    val microphone: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.RECORD_AUDIO)
    private val app = ApplicationProvider.getApplicationContext<StarlingApplication>()
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val pcm = ByteArray(64) { (it * 3).toByte() }

    private class Source(private val pcm: ByteArray) : PcmSource {
        val written = CountDownLatch(1)
        @Volatile private var stopped = false
        private var sent = false
        override fun start() = Unit
        override fun read(buffer: ByteArray, count: Int): Int {
            if (!sent) {
                check(count >= pcm.size)
                sent = true
                pcm.copyInto(buffer)
                return pcm.size
            }
            // A second read means the capture has written the first chunk.
            written.countDown()
            while (!stopped) Thread.sleep(5)
            return 0
        }
        override fun stop() { stopped = true }
        override fun release() { stopped = true }
    }

    @Test
    fun inferenceFailureKeepsTheExactCaptureForRetry() {
        val previous = PcmSource.debugSource
        val source = Source(pcm)
        val take = AtomicReference<TakeRecorder.Take>()
        val completed = AtomicReference<Recording>()
        val settled = CountDownLatch(1)
        val coordinator = TranscriptionCoordinator(
            app.recordings, OnDeviceBackend(app.onDeviceEngine), injectedFailure = { "take test failure" },
        )
        try {
            PcmSource.debugSource = { source }
            instrumentation.runOnMainSync {
                val started = TakeRecorder(app, app.recordings, coordinator).start()
                assertTrue("capture did not start: $started", started is TakeRecorder.Start.Started)
                take.set((started as TakeRecorder.Start.Started).take)
            }
            assertTrue(source.written.await(10, TimeUnit.SECONDS))
            instrumentation.runOnMainSync {
                take.get().stop(onTranscribed = { completed.set(it); settled.countDown() })
            }
            assertTrue("transcription did not settle", settled.await(10, TimeUnit.SECONDS))
            assertEquals(RecordingStatus.FAILED, completed.get().status)
            assertEquals("take test failure", completed.get().errorMessage)
            assertSavedAudio(take.get().recording.id)
        } finally {
            instrumentation.runOnMainSync { take.get()?.stop(transcribe = false) }
            take.get()?.let { app.recordings.delete(it.recording.id) }
            coordinator.shutdown()
            PcmSource.debugSource = previous
        }
    }

    @Test
    fun popupCancelRetainsAudioWithoutAnInferenceAttempt() {
        val previous = PcmSource.debugSource
        val source = Source(pcm)
        val before = app.recordings.list().map { it.id }.toSet()
        var id: String? = null
        try {
            PcmSource.debugSource = { source }
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).setPackage(app.packageName)
            ActivityScenario.launchActivityForResult<RecognizeSpeechActivity>(intent).use { scenario ->
                assertTrue(source.written.await(10, TimeUnit.SECONDS))
                id = app.recordings.list().single { it.id !in before }.id
                scenario.onActivity {
                    assertTrue(it.capturing)
                    it.findViewById<Button>(R.id.recognize_cancel_button).performClick()
                }
                assertEquals(Activity.RESULT_CANCELED, scenario.result.resultCode)
            }
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
            while (app.recordings.get(id!!).status == RecordingStatus.RECORDING && System.nanoTime() < deadline) {
                Thread.sleep(10)
            }
            val saved = app.recordings.get(id!!)
            assertEquals(RecordingStatus.PENDING, saved.status)
            assertEquals(0, saved.attempts)
            assertSavedAudio(id!!)
        } finally {
            id?.let { app.recordings.delete(it) }
            PcmSource.debugSource = previous
        }
    }

    private fun assertSavedAudio(id: String) {
        app.recordings.withRequestAudio(id) { file ->
            val decoded = WavPcm.decodePcm16(file) ?: throw AssertionError("saved audio is unreadable")
            assertEquals(16_000, decoded.sampleRate)
            assertArrayEquals(pcm, decoded.pcm)
        }
    }
}
