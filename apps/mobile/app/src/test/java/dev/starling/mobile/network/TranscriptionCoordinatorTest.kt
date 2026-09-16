package dev.starling.mobile.network

import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.engine.OnDeviceBackend
import dev.starling.mobile.engine.OnDeviceEngine
import dev.starling.mobile.storage.RecordingStore
import java.io.File
import java.nio.file.Files
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TranscriptionCoordinatorTest {
    @Test
    fun deduplicatedTranscribeSurfacesCurrentStateAndSendsOneRequest() {
        val server = MockWebServer()
        // The body delay keeps the first request in flight while the second
        // transcribe() call is deduplicated against it.
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setBody("""{"text":"exact text"}""")
                .setBodyDelay(2, TimeUnit.SECONDS),
        )
        server.start()
        val directory = Files.createTempDirectory("starling-coordinator-test").toFile()
        val store = RecordingStore(directory)
        val coordinator = TranscriptionCoordinator(
            store,
            loadConfig = { BackendConfig("http://127.0.0.1:${server.port}", true) },
            onDevice = OnDeviceBackend(OnDeviceEngine(directory)),
            postToMain = { it.run() },
        )
        try {
            val recording = committedRecording(store)

            // First call: queued, the request starts on the executor.
            val firstOutcome = AtomicReference<Recording?>()
            val firstDone = CountDownLatch(1)
            coordinator.transcribe(recording.id) { completed ->
                firstOutcome.set(completed)
                firstDone.countDown()
            }

            // Second (double-tapped Retry) call while the first is in flight:
            // it must still report the durable state instead of returning
            // silently, which is what left the UI stuck on "Contacting…".
            val secondOutcome = AtomicReference<Recording?>()
            val secondDone = CountDownLatch(1)
            coordinator.transcribe(recording.id) { completed ->
                secondOutcome.set(completed)
                secondDone.countDown()
            }
            assertTrue(secondDone.await(1, TimeUnit.SECONDS))
            assertEquals(RecordingStatus.TRANSCRIBING, secondOutcome.get()?.status)

            // The original request delivers the final outcome to its caller.
            assertTrue(firstDone.await(15, TimeUnit.SECONDS))
            assertEquals(RecordingStatus.TRANSCRIBED, firstOutcome.get()?.status)
            assertEquals("exact text", firstOutcome.get()?.rawTranscript)
            assertEquals(1, server.requestCount)
        } finally {
            coordinator.shutdown()
            server.shutdown()
            directory.deleteRecursively()
        }
    }

    /** A finalized WAV committed like AudioCapture.stop() would leave it. */
    private fun committedRecording(store: RecordingStore): Recording {
        val recording = store.create()
        val writer = WavWriter(store.partialFile(recording))
        writer.write(ByteArray(64), 64)
        writer.finish()
        return store.commitAudio(recording, 0.004)
    }
}
