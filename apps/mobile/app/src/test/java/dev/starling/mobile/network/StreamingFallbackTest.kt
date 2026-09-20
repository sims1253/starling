package dev.starling.mobile.network

import dev.starling.mobile.audio.WavWriter
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import okio.ByteString
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * The fallback contract of the streaming integration at the level the
 * coordinator applies it: when the socket dies mid-recording, the session
 * resolves finish() to Fallback and the ordinary batch client then
 * transcribes the same saved WAV that was being written all along. The
 * Android glue in TranscriptionCoordinator (executor, store marks, main
 * thread delivery) is the same thin, untested layer as its batch path.
 */
class StreamingFallbackTest {
    @Test fun midStreamFailureStillTranscribesTheSavedWavThroughTheBatchPath() {
        val server = MockWebServer()
        val wav = savedWav()
        // The server accepts the stream, transcribes one window, and dies
        // the moment the second chunk arrives.
        var sawFirstChunk = false
        val serverStream = object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                webSocket.send("""{"type":"partial","text":"hello"}""")
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                if (sawFirstChunk) {
                    webSocket.cancel()
                } else {
                    sawFirstChunk = true
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null)
            }
        }
        server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse = when {
                request.path == "/stream" -> MockResponse().withWebSocketUpgrade(serverStream)
                request.path == "/inference" && request.method == "POST" ->
                    MockResponse().setResponseCode(200)
                        .setBody("""{"text":"batch transcript of the saved wav"}""")
                else -> MockResponse().setResponseCode(404)
            }
        }
        server.start()
        try {
            val events = mutableListOf<StreamEvent>()
            val session = StreamClient(finalTimeoutMillis = 5_000)
                .connect("ws://127.0.0.1:${server.port}/stream") { event ->
                    synchronized(events) { events.add(event) }
                }
            val pcm = ByteArray(64) { it.toByte() }
            session.onAudio(pcm, pcm.size)
            session.onAudio(pcm, pcm.size)
            await("the stream to be interrupted") {
                synchronized(events) { events.any { it is StreamEvent.Interrupted } }
            }
            assertFalse(session.acceptsAudio())

            // Stop: the WAV is already durable; the stream cannot be trusted.
            val outcome = session.finish()
            assertTrue(outcome is CommitOutcome.Fallback)

            // The coordinator's fallback branch: batch-upload the saved WAV.
            val result = InferenceClient().transcribe(
                wav,
                BackendConfig("http://127.0.0.1:${server.port}", true),
            )
            assertEquals(InferenceResult.Success("batch transcript of the saved wav"), result)

            // The upload carried the saved WAV itself, like any retry.
            val upload = pollForUpload(server)
            assertEquals("/inference", upload.path)
            assertEquals("POST", upload.method)
            val body = upload.body.readByteArray().toString(Charsets.UTF_8)
            assertTrue(body.contains("""name="file"; filename="recording.wav""""))
            assertTrue(body.contains("audio/wav"))
        } finally {
            runCatching { server.shutdown() }
            wav.delete()
        }
    }

    /** A finalized 16 kHz mono WAV exactly like a settled capture leaves. */
    private fun savedWav(): File {
        val wav = File.createTempFile("starling-fallback-", ".wav")
        val writer = WavWriter(wav)
        val payload = ByteArray(320) { (it % 251).toByte() }
        writer.write(payload, payload.size)
        writer.finish()
        return wav
    }

    private fun pollForUpload(server: MockWebServer): RecordedRequest {
        val deadline = System.currentTimeMillis() + 5_000
        while (System.currentTimeMillis() < deadline) {
            val request = server.takeRequest(100, TimeUnit.MILLISECONDS)
            if (request != null && request.method == "POST") return request
        }
        throw AssertionError("The batch upload of the saved WAV never arrived")
    }

    private fun await(what: String, condition: () -> Boolean) {
        val deadline = System.currentTimeMillis() + 5_000
        while (System.currentTimeMillis() < deadline) {
            if (condition()) return
            Thread.sleep(20)
        }
        assertTrue("Timed out waiting for $what", condition())
    }
}
