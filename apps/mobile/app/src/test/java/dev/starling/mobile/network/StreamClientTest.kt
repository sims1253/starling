package dev.starling.mobile.network

import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okio.ByteString
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class StreamClientTest {
    // URL derivation — the same trusted-host policy as the batch client.

    @Test fun streamUrlsFollowTheBatchEndpointAndPolicy() {
        assertEquals("ws://127.0.0.1:8181/stream", streamUrl("http://127.0.0.1:8181", true))
        assertEquals("wss://server.example:8181/stream", streamUrl("https://server.example:8181", false))
        assertEquals("ws://192.168.1.20:8181/stream", streamUrl("http://192.168.1.20:8181/inference", true))
        assertEquals("wss://server.example/stream", streamUrl("https://server.example/v1/audio/transcriptions", false))
        assertEquals("wss://server.example/starling/stream", streamUrl("https://server.example/starling/inference", false))
        assertEquals("wss://server.example/stream", streamUrl("https://server.example/", false))
    }

    @Test fun endpointsThatFailThePolicyHaveNoStreamUrl() {
        assertNull(streamUrl("http://192.168.1.20:8181", false)) // no cleartext opt-in
        assertNull(streamUrl("http://example.com:8181", true)) // public cleartext host
        assertNull(streamUrl("https://user:secret@example.com", false)) // credentials
        assertNull(streamUrl("ftp://example.com", false)) // not HTTP(S)
    }

    // Live flows against a mock /stream server.

    @Test fun connectPartialCommitFinalDeliversTheVerbatimTranscript() {
        val harness = StreamHarness(
            serverBehavior = { text, socket ->
                if (text == COMMIT) {
                    socket.send("""{"type":"final","text":"hello world","duration_s":2.0}""")
                }
            },
        )
        try {
            val session = harness.connect()
            harness.awaitLive()
            val chunk = harness.pcm(64)
            session.onAudio(chunk, chunk.size)
            harness.awaitAudioCount(1)
            assertArrayEquals(chunk, harness.audio().first().toByteArray())

            harness.serverSocket()!!.send("""{"type":"partial","text":"hello"}""")
            harness.awaitEvent { it is StreamEvent.Partial && it.text == "hello" }

            // finish() blocks while the server answers the commit with final.
            val outcome = session.finish()
            assertEquals(CommitOutcome.Final("hello world"), outcome)
            // Idempotent: a second finish returns the same settled outcome.
            assertEquals(outcome, session.finish())
            assertTrue(harness.serverTexts().count { it == COMMIT } == 1)
        } finally {
            harness.close()
        }
    }

    @Test fun connectFailureInterruptsAndFallsBack() {
        val harness = StreamHarness(upgrade = false)
        try {
            val session = harness.connect()
            harness.awaitEvent { it is StreamEvent.Interrupted }
            assertFalse(session.acceptsAudio())
            val outcome = session.finish()
            assertTrue(outcome is CommitOutcome.Fallback)
        } finally {
            harness.close()
        }
    }

    @Test fun midStreamFailureStopsForwardingAndFallsBack() {
        val chunks = AtomicInteger()
        val harness = StreamHarness(
            onBinary = { _, socket ->
                when (chunks.incrementAndGet()) {
                    // The first window transcribes; the server dies when the
                    // next chunk arrives, mid-recording.
                    1 -> socket.send("""{"type":"partial","text":"hel"}""")
                    2 -> socket.cancel()
                }
            },
        )
        try {
            val session = harness.connect()
            harness.awaitLive()
            session.onAudio(harness.pcm(32), 32)
            harness.awaitEvent { it is StreamEvent.Partial }
            session.onAudio(harness.pcm(32), 32)
            harness.awaitEvent { it is StreamEvent.Interrupted }
            assertFalse(session.acceptsAudio())

            // The capture chunk listener keeps being called; forwarding must
            // be a harmless no-op, and Stop falls back to the batch path.
            session.onAudio(harness.pcm(32), 32)
            val outcome = session.finish()
            assertTrue(outcome is CommitOutcome.Fallback)
        } finally {
            harness.close()
        }
    }

    @Test fun bufferCapErrorSurfacesDistinctlyAndFallsBack() {
        val capMessage = "stream buffer limit reached (60 s live buffer); audio ignored until reset"
        val harness = StreamHarness()
        try {
            val session = harness.connect()
            harness.awaitLive()
            harness.serverSocket()!!.send("""{"type":"error","message":"$capMessage"}""")
            harness.awaitEvent {
                it is StreamEvent.Interrupted && it.bufferLimitReached && it.reason == capMessage
            }
            assertFalse(session.acceptsAudio())
            assertEquals(CommitOutcome.Fallback(capMessage), session.finish())
        } finally {
            harness.close()
        }
    }

    @Test fun busyCommitIsRetriedWithinTheFinalWait() {
        val commits = AtomicInteger()
        val harness = StreamHarness(
            serverBehavior = { text, socket ->
                if (text == COMMIT) {
                    // First commit is busy; the retried commit succeeds.
                    if (commits.incrementAndGet() == 1) {
                        socket.send("""{"type":"error","message":"server busy"}""")
                    } else {
                        socket.send("""{"type":"final","text":"after busy"}""")
                    }
                }
            },
            client = StreamClient(
                finalTimeoutMillis = 5_000,
                busyRetryDelaysMillis = longArrayOf(100),
            ),
        )
        try {
            val session = harness.connect()
            harness.awaitLive()
            val outcome = session.finish()
            assertEquals(CommitOutcome.Final("after busy"), outcome)
            assertTrue(harness.serverTexts().count { it == COMMIT } == 2)
        } finally {
            harness.close()
        }
    }

    @Test fun finalTimeoutFallsBackInsteadOfHanging() {
        val harness = StreamHarness(
            // The server accepts the commit and never answers.
            client = StreamClient(finalTimeoutMillis = 300),
        )
        try {
            val session = harness.connect()
            harness.awaitLive()
            val outcome = session.finish()
            assertTrue(outcome is CommitOutcome.Fallback)
            assertTrue((outcome as CommitOutcome.Fallback).reason.contains("timed out"))
        } finally {
            harness.close()
        }
    }

    @Test fun audioSentBeforeTheSocketOpensArrivesFirstAndInOrder() {
        // The upgrade response is delayed, so both chunks go through the
        // connect backlog; the flush must preserve the recording's order or
        // the stream's final transcript would start mid-recording.
        val harness = StreamHarness(headersDelayMillis = 400)
        try {
            val session = harness.connect()
            val first = harness.pcm(16)
            val second = harness.pcm(16)
            session.onAudio(first, first.size)
            session.onAudio(second, second.size)
            harness.awaitAudioCount(2)
            assertArrayEquals(first, harness.audio()[0].toByteArray())
            assertArrayEquals(second, harness.audio()[1].toByteArray())
        } finally {
            harness.close()
        }
    }

    @Test fun backlogChunksAlwaysPrecedeChunksSentAfterLive() {
        // Regression guard for the atomic backlog drain: a chunk buffered
        // before the socket opened must never reach the server behind a
        // chunk sent after the Live event, whatever the interleaving.
        val harness = StreamHarness(headersDelayMillis = 300)
        try {
            val session = harness.connect()
            val early = harness.pcm(8)
            session.onAudio(early, early.size)
            harness.awaitLive()
            val late = harness.pcm(8)
            session.onAudio(late, late.size)
            harness.awaitAudioCount(2)
            assertArrayEquals(early, harness.audio()[0].toByteArray())
            assertArrayEquals(late, harness.audio()[1].toByteArray())
        } finally {
            harness.close()
        }
    }

    @Test fun keepalivePingsAreAnsweredAndKeepTheSessionAlive() {
        val harness = StreamHarness(
            serverBehavior = { text, socket ->
                if (text == PING) socket.send("""{"type":"pong"}""")
            },
            client = StreamClient(keepaliveIntervalMillis = 100),
        )
        try {
            val session = harness.connect()
            harness.awaitLive()
            harness.awaitServerTexts(2) { texts -> texts.count { it == PING } >= 2 }
            assertTrue(session.acceptsAudio())
        } finally {
            harness.close()
        }
    }

    @Test fun closeAbandonsWithoutCommitting() {
        val harness = StreamHarness()
        try {
            val session = harness.connect()
            harness.awaitLive()
            session.close()
            val outcome = session.finish()
            assertTrue(outcome is CommitOutcome.Fallback)
            assertTrue(harness.serverTexts().none { it == COMMIT })
        } finally {
            harness.close()
        }
    }

    private companion object {
        const val COMMIT = """{"type":"commit"}"""
        const val PING = """{"type":"ping"}"""
    }

    /** One mock /stream server plus a recording client-events sink. */
    private class StreamHarness(
        serverBehavior: (String, WebSocket) -> Unit = { _, _ -> },
        private val onBinary: (ByteString, WebSocket) -> Unit = { _, _ -> },
        private val upgrade: Boolean = true,
        private val headersDelayMillis: Long = 0,
        client: StreamClient? = null,
    ) : AutoCloseable {
        val server = MockWebServer()
        private val serverStream = ServerStream(serverBehavior, onBinary)
        private val clientEvents = mutableListOf<StreamEvent>()
        private val streamClient = client ?: StreamClient(finalTimeoutMillis = 5_000)

        init {
            if (!upgrade) {
                server.enqueue(MockResponse().setResponseCode(404))
            } else {
                val response = MockResponse().withWebSocketUpgrade(serverStream)
                if (headersDelayMillis > 0) {
                    response.setHeadersDelay(headersDelayMillis, TimeUnit.MILLISECONDS)
                }
                server.enqueue(response)
            }
            server.start()
        }

        fun connect(): StreamSession =
            streamClient.connect("ws://127.0.0.1:${server.port}/stream") { event ->
                synchronized(clientEvents) { clientEvents.add(event) }
            }

        fun serverSocket(): WebSocket? = serverStream.socket

        fun audio(): List<ByteString> = serverStream.audio.toList()

        fun serverTexts(): List<String> = serverStream.texts.toList()

        fun eventsSnapshot(): List<StreamEvent> = synchronized(clientEvents) { clientEvents.toList() }

        fun pcm(size: Int): ByteArray = ByteArray(size) { it.toByte() }

        fun awaitLive() {
            awaitEvent { it is StreamEvent.Live }
        }

        fun awaitEvent(predicate: (StreamEvent) -> Boolean) {
            await("stream event $predicate") {
                synchronized(clientEvents) { clientEvents.any(predicate) }
            }
        }

        fun awaitAudioCount(count: Int) {
            await("$count audio frames") { serverStream.audio.size >= count }
        }

        fun awaitServerTexts(count: Int, predicate: (List<String>) -> Boolean) {
            await("$count server text frames") { predicate(serverStream.texts.toList()) }
        }

        private fun await(what: String, condition: () -> Boolean) {
            val deadline = System.currentTimeMillis() + 5_000
            while (System.currentTimeMillis() < deadline) {
                if (condition()) return
                Thread.sleep(20)
            }
            assertTrue("Timed out waiting for $what", condition())
        }

        override fun close() {
            runCatching { server.shutdown() }
        }

        private class ServerStream(
            private val onText: (String, WebSocket) -> Unit,
            private val onBinary: (ByteString, WebSocket) -> Unit,
        ) : WebSocketListener() {
            val audio = mutableListOf<ByteString>()
            val texts = mutableListOf<String>()

            @Volatile
            var socket: WebSocket? = null

            override fun onOpen(webSocket: WebSocket, response: Response) {
                socket = webSocket
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                synchronized(audio) { audio.add(bytes) }
                onBinary(bytes, webSocket)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                synchronized(texts) { texts.add(text) }
                onText(text, webSocket)
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null)
            }
        }
    }
}
