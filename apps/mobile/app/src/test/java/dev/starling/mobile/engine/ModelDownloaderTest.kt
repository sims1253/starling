package dev.starling.mobile.engine

import java.io.File
import java.nio.file.Files
import java.security.MessageDigest
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.SocketPolicy
import okio.Buffer
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/** Resume, restart and verification behavior of [ModelDownloader] against a local server. */
class ModelDownloaderTest {
    private lateinit var server: MockWebServer
    private lateinit var directory: File
    private val payload = ByteArray(3 * 1024 * 1024 + 17) { (it * 31 + 7).toByte() }

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()
        directory = Files.createTempDirectory("starling-download-test").toFile()
    }

    @After
    fun tearDown() {
        server.shutdown()
        directory.deleteRecursively()
    }

    private fun spec(sha: String = sha256(payload)) = ModelDownload(
        displayName = "test",
        url = server.url("/model.gguf").toString(),
        sizeBytes = payload.size.toLong(),
        sha256 = sha,
    )

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    private fun body(bytes: ByteArray) = Buffer().write(bytes)

    @Test
    fun fullDownloadIsVerified() {
        server.enqueue(MockResponse().setBody(body(payload)))
        val partial = File(directory, "model.part")
        var last = 0L

        val result = ModelDownloader().download(spec(), partial) { bytes, _ -> last = bytes }

        assertTrue("$result", result is ModelDownloader.Result.Done)
        assertArrayEquals(payload, partial.readBytes())
        assertEquals(payload.size.toLong(), last)
        assertNull(server.takeRequest().getHeader("Range"))
    }

    @Test
    fun partialDownloadResumesWithARangeRequest() {
        val half = payload.size / 2
        val partial = File(directory, "model.part").apply { writeBytes(payload.copyOfRange(0, half)) }
        server.enqueue(
            MockResponse()
                .setResponseCode(206)
                .setHeader("Content-Range", "bytes $half-${payload.size - 1}/${payload.size}")
                .setBody(body(payload.copyOfRange(half, payload.size))),
        )

        val result = ModelDownloader().download(spec(), partial) { _, _ -> }

        assertTrue("$result", result is ModelDownloader.Result.Done)
        assertEquals("bytes=$half-", server.takeRequest().getHeader("Range"))
        assertArrayEquals(payload, partial.readBytes())
    }

    @Test
    fun ignoredRangeStartsOver() {
        val partial = File(directory, "model.part").apply { writeBytes(ByteArray(1000) { 1 }) }
        server.enqueue(MockResponse().setBody(body(payload)))   // 200: the whole file

        val result = ModelDownloader().download(spec(), partial) { _, _ -> }

        assertTrue("$result", result is ModelDownloader.Result.Done)
        assertArrayEquals(payload, partial.readBytes())
    }

    @Test
    fun checksumMismatchDiscardsTheFile() {
        server.enqueue(MockResponse().setBody(body(payload)))
        val partial = File(directory, "model.part")

        val result = ModelDownloader().download(spec(sha = "00".repeat(32)), partial) { _, _ -> }

        assertTrue("$result", result is ModelDownloader.Result.Failed)
        assertFalse("a corrupt download must not survive", partial.exists())
    }

    @Test
    fun droppedConnectionKeepsThePartialForAResume() {
        server.enqueue(
            MockResponse()
                .setBody(body(payload))
                .setSocketPolicy(SocketPolicy.DISCONNECT_DURING_RESPONSE_BODY),
        )
        val partial = File(directory, "model.part")

        val result = ModelDownloader().download(spec(), partial) { _, _ -> }

        assertTrue("$result", result is ModelDownloader.Result.Failed)
        assertTrue("received bytes are kept", partial.length() in 1 until payload.size)
        val have = partial.length().toInt()
        assertArrayEquals(payload.copyOfRange(0, have), partial.readBytes())
    }

    @Test
    fun serverErrorIsReported() {
        server.enqueue(MockResponse().setResponseCode(404))

        val result = ModelDownloader().download(spec(), File(directory, "model.part")) { _, _ -> }

        assertTrue("$result", result is ModelDownloader.Result.Failed)
        assertTrue((result as ModelDownloader.Result.Failed).reason.contains("404"))
    }

    @Test
    fun contentRangeParsing() {
        assertEquals(100L, ModelDownloader.rangeStart("bytes 100-199/200"))
        assertEquals(-1L, ModelDownloader.rangeStart(null))
        assertEquals(-1L, ModelDownloader.rangeStart("bytes */200"))
    }
}
