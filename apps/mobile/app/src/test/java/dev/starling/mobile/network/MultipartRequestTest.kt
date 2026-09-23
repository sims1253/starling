package dev.starling.mobile.network

import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.ServerSocket
import java.net.Socket
import java.nio.file.Files
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MultipartRequestTest {
    @Test
    fun openAiMultipartContainsRequiredFieldsOnLoopback() {
        val audio = Files.createTempFile("starling-mobile", ".wav").toFile()
        val server = ServerSocket(0)
        val capturedBody = AtomicReference<ByteArray?>()
        val serverFailure = AtomicReference<Throwable?>()
        val thread = Thread {
            try {
                server.accept().use { socket ->
                    val input = BufferedInputStream(socket.getInputStream())
                    val headers = readUntilHeaderEnd(input).toString(Charsets.ISO_8859_1)
                    val contentLength = headers.lineSequence()
                        .first { it.startsWith("Content-Length:", ignoreCase = true) }
                        .substringAfter(':')
                        .trim()
                        .toInt()
                    capturedBody.set(readExactly(input, contentLength))
                    socket.getOutputStream().use { output ->
                        output.write("HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".toByteArray())
                        output.flush()
                    }
                }
            } catch (failure: Throwable) {
                serverFailure.set(failure)
            }
        }

        try {
            audio.writeBytes(byteArrayOf(0, 1, 2, 3))
            val boundary = "StarlingTestBoundary"
            val request = MultipartRequest(
                boundary = boundary,
                audioFile = audio,
                fields = listOf("model" to "parakeet", "response_format" to "json"),
            )
            thread.start()
            Socket("127.0.0.1", server.localPort).use { socket ->
                val output = socket.getOutputStream()
                output.write(
                    (
                        "POST /v1/audio/transcriptions HTTP/1.1\r\n" +
                            "Host: 127.0.0.1\r\n" +
                            "Content-Type: multipart/form-data; boundary=$boundary\r\n" +
                            "Content-Length: ${request.contentLength}\r\n" +
                            "Connection: close\r\n\r\n"
                        ).toByteArray(Charsets.ISO_8859_1),
                )
                request.writeTo(output)
                output.flush()
                socket.shutdownOutput()
                socket.getInputStream().readBytes()
            }
            thread.join(5_000)

            assertNull(serverFailure.get())
            val body = capturedBody.get()
            assertNotNull(body)
            val bodyText = body!!.toString(Charsets.UTF_8)
            assertTrue(bodyText.contains("name=\"model\"\r\n\r\nparakeet"))
            assertTrue(bodyText.contains("name=\"response_format\"\r\n\r\njson"))
            assertTrue(bodyText.contains("name=\"file\"; filename=\"recording.wav\""))
            assertEquals("StarlingTestBoundary", bodyText.substringAfter("--").substringBefore("\r\n"))
        } finally {
            server.close()
            thread.join(5_000)
            audio.delete()
        }
    }

    private fun readUntilHeaderEnd(input: BufferedInputStream): ByteArray {
        val output = ByteArrayOutputStream()
        var matched = 0
        val marker = byteArrayOf('\r'.code.toByte(), '\n'.code.toByte(), '\r'.code.toByte(), '\n'.code.toByte())
        while (matched < marker.size) {
            val value = input.read()
            check(value >= 0) { "loopback server received truncated headers" }
            output.write(value)
            matched = if (value == marker[matched].toInt()) matched + 1 else 0
        }
        return output.toByteArray()
    }

    private fun readExactly(input: BufferedInputStream, length: Int): ByteArray {
        val output = ByteArray(length)
        var offset = 0
        while (offset < length) {
            val count = input.read(output, offset, length - offset)
            check(count >= 0) { "loopback server received truncated body" }
            offset += count
        }
        return output
    }
}
