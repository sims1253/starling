package dev.starling.mobile.network

import java.io.File
import java.io.OutputStream

/**
 * Dependency-free multipart encoder for Starling transcription requests.
 * Keeping this separate makes the wire fields testable on
 * the JVM without trying to execute Android's mocked JSON classes.
 */
internal class MultipartRequest(
    private val boundary: String,
    private val audioFile: File,
    private val fields: List<Pair<String, String>>,
) {
    private val prefix: ByteArray = buildString {
        fields.forEach { (name, value) ->
            append("--$boundary\r\n")
            append("Content-Disposition: form-data; name=\"$name\"\r\n\r\n")
            append(value)
            append("\r\n")
        }
        append("--$boundary\r\n")
        append("Content-Disposition: form-data; name=\"file\"; filename=\"recording.wav\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
    }.toByteArray(Charsets.UTF_8)
    private val suffix = "\r\n--$boundary--\r\n".toByteArray(Charsets.UTF_8)

    val contentLength: Long
        get() = prefix.size.toLong() + audioFile.length() + suffix.size

    fun writeTo(output: OutputStream) {
        output.write(prefix)
        audioFile.inputStream().use { input -> input.copyTo(output, BUFFER_SIZE) }
        output.write(suffix)
    }

    companion object {
        private const val BUFFER_SIZE = 16 * 1024
    }
}
