package dev.starling.mobile.network

import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID
import java.util.concurrent.TimeUnit

sealed interface InferenceResult {
    data class Success(val rawTranscript: String) : InferenceResult
    data class Failure(val message: String, val retryable: Boolean) : InferenceResult
}

/** Minimal platform transport for the Starling multipart HTTP contract. */
internal fun inferenceUrl(endpoint: String, protocol: BackendProtocol): String {
    val base = endpoint.trimEnd('/')
    // A full route is useful with the repository's optional OpenAI-shaped
    // adapter. A plain server URL continues to use the documented route.
    return when (protocol) {
        BackendProtocol.STARLING -> if (
            base.endsWith("/inference", ignoreCase = true) ||
            base.endsWith("/transcribe", ignoreCase = true)
        ) {
            base
        } else {
            "$base/inference"
        }
        BackendProtocol.OPENAI -> when {
            base.endsWith("/transcriptions", ignoreCase = true) -> base
            base.endsWith("/v1", ignoreCase = true) -> "$base/audio/transcriptions"
            else -> "$base/v1/audio/transcriptions"
        }
    }
}

class InferenceClient {
    fun transcribe(audioFile: File, config: BackendConfig): InferenceResult {
        if (!audioFile.isFile) return InferenceResult.Failure("The recording audio is missing", false)
        if (audioFile.length() > MAX_UPLOAD_BYTES) {
            return InferenceResult.Failure("The recording is larger than the server upload limit", false)
        }

        val validation = EndpointPolicy.validate(config.endpoint, config.allowTrustedLanHttp)
        if (validation !is EndpointValidation.Valid) {
            return InferenceResult.Failure(
                (validation as EndpointValidation.Invalid).message,
                false,
            )
        }
        if (config.protocol == BackendProtocol.OPENAI && config.model.trim().isEmpty()) {
            return InferenceResult.Failure("Enter the model name served by the backend", false)
        }

        var lastFailure: InferenceResult.Failure? = null
        repeat(MAX_ATTEMPTS) { attempt ->
            val result = runCatching { transcribeOnce(audioFile, validation.endpoint, config) }
                .getOrElse { exception ->
                    InferenceResult.Failure(
                        exception.message ?: "Unable to reach the Starling backend",
                        true,
                    )
                }
            if (result is InferenceResult.Success) return result
            lastFailure = result as InferenceResult.Failure
            if (!lastFailure!!.retryable || attempt == MAX_ATTEMPTS - 1) return lastFailure!!
            try {
                Thread.sleep(RETRY_DELAYS_MILLIS[attempt])
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                return InferenceResult.Failure("Transcription retry was interrupted", true)
            }
        }
        return lastFailure ?: InferenceResult.Failure("Transcription failed", true)
    }

    private fun transcribeOnce(audioFile: File, endpoint: String, config: BackendConfig): InferenceResult {
        val boundary = "----StarlingMobile${UUID.randomUUID()}"
        val fields = if (config.protocol == BackendProtocol.OPENAI) {
            listOf("model" to config.model.trim(), "response_format" to "json")
        } else {
            emptyList()
        }
        val multipart = MultipartRequest(boundary, audioFile, fields)
        val url = URL(inferenceUrl(endpoint, config.protocol))
        val connection = (url.openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = false
            requestMethod = "POST"
            connectTimeout = CONNECT_TIMEOUT_MILLIS
            readTimeout = READ_TIMEOUT_MILLIS
            doOutput = true
            useCaches = false
            setRequestProperty("Accept", "application/json")
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            setRequestProperty("X-Request-Id", UUID.randomUUID().toString())
            setFixedLengthStreamingMode(multipart.contentLength)
        }

        try {
            connection.outputStream.use { output ->
                multipart.writeTo(output)
            }

            val status = connection.responseCode
            if (status in 300..399) {
                return InferenceResult.Failure(
                    "Server redirect blocked (HTTP $status). Set the final endpoint explicitly.",
                    false,
                )
            }
            if (status !in 200..299) {
                val detail = runCatching {
                    connection.errorStream?.let { stream ->
                        BufferedInputStream(stream).use { readBounded(it, MAX_RESPONSE_BYTES) }
                    }?.let { bytes ->
                        val payload = JSONObject(bytes.toString(Charsets.UTF_8))
                        val error = payload.opt("error")
                        sequenceOf(
                            (error as? JSONObject)?.opt("message"),
                            payload.opt("detail"), error, payload.opt("message"),
                        ).filterIsInstance<String>().firstOrNull { it.isNotBlank() }
                    }
                }.getOrNull()
                return InferenceResult.Failure(
                    message = detail ?: "Starling backend returned HTTP $status",
                    retryable = status == 408 || status == 425 || status == 429 || status >= 500,
                )
            }
            val body = BufferedInputStream(connection.inputStream).use { readBounded(it, MAX_RESPONSE_BYTES) }
            val text = runCatching { JSONObject(body.toString(Charsets.UTF_8)).getString("text") }
                .getOrElse {
                    return InferenceResult.Failure("Starling returned an invalid transcript response", false)
                }
            // Preserve the backend's exact text. Do not trim or run a cleaner.
            return InferenceResult.Success(text)
        } finally {
            connection.disconnect()
        }
    }

    private fun readBounded(input: BufferedInputStream, maximumBytes: Int): ByteArray {
        val output = ByteArrayOutputStream()
        val buffer = ByteArray(BUFFER_SIZE)
        var total = 0
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            total += count
            if (total > maximumBytes) throw IOException("Backend response was too large")
            output.write(buffer, 0, count)
        }
        return output.toByteArray()
    }

    companion object {
        private const val BUFFER_SIZE = 16 * 1024
        private const val MAX_UPLOAD_BYTES = 256L * 1024 * 1024
        private const val MAX_RESPONSE_BYTES = 2 * 1024 * 1024
        private const val MAX_ATTEMPTS = 3
        private const val CONNECT_TIMEOUT_MILLIS = 15_000
        private val READ_TIMEOUT_MILLIS = TimeUnit.MINUTES.toMillis(10).toInt()
        private val RETRY_DELAYS_MILLIS = longArrayOf(500, 1_500)
    }
}
