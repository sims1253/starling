package dev.starling.mobile.engine

import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import okhttp3.Call
import okhttp3.OkHttpClient
import okhttp3.Request

/**
 * Resumable, verified download of a [ModelDownload] into a partial file.
 *
 * The partial file survives cancellation, a lost connection and process
 * death: the next attempt asks for the remaining bytes with an HTTP Range
 * request (Hugging Face and its CDN honor them) and starts over only when the
 * server ignores the range. A finished file is checked against the expected
 * size and SHA-256; a mismatch deletes it, so a corrupt download can never
 * be offered for import. Blocking: call [download] from a worker thread,
 * [cancel] from any thread.
 */
class ModelDownloader(private val client: OkHttpClient = defaultClient()) {
    sealed interface Result {
        /** [file] holds exactly the expected bytes. */
        data class Done(val file: File) : Result

        /** User-facing reason; a partial file that can be resumed is kept. */
        data class Failed(val reason: String) : Result

        /** Stopped by [cancel]; the partial file is kept for a later resume. */
        data object Cancelled : Result
    }

    @Volatile private var cancelled = false
    @Volatile private var call: Call? = null

    fun cancel() {
        cancelled = true
        call?.cancel()
    }

    /**
     * Downloads [spec] into [partial], resuming whatever it already holds.
     * [progress] receives (bytes on disk, total bytes, verifying) roughly
     * every MiB while downloading, then once with verifying = true while the
     * checksum runs.
     */
    fun download(spec: ModelDownload, partial: File, progress: (Long, Long, Boolean) -> Unit): Result {
        cancelled = false
        partial.parentFile?.mkdirs()
        if (partial.length() > spec.sizeBytes) partial.delete()
        if (partial.length() < spec.sizeBytes) {
            val needed = spec.sizeBytes - partial.length() + FREE_SPACE_MARGIN_BYTES
            val free = partial.parentFile?.usableSpace ?: Long.MAX_VALUE
            if (free < needed) {
                return Result.Failed(
                    "Not enough free storage: the model needs ${needed / MB} MB, ${free / MB} MB are free.",
                )
            }
            val fetched = fetch(spec, partial) { bytes, total -> progress(bytes, total, false) }
            if (fetched != null) return fetched
        }
        if (partial.length() != spec.sizeBytes) {
            partial.delete()
            return Result.Failed("The server sent a file of the wrong size.")
        }
        progress(spec.sizeBytes, spec.sizeBytes, true)
        val digest = runCatching { sha256(partial) }.getOrElse {
            return Result.Failed("The download could not be read back: ${it.message}")
        }
        if (!digest.equals(spec.sha256, ignoreCase = true)) {
            partial.delete()
            return Result.Failed("The download was corrupted (checksum mismatch); it was discarded.")
        }
        return Result.Done(partial)
    }

    /** Appends the missing bytes to [partial]; null when the transfer completed. */
    private fun fetch(spec: ModelDownload, partial: File, progress: (Long, Long) -> Unit): Result? {
        var offset = partial.length()
        val request = Request.Builder()
            .url(spec.url)
            .apply { if (offset > 0) header("Range", "bytes=$offset-") }
            .build()
        val current = client.newCall(request)
        call = current
        if (cancelled) return Result.Cancelled
        try {
            current.execute().use { response ->
                when {
                    response.code == 206 && rangeStart(response.header("Content-Range")) == offset -> Unit
                    response.code == 206 -> {
                        // A resume from another position cannot be appended.
                        partial.delete()
                        return Result.Failed("The server resumed at the wrong position; tap Download to start over.")
                    }
                    response.code == 200 -> offset = 0   // range ignored: start over
                    response.isSuccessful -> return Result.Failed("The server answered HTTP ${response.code}.")
                    response.code == 416 -> {
                        // Our partial is not a prefix the server recognizes.
                        partial.delete()
                        return Result.Failed("The download could not be resumed; tap Download to start over.")
                    }
                    else -> return Result.Failed("The server answered HTTP ${response.code}.")
                }
                FileOutputStream(partial, offset > 0).use { out ->
                    val input = response.body.byteStream()
                    val buffer = ByteArray(BUFFER_BYTES)
                    var written = offset
                    var reported = written
                    progress(written, spec.sizeBytes)
                    while (true) {
                        if (cancelled) return Result.Cancelled
                        val n = input.read(buffer)
                        if (n < 0) break
                        if (written + n > spec.sizeBytes) {
                            out.close()
                            partial.delete()
                            return Result.Failed("The server sent more data than expected; the download was discarded.")
                        }
                        out.write(buffer, 0, n)
                        written += n
                        if (written - reported >= PROGRESS_STEP_BYTES) {
                            progress(written, spec.sizeBytes)
                            reported = written
                        }
                    }
                    out.fd.sync()
                }
            }
        } catch (e: IOException) {
            if (cancelled) return Result.Cancelled
            return Result.Failed(
                "The download was interrupted (${e.message ?: e::class.java.simpleName}); tap Download to resume.",
            )
        } finally {
            call = null
        }
        if (partial.length() < spec.sizeBytes) {
            return Result.Failed("The connection closed early; tap Download to resume.")
        }
        return null
    }

    companion object {
        private const val BUFFER_BYTES = 256 * 1024
        private const val PROGRESS_STEP_BYTES = 1L shl 20
        private const val MB = 1_000_000L

        /** Headroom left on the volume so recordings and app data keep working. */
        private const val FREE_SPACE_MARGIN_BYTES = 200L * MB

        /** Hugging Face redirects to its CDN; only the connect and per-read waits are bounded. */
        fun defaultClient(): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .callTimeout(0, TimeUnit.SECONDS)
            .followRedirects(true)
            .followSslRedirects(true)
            .build()

        /** First byte of a `Content-Range: bytes a-b/n` header, or -1. */
        internal fun rangeStart(header: String?): Long =
            header?.removePrefix("bytes ")?.substringBefore('-')?.trim()?.toLongOrNull() ?: -1L

        internal fun sha256(file: File): String {
            val md = MessageDigest.getInstance("SHA-256")
            FileInputStream(file).use { input ->
                val buffer = ByteArray(BUFFER_BYTES)
                while (true) {
                    val n = input.read(buffer)
                    if (n < 0) break
                    md.update(buffer, 0, n)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
