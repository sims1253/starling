package dev.starling.mobile.engine

import dev.starling.mobile.network.InferenceResult
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.UUID

/**
 * Owns the on-device Parakeet engine: the imported GGUF, the native model
 * context, and its warmup. The model loads lazily on first use and then
 * stays resident (hundreds of MB for a 0.6B q4 model). All calls are
 * serialized; the engine additionally mutexes inside the C API.
 */
class OnDeviceEngine(modelDir: File) {
    /** Where in [importModel] a rejection happened; import failures report their stage. */
    enum class ImportStage { OPEN, COPY, VALIDATE, PROMOTE }

    sealed interface ImportResult {
        data class Imported(val sizeBytes: Long) : ImportResult
        data class Rejected(val reason: String, val stage: ImportStage = ImportStage.COPY) : ImportResult
    }

    private val modelFile = File(modelDir, MODEL_FILE_NAME)
    private val lock = Any()

    private var handle: Long = 0L
    private var loadError: String? = null

    fun hasModel(): Boolean = modelFile.isFile && modelFile.length() >= ModelFiles.MIN_MODEL_BYTES

    fun modelSizeBytes(): Long = if (modelFile.isFile) modelFile.length() else 0L

    fun lastLoadError(): String? = loadError

    /**
     * Imports [input] as the on-device model (B08, transactional):
     *
     * 1. the whole import runs under the engine lock, so concurrent imports
     *    queue instead of racing and a concurrent transcription can never
     *    observe a half-published model;
     * 2. the payload is copied into a unique staging file (a failed or
     *    interrupted import can never corrupt another import's staging) and
     *    fsynced, so a later promotion can never publish a half-written file;
     * 3. the staged file is validated (magic, size, bounded GGUF metadata
     *    parse, Parakeet model family) BEFORE the active model is touched;
     * 4. promotion is a single atomic rename over the target — the previous
     *    model is never deleted first, so any failure at any earlier stage
     *    (or the rename itself) leaves the last usable model in place.
     *
     * Staging files from interrupted imports are swept at the start of the
     * next import (the lock guarantees none is in use), bounding storage to
     * one staging file at a time. Blocking; call from a background thread.
     * [input] is always closed.
     */
    fun importModel(input: InputStream): ImportResult =
        importModel(input) { staged, target -> staged.renameTo(target) }

    /**
     * Testable core of [importModel]; [promote] is the promotion seam
     * (default: an atomic rename). Returning false from it simulates a
     * failed promotion (device full, permissions, I/O error) and must leave
     * the previous model usable.
     */
    internal fun importModel(input: InputStream, promote: (staged: File, target: File) -> Boolean): ImportResult =
        synchronized(lock) {
            input.use { stream ->
                sweepStaleStaging()
                val staged = File(modelFile.parentFile, "$MODEL_FILE_NAME.${UUID.randomUUID()}.importing")
                val copied = runCatching {
                    modelFile.parentFile?.mkdirs()
                    FileOutputStream(staged).use { output ->
                        stream.copyTo(output, BUFFER_SIZE)
                        // Durability before promotion: the rename below must
                        // never publish a file whose blocks are not yet on
                        // disk. A sync failure is a copy-stage failure.
                        output.fd.sync()
                    }
                }
                if (copied.isFailure) {
                    staged.delete()
                    return ImportResult.Rejected("The model could not be copied to private storage.", ImportStage.COPY)
                }
                val size = staged.length()

                val rejection = ModelFiles.validateStaged(staged)
                if (rejection != null) {
                    staged.delete()
                    return ImportResult.Rejected(rejection, ImportStage.VALIDATE)
                }

                // Single atomic step: rename(2) over the target replaces it
                // or fails — the previous model is never deleted first, so a
                // failed promotion keeps the last usable model recoverable.
                if (!promote(staged, modelFile)) {
                    staged.delete()
                    return ImportResult.Rejected(
                        "The model could not be moved into place; the previous model was kept.",
                        ImportStage.PROMOTE,
                    )
                }
                unload() // the next transcription reloads from the new file
                ImportResult.Imported(size)
            }
        }

    /**
     * Removes leftover staging files (the unique ones from interrupted
     * imports and the fixed-name ones written by earlier app versions).
     * Only ever called with [lock] held, so no staging file can be in use.
     */
    private fun sweepStaleStaging() {
        val directory = modelFile.parentFile ?: return
        val stale = directory.listFiles { file ->
            file.isFile && (file.name == LEGACY_STAGING_NAME ||
                (file.name.startsWith("$MODEL_FILE_NAME.") && file.name.endsWith(".importing")))
        } ?: return
        for (file in stale) file.delete()
    }

    /** Blocking transcription of a finalized WAV recording. */
    fun transcribe(audioFile: File): InferenceResult = synchronized(lock) {
        if (!hasModel()) {
            return InferenceResult.Failure(
                "Import a Parakeet model first to transcribe on this device.",
                false,
            )
        }
        if (handle == 0L) {
            val abi = StarlingNative.abiVersion()
            if (abi != StarlingNative.EXPECTED_ABI_VERSION) {
                loadError = "engine ABI $abi, expected ${StarlingNative.EXPECTED_ABI_VERSION}"
                return InferenceResult.Failure("The on-device engine is incompatible: $loadError", false)
            }
            val loaded = StarlingNative.load(modelFile.absolutePath)
            if (loaded == 0L) {
                val reason = StarlingNative.lastError(0L) ?: "the model could not be loaded"
                loadError = reason
                return InferenceResult.Failure("The on-device model failed to load: $reason", false)
            }
            handle = loaded
            loadError = null
            // Absorb lazy graph construction before the first real request,
            // mirroring starling-serve's warmup.
            StarlingNative.transcribe(handle, FloatArray(Warmup.SAMPLES), Warmup.SAMPLE_RATE)
        }

        val decoded = WavPcm.decodeMonoFloat(audioFile)
            ?: return InferenceResult.Failure("The recording audio could not be decoded.", false)
        if (decoded.sampleRate != WavWriterContract.SAMPLE_RATE) {
            return InferenceResult.Failure(
                "The recording sample rate (${decoded.sampleRate} Hz) is not supported.",
                false,
            )
        }
        // Bound each engine call like the serving layer (30 s step, 2 s
        // overlap): one full-attention pass over the whole clip would scale
        // quadratically with the recording length.
        val windows = ChunkedTranscription.planWindows(decoded.samples.size, decoded.sampleRate)
        val texts = ArrayList<String>(windows.size)
        for (window in windows) {
            val samples = if (window.start == 0 && window.endExclusive == decoded.samples.size) {
                decoded.samples
            } else {
                decoded.samples.copyOfRange(window.start, window.endExclusive)
            }
            val text = StarlingNative.transcribe(handle, samples, decoded.sampleRate)
                ?: return InferenceResult.Failure(
                    "The on-device engine returned an error: ${
                        StarlingNative.lastError(handle) ?: "unknown error"
                    }",
                    false,
                )
            texts.add(text)
        }
        // A single window is the direct path; joining would only normalize.
        val text = if (texts.size == 1) texts[0] else ChunkedTranscription.joinTexts(texts)
        InferenceResult.Success(text)
    }

    private fun unload() {
        if (handle != 0L) {
            StarlingNative.free(handle)
            handle = 0L
            loadError = null
        }
    }

    companion object {
        private const val MODEL_FILE_NAME = "parakeet.gguf"

        /** Fixed staging name written by app versions before B08; still swept. */
        private const val LEGACY_STAGING_NAME = "$MODEL_FILE_NAME.importing"
        private const val BUFFER_SIZE = 64 * 1024

        // Silence for warmup; matches serve's dummy-clip warmup purpose.
        private object Warmup {
            const val SAMPLE_RATE = 16_000
            const val SAMPLES = SAMPLE_RATE
        }

        // The capture pipeline's fixed rate (audio/WavWriter).
        private object WavWriterContract {
            const val SAMPLE_RATE = 16_000
        }
    }
}
