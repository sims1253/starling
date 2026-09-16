package dev.starling.mobile.engine

import dev.starling.mobile.network.InferenceResult
import java.io.DataInputStream
import java.io.File
import java.io.InputStream

/**
 * Owns the on-device Parakeet engine: the imported GGUF, the native model
 * context, and its warmup. The model loads lazily on first use and then
 * stays resident (hundreds of MB for a 0.6B q4 model). All calls are
 * serialized; the engine additionally mutexes inside the C API.
 */
class OnDeviceEngine(modelDir: File) {
    sealed interface ImportResult {
        data class Imported(val sizeBytes: Long) : ImportResult
        data class Rejected(val reason: String) : ImportResult
    }

    private val modelFile = File(modelDir, MODEL_FILE_NAME)
    private val lock = Any()

    private var handle: Long = 0L
    private var loadError: String? = null

    fun hasModel(): Boolean = modelFile.isFile && modelFile.length() >= ModelFiles.MIN_MODEL_BYTES

    fun modelSizeBytes(): Long = if (modelFile.isFile) modelFile.length() else 0L

    fun lastLoadError(): String? = loadError

    /**
     * Imports [input] as the on-device model: validated, written to a
     * temporary file, then atomically promoted. Any previously loaded model
     * is unloaded so the next transcription reloads from the new file.
     * Blocking; call from a background thread. [input] is always closed.
     */
    fun importModel(input: InputStream): ImportResult {
        input.use { stream ->
            val temporary = File(modelFile.parentFile, "$MODEL_FILE_NAME.importing")
            runCatching {
                modelFile.parentFile?.mkdirs()
                temporary.outputStream().use { output ->
                    stream.copyTo(output, BUFFER_SIZE)
                }
            }.onFailure {
                temporary.delete()
                return ImportResult.Rejected("The model could not be copied to private storage.")
            }

            val size = temporary.length()
            val header = ByteArray(ModelFiles.GGUF_MAGIC_SIZE)
            runCatching {
                temporary.inputStream().use { headerStream ->
                    // A single read may legally return short.
                    DataInputStream(headerStream).readFully(header)
                }
            }.onFailure {
                temporary.delete()
                return ImportResult.Rejected("The model file could not be read.")
            }
            val rejection = ModelFiles.validate(size, header)
            if (rejection != null) {
                temporary.delete()
                return ImportResult.Rejected(rejection)
            }

            synchronized(lock) {
                unload()
                if (modelFile.exists() && !modelFile.delete()) {
                    temporary.delete()
                    return ImportResult.Rejected("The previous model could not be replaced.")
                }
                if (!temporary.renameTo(modelFile)) {
                    temporary.delete()
                    return ImportResult.Rejected("The model could not be moved into place.")
                }
            }
            return ImportResult.Imported(size)
        }
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
