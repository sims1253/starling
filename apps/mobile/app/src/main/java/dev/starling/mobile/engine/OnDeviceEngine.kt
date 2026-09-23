package dev.starling.mobile.engine

import android.system.Os
import android.system.OsConstants
import android.util.Log
import dev.starling.mobile.network.InferenceResult
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID

/**
 * Owns the on-device Parakeet engine: the imported GGUF, the native model
 * context, and its warmup. The model loads lazily on first use and then
 * stays resident (hundreds of MB for a 0.6B q4 model) until
 * [releaseWhenIdle] drops it under memory pressure; the next use reloads it.
 * All calls are serialized; the engine additionally mutexes inside the C API.
 *
 * [memoryGate] is consulted before each load with the model's size and
 * returns a reason to refuse it (not enough free memory), so a load that
 * cannot fit fails with an explanation instead of the process being killed.
 * [nativeSupport] decides whether this CPU can run the native library at all
 * (see [NativeSupport]); it is checked before the library is first loaded.
 */
class OnDeviceEngine(
    modelDir: File,
    private val memoryGate: (modelBytes: Long) -> String? = { null },
    private val nativeSupport: () -> String? = NativeSupport::unsupportedReason,
    private val nativePolicy: NativeCallPolicy = NativeCallPolicy.None,
) : OnDeviceStreamSession.LiveEngine {
    /** One native transcription: how much audio, how long, on which device. */
    data class RunStats(val device: String, val audioSeconds: Double, val elapsedMillis: Long)

    /** Where in [importModel] a rejection happened; import failures report their stage. */
    enum class ImportStage { OPEN, COPY, VALIDATE, PROMOTE }

    sealed interface ImportResult {
        data class Imported(val sizeBytes: Long) : ImportResult
        data class Rejected(val reason: String, val stage: ImportStage) : ImportResult
    }

    private val modelFile = File(modelDir, MODEL_FILE_NAME)

    /** Guards engine state: the native handle and the model file's identity. */
    private val lock = Any()

    /**
     * Serializes imports. The staging sweep at the start of an import deletes
     * every staging file, so a second import must never be mid-copy while it
     * runs — hence a dedicated lock, distinct from [lock]: the multi-hundred-MB
     * copy and the staged-file validation do not touch engine state and must
     * not block a concurrent transcription.
     */
    private val importLock = Any()

    private var handle: Long = 0L
    private var loadError: String? = null

    /** The device the loaded engine runs on (from the native backend); null before the first load. */
    @Volatile
    var deviceName: String? = null
        private set

    /** The most recent transcription (a whole recording, or one live window). */
    @Volatile
    var lastRun: RunStats? = null
        private set

    fun hasModel(): Boolean = modelFile.isFile && modelFile.length() >= ModelFiles.MIN_MODEL_BYTES

    fun modelSizeBytes(): Long = if (modelFile.isFile) modelFile.length() else 0L

    fun lastLoadError(): String? = loadError

    /**
     * Imports [input] as the on-device model (B08, transactional):
     *
     * 1. imports are serialized on a dedicated lock, so the opening sweep
     *    can never delete another import's in-flight staging file; the
     *    engine lock is taken only around the publish step, so a concurrent
     *    transcription is blocked for the atomic promotion, not for the
     *    whole multi-hundred-MB copy, and can never observe a half-published
     *    model;
     * 2. the payload is copied into a unique staging file (a failed or
     *    interrupted import can never corrupt another import's staging) and
     *    fsynced, so a later promotion can never publish a half-written file;
     * 3. the staged file is validated (magic, size, bounded GGUF metadata +
     *    tensor-info parse including the data-section size, Parakeet model
     *    family) BEFORE the active model is touched;
     * 4. promotion is a single atomic rename over the target — the previous
     *    model is never deleted first, so any failure at any earlier stage
     *    (or the rename itself) leaves the last usable model in place.
     *
     * Staging files from interrupted imports are swept at the start of the
     * next import (the import lock guarantees none is in use), bounding
     * storage to one staging file at a time. Blocking; call from a
     * background thread. [input] is always closed.
     */
    fun importModel(input: InputStream): ImportResult =
        importModel(input, ::promoteByMove)

    /**
     * Testable core of [importModel]; [promote] is the promotion seam
     * (default: [promoteByMove]'s atomic rename). Returning false from it
     * simulates a failed promotion (device full, permissions, I/O error) and
     * must leave the previous model usable.
     */
    internal fun importModel(input: InputStream, promote: (staged: File, target: File) -> Boolean): ImportResult =
        synchronized(importLock) {
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
                    deleteFile(staged, "staging file")
                    return ImportResult.Rejected("The model could not be copied to private storage.", ImportStage.COPY)
                }
                val size = staged.length()

                try {
                    // Validation reads only the staging file, so it runs
                    // outside the engine lock; a concurrent transcription
                    // keeps serving the previous model meanwhile.
                    val rejection = ModelFiles.validateStaged(staged)
                    if (rejection != null) {
                        return ImportResult.Rejected(rejection, ImportStage.VALIDATE)
                    }

                    synchronized(lock) {
                        // Single atomic step: rename(2) over the target
                        // replaces it or fails — the previous model is never
                        // deleted first, so a failed promotion keeps the
                        // last usable model recoverable.
                        if (!promote(staged, modelFile)) {
                            return ImportResult.Rejected(
                                "The model could not be moved into place; the previous model was kept.",
                                ImportStage.PROMOTE,
                            )
                        }
                        fsyncModelDirectory()
                        // The new file is already in place; a native failure
                        // here must not mask a completed import (the next
                        // transcription reloads from the new file anyway).
                        runCatching { unload() }
                    }
                    ImportResult.Imported(size)
                } finally {
                    // Every path that reaches here without a rename-based
                    // promotion leaves the staged file behind: rejection,
                    // failed promotion, a copy-based test seam, or an
                    // unexpected throw. Delete it so at most one staging
                    // file ever exists.
                    deleteFile(staged, "staging file")
                }
            }
        }

    /**
     * Promotion: a single atomic rename over the target replaces it or
     * fails — the previous model is never deleted first. [File.renameTo] is
     * documented as platform-dependent (it can return false with no errno
     * when the target exists), so the preferred path is java.nio
     * [Files.move] with [StandardCopyOption.ATOMIC_MOVE], which pins the
     * POSIX replace semantics; filesystems that refuse atomic moves get
     * [moveAsideFirst], which still never deletes the previous model first.
     */
    internal fun promoteByMove(staged: File, target: File): Boolean =
        try {
            Files.move(staged.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE)
            true
        } catch (_: AtomicMoveNotSupportedException) {
            moveAsideFirst(staged, target)
        } catch (_: IOException) {
            false
        }

    /**
     * Fallback promotion for filesystems that cannot rename atomically over
     * an existing target: the previous model is renamed aside — never
     * deleted — the staged file moved in, and the previous model restored
     * (best effort) when the move-in fails. A crash between the two renames
     * leaves the previous model's bytes in the aside file rather than
     * nowhere.
     */
    internal fun moveAsideFirst(staged: File, target: File): Boolean {
        val aside = File(target.parentFile, target.name + ASIDE_SUFFIX)
        val hadPrevious = target.exists()
        if (hadPrevious && !target.renameTo(aside)) return false
        if (staged.renameTo(target)) {
            if (hadPrevious) deleteFile(aside, "replaced model")
            return true
        }
        if (hadPrevious) aside.renameTo(target)
        return false
    }

    /**
     * Makes the promotion rename itself durable. The staged file's bytes
     * were fsynced before the rename, but without a directory fsync a crash
     * can roll the rename back on filesystems that journal metadata lazily.
     * On Linux/bionic, opening a directory O_RDONLY yields a valid fsync-able
     * descriptor (O_DIRECTORY is not in android.system.OsConstants). Best
     * effort: a failure here cannot undo the completed rename.
     */
    private fun fsyncModelDirectory() {
        val directory = modelFile.parentFile?.takeIf { it.isDirectory } ?: return
        runCatching {
            val fd = Os.open(directory.absolutePath, OsConstants.O_RDONLY, 0)
            try {
                Os.fsync(fd)
            } finally {
                Os.close(fd)
            }
        }
    }

    /**
     * Removes leftover staging files: the unique ones from interrupted
     * imports and the fixed "parakeet.gguf.importing" name written by app
     * versions before B08 (both match the same prefix/suffix predicate).
     * Only ever called with [importLock] held, so no staging file can be in
     * use.
     */
    private fun sweepStaleStaging() {
        val directory = modelFile.parentFile ?: return
        val stale = directory.listFiles { file ->
            file.isFile && file.name.startsWith("$MODEL_FILE_NAME.") && file.name.endsWith(".importing")
        } ?: return
        for (file in stale) deleteFile(file, "stale staging file")
    }

    /** Deletes [file]; a real deletion failure (file still present) is logged, never thrown. */
    private fun deleteFile(file: File, what: String) {
        if (!file.delete() && file.exists()) {
            // runCatching: Log is a stub in local unit tests, and failed
            // diagnostics must never take down an import.
            runCatching { Log.w(TAG, "Could not delete $what: ${file.name}") }
        }
    }

    /**
     * Loads the model when it is not resident. Caller holds [lock]. Returns
     * null when the engine is ready, else the user-facing reason it is not.
     */
    private fun ensureLoadedLocked(): String? {
        if (!hasModel()) return "Import a Parakeet model first to transcribe on this device."
        if (handle != 0L) return null
        nativeSupport()?.let { reason ->
            loadError = reason
            return "The on-device engine cannot run here: $reason"
        }
        memoryGate(modelSizeBytes())?.let { reason ->
            loadError = reason
            return "The on-device model was not loaded: $reason"
        }
        NativeSupport.applyThreadDefault()
        nativePolicy.beforeFirstLoad()
        val abi = StarlingNative.abiVersion()
        if (abi != StarlingNative.EXPECTED_ABI_VERSION) {
            loadError = "engine ABI $abi, expected ${StarlingNative.EXPECTED_ABI_VERSION}"
            return "The on-device engine is incompatible: $loadError"
        }
        val loaded = nativePolicy.guard { StarlingNative.load(modelFile.absolutePath) }
        if (loaded == 0L) {
            val reason = StarlingNative.lastError(0L) ?: "the model could not be loaded"
            loadError = reason
            return "The on-device model failed to load: $reason"
        }
        handle = loaded
        loadError = null
        deviceName = StarlingNative.backendName()
        // Absorb lazy graph construction before the first real request,
        // mirroring starling-serve's warmup.
        nativePolicy.guard { StarlingNative.transcribe(handle, FloatArray(Warmup.SAMPLES), Warmup.SAMPLE_RATE) }
        return null
    }

    /** Loads the model ahead of a live session; null when ready. Blocking. */
    override fun prepare(): String? = synchronized(lock) { ensureLoadedLocked() }

    /** One live-stream window of 16 kHz mono samples. Blocking. */
    override fun transcribeWindow(samples: FloatArray): OnDeviceStreamSession.WindowResult = synchronized(lock) {
        ensureLoadedLocked()?.let { return OnDeviceStreamSession.WindowResult.Failed(it) }
        val started = System.nanoTime()
        val text = nativePolicy.guard { StarlingNative.transcribe(handle, samples, ChunkStreamer.SAMPLE_RATE) }
        recordRun(samples.size, started)
        if (text == null) {
            return OnDeviceStreamSession.WindowResult.Failed(
                "the on-device engine returned an error: ${StarlingNative.lastError(handle) ?: "unknown error"}",
            )
        }
        OnDeviceStreamSession.WindowResult.Text(text)
    }

    /**
     * Frees the resident model once no call is using it (waits for an
     * in-flight transcription). Blocking; call off the main thread. The
     * next transcription reloads the model from disk.
     *
     * A live [OnDeviceStreamSession] is not protected: its next window
     * reloads the model, which can take long enough for the session to fall
     * behind and hand the recording to the batch path. That is deliberate:
     * this is only called under memory pressure, where keeping hundreds of
     * MB resident risks the process being killed mid-recording.
     */
    fun releaseWhenIdle() = synchronized(lock) { unload() }

    /** Blocking transcription of a finalized WAV recording. */
    fun transcribe(audioFile: File): InferenceResult = synchronized(lock) {
        ensureLoadedLocked()?.let { return InferenceResult.Failure(it, false) }

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
        val started = System.nanoTime()
        for (window in windows) {
            val samples = if (window.start == 0 && window.endExclusive == decoded.samples.size) {
                decoded.samples
            } else {
                decoded.samples.copyOfRange(window.start, window.endExclusive)
            }
            val text = nativePolicy.guard { StarlingNative.transcribe(handle, samples, decoded.sampleRate) }
                ?: return InferenceResult.Failure(
                    "The on-device engine returned an error: ${
                        StarlingNative.lastError(handle) ?: "unknown error"
                    }",
                    false,
                )
            texts.add(text)
        }
        recordRun(decoded.samples.size, started)
        // A single window is the direct path; joining would only normalize.
        val text = if (texts.size == 1) texts[0] else ChunkedTranscription.joinTexts(texts)
        InferenceResult.Success(text)
    }

    private fun recordRun(samples: Int, startedNanos: Long) {
        lastRun = RunStats(
            device = deviceName ?: "unknown",
            audioSeconds = samples.toDouble() / ChunkStreamer.SAMPLE_RATE,
            elapsedMillis = (System.nanoTime() - startedNanos) / 1_000_000,
        )
    }

    private fun unload() {
        if (handle != 0L) {
            StarlingNative.free(handle)
            handle = 0L
            loadError = null
        }
    }

    companion object {
        private const val TAG = "OnDeviceEngine"
        private const val MODEL_FILE_NAME = "parakeet.gguf"

        /** Holds the previous model during a [moveAsideFirst] promotion. */
        private const val ASIDE_SUFFIX = ".previous"
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
