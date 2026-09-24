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
 * Owns the on-device Parakeet engine: the installed GGUFs, which one is
 * active, the native model context, and its warmup. Every `*.gguf` in
 * [modelDir] is an installed model (imports keep their file name, downloads
 * their catalog name); the active one is named in [ACTIVE_FILE_NAME], and
 * falls back to the first installed model when that file is missing or
 * names a deleted model. The model loads lazily on first use and then
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
    private val modelDir: File,
    private val memoryGate: (modelBytes: Long) -> String? = { null },
    private val nativeSupport: () -> String? = NativeSupport::unsupportedReason,
) : OnDeviceStreamSession.LiveEngine {
    /** Where in [importModel] a rejection happened; import failures report their stage. */
    enum class ImportStage { OPEN, COPY, VALIDATE, PROMOTE }

    sealed interface ImportResult {
        data class Imported(val sizeBytes: Long, val name: String) : ImportResult
        data class Rejected(val reason: String, val stage: ImportStage) : ImportResult
    }

    data class InstalledModel(val name: String, val sizeBytes: Long, val active: Boolean)

    private val activeFile = File(modelDir, ACTIVE_FILE_NAME)

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

    /** The model file [handle] was loaded from; a different active model forces a reload. */
    private var loadedFile: File? = null
    private var loadError: String? = null

    // Guarded by [lock]: live sessions in progress, and a memory-pressure
    // release that waits for them to end.
    private var liveSessions = 0
    private var releasePending = false

    fun hasModel(): Boolean = activeModelFile() != null

    fun modelSizeBytes(): Long = activeModelFile()?.length() ?: 0L

    /** File name of the model transcription uses, or null when none is installed. */
    fun activeModelName(): String? = activeModelFile()?.name

    /** Every installed model, sorted by name, with the active one marked. */
    fun installedModels(): List<InstalledModel> {
        val active = activeModelFile()
        return installedFiles().map { InstalledModel(it.name, it.length(), it == active) }
    }

    /** Makes the installed model [name] the active one; false when it is not installed. */
    fun selectModel(name: String): Boolean = synchronized(lock) {
        val file = installedFile(name) ?: return false
        writeActiveName(file.name)
        // Free the previous model now; the next transcription loads this one.
        if (loadedFile != file) unload()
        true
    }

    /**
     * Deletes the installed model [name]. Deleting the active model makes the
     * first remaining one active. Waits for an in-flight transcription.
     */
    fun deleteModel(name: String): Boolean = synchronized(importLock) {
        synchronized(lock) {
            val file = installedFile(name) ?: return false
            if (file == loadedFile) unload()
            if (!file.delete()) return false
            if (readActiveName() == file.name) activeFile.delete()
            true
        }
    }

    /**
     * Renames the installed model [from] to [to] (a sanitized file name),
     * keeping it active if it was. Refuses to overwrite another model.
     */
    fun renameModel(from: String, to: String): Boolean = synchronized(importLock) {
        synchronized(lock) {
            val file = installedFile(from) ?: return false
            val target = File(modelDir, sanitizeModelName(to))
            if (target.exists()) return false
            val wasActive = activeModelFile() == file
            if (file == loadedFile) unload()
            if (!file.renameTo(target)) return false
            if (wasActive) writeActiveName(target.name)
            fsyncModelDirectory()
            true
        }
    }

    private fun installedFiles(): List<File> =
        modelDir.listFiles { file ->
            file.isFile && isModelName(file.name) && file.length() >= ModelFiles.MIN_MODEL_BYTES
        }.orEmpty().sortedBy { it.name }

    /** The installed model file [name], for read-only checks such as hashing. */
    fun modelFile(name: String): File? = installedFile(name)

    private fun installedFile(name: String): File? =
        installedFiles().firstOrNull { it.name == name }

    private fun activeModelFile(): File? {
        val installed = installedFiles()
        val named = readActiveName()
        return installed.firstOrNull { it.name == named } ?: installed.firstOrNull()
    }

    private fun readActiveName(): String? =
        runCatching { activeFile.readText(Charsets.UTF_8).trim() }.getOrNull()?.takeIf(String::isNotEmpty)

    /** Atomically records [name] as the active model. */
    private fun writeActiveName(name: String) {
        modelDir.mkdirs()
        val temporary = File(modelDir, "$ACTIVE_FILE_NAME.tmp")
        FileOutputStream(temporary).use { output ->
            output.write(name.toByteArray(Charsets.UTF_8))
            output.fd.sync()
        }
        if (!temporary.renameTo(activeFile)) {
            deleteFile(temporary, "active model marker")
            throw IOException("Unable to record the active model")
        }
        fsyncModelDirectory()
    }

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
    fun importModel(input: InputStream, name: String = DEFAULT_MODEL_NAME): ImportResult =
        importModel(input, name, ::promoteByMove)

    /**
     * Testable core of [importModel]; [promote] is the promotion seam
     * (default: [promoteByMove]'s atomic rename). Returning false from it
     * simulates a failed promotion (device full, permissions, I/O error) and
     * must leave the previous model usable.
     */
    internal fun importModel(
        input: InputStream,
        name: String = DEFAULT_MODEL_NAME,
        promote: (staged: File, target: File) -> Boolean,
    ): ImportResult =
        synchronized(importLock) {
            input.use { stream ->
                sweepStaleStaging()
                val staged = stagingFile()
                val copied = runCatching {
                    modelDir.mkdirs()
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
                    val detail = copied.exceptionOrNull()?.localizedMessage?.takeIf(String::isNotBlank)
                    return ImportResult.Rejected(
                        "The model could not be copied to private storage" +
                            (detail?.let { ": $it" } ?: "."),
                        ImportStage.COPY,
                    )
                }
                publishStaged(staged, File(modelDir, sanitizeModelName(name)), promote)
            }
        }

    /**
     * Imports a finished, checksum-verified download ([ModelDownloader]) as
     * the on-device model. The file is renamed into a staging name in the
     * model directory — no second multi-hundred-MB copy — and then goes
     * through the same validation and atomic promotion as [importModel].
     * [downloaded] must live in the model directory (a file elsewhere is
     * refused and left alone); once staged it is consumed either way.
     */
    fun adoptDownloaded(downloaded: File, name: String = DEFAULT_MODEL_NAME): ImportResult =
        adoptDownloaded(downloaded, name, ::promoteByMove)

    internal fun adoptDownloaded(
        downloaded: File,
        name: String = DEFAULT_MODEL_NAME,
        promote: (staged: File, target: File) -> Boolean,
    ): ImportResult =
        synchronized(importLock) {
            sweepStaleStaging()
            val staged = stagingFile()
            // A file outside the model directory is not ours: refuse, keep it.
            if (downloaded.parentFile?.canonicalFile != modelDir.canonicalFile) {
                return ImportResult.Rejected("The downloaded model is not in the model directory.", ImportStage.COPY)
            }
            if (!downloaded.renameTo(staged)) {
                deleteFile(downloaded, "downloaded model")
                return ImportResult.Rejected("The downloaded model could not be staged.", ImportStage.COPY)
            }
            publishStaged(staged, File(modelDir, sanitizeModelName(name)), promote)
        }

    private fun stagingFile() = File(modelDir, "$DEFAULT_MODEL_NAME.${UUID.randomUUID()}.importing")

    /**
     * Validates [staged], promotes it over [modelFile] (replacing an installed
     * model of the same name), and makes it the active model; [staged] is gone
     * afterwards. Holds [importLock].
     */
    private fun publishStaged(
        staged: File,
        modelFile: File,
        promote: (staged: File, target: File) -> Boolean,
    ): ImportResult {
        val size = staged.length()
        try {
            // Validation reads only the staging file, so it runs outside the
            // engine lock; a concurrent transcription keeps serving the
            // previous model meanwhile.
            val rejection = ModelFiles.validateStaged(staged)
            if (rejection != null) {
                return ImportResult.Rejected(rejection, ImportStage.VALIDATE)
            }

            synchronized(lock) {
                // Single atomic step: rename(2) over the target replaces it
                // or fails — the previous model is never deleted first, so a
                // failed promotion keeps the last usable model recoverable.
                if (!promote(staged, modelFile)) {
                    return ImportResult.Rejected(
                        "The model could not be moved into place; the previous model was kept.",
                        ImportStage.PROMOTE,
                    )
                }
                fsyncModelDirectory()
                // The new file is already in place; a native failure here
                // must not mask a completed import (the next transcription
                // reloads from the new file anyway).
                runCatching { unload() }
                // A fresh import is what the user wants to use next. The
                // model is installed even if this marker write fails.
                runCatching { writeActiveName(modelFile.name) }
            }
            return ImportResult.Imported(size, modelFile.name)
        } finally {
            // Every path that reaches here without a rename-based promotion
            // leaves the staged file behind: rejection, failed promotion, a
            // copy-based test seam, or an unexpected throw. Delete it so at
            // most one staging file ever exists.
            deleteFile(staged, "staging file")
        }
    }

    /** Where an in-progress download of [spec] lives (resumable across restarts). */
    fun downloadFile(spec: ModelDownload): File =
        File(modelDir, "download-${spec.sha256.take(16)}.part")

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
        val directory = modelDir.takeIf { it.isDirectory } ?: return
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
     * versions before B08 (both end in ".importing").
     * Only ever called with [importLock] held, so no staging file can be in
     * use.
     */
    private fun sweepStaleStaging() {
        val stale = modelDir.listFiles { file ->
            file.isFile && file.name.endsWith(".importing")
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
        val modelFile = activeModelFile()
            ?: return "Download or import a Parakeet model first to transcribe on this device."
        // The user picked another model since this one was loaded.
        if (handle != 0L && loadedFile != modelFile) unload()
        if (handle != 0L) return null
        nativeSupport()?.let { reason ->
            loadError = reason
            return "The on-device engine cannot run here: $reason"
        }
        memoryGate(modelFile.length())?.let { reason ->
            loadError = reason
            return "The on-device model was not loaded: $reason"
        }
        NativeSupport.applyThreadDefault()
        NativeSupport.applyFastEngineDefaults(modelFile.parentFile)
        val abi = StarlingNative.abiVersion()
        if (abi != StarlingNative.EXPECTED_ABI_VERSION) {
            loadError = "engine ABI $abi, expected ${StarlingNative.EXPECTED_ABI_VERSION}"
            return "The on-device engine is incompatible: $loadError"
        }
        val loaded = StarlingNative.load(modelFile.absolutePath)
        if (loaded == 0L) {
            val reason = StarlingNative.lastError(0L) ?: "the model could not be loaded"
            loadError = reason
            return "The on-device model failed to load: $reason"
        }
        handle = loaded
        loadedFile = modelFile
        loadError = null
        // Absorb lazy graph construction before the first real request,
        // mirroring starling-serve's warmup.
        StarlingNative.transcribe(handle, FloatArray(Warmup.SAMPLES), Warmup.SAMPLE_RATE)
        return null
    }

    /** Loads the model ahead of a live session; null when ready. Blocking. */
    override fun prepare(): String? = synchronized(lock) { ensureLoadedLocked() }

    /** One live-stream window of 16 kHz mono samples. Blocking. */
    override fun transcribeWindow(samples: FloatArray): OnDeviceStreamSession.WindowResult = synchronized(lock) {
        ensureLoadedLocked()?.let { return OnDeviceStreamSession.WindowResult.Failed(it) }
        val text = StarlingNative.transcribe(handle, samples, ChunkStreamer.SAMPLE_RATE)
            ?: return OnDeviceStreamSession.WindowResult.Failed(
                "the on-device engine returned an error: ${StarlingNative.lastError(handle) ?: "unknown error"}",
            )
        OnDeviceStreamSession.WindowResult.Text(text)
    }

    /**
     * Frees the resident model once no call is using it (waits for an
     * in-flight transcription). Blocking; call off the main thread. The
     * next transcription reloads the model from disk.
     *
     * While a live [OnDeviceStreamSession] runs, the release is deferred
     * until the last session ends: unloading between its windows would force
     * a multi-hundred-MB reload mid-recording, likely pushing the stream
     * past its live-buffer cap. A recording is minutes at most, and the
     * release follows it immediately.
     */
    fun releaseWhenIdle() = synchronized(lock) {
        if (liveSessions > 0) releasePending = true else unload()
    }

    override fun liveSessionStarted() {
        synchronized(lock) { liveSessions++ }
    }

    override fun liveSessionEnded() {
        synchronized(lock) {
            liveSessions = maxOf(0, liveSessions - 1)
            if (liveSessions == 0 && releasePending) {
                releasePending = false
                unload()
            }
        }
    }

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
            loadedFile = null
            loadError = null
        }
    }

    companion object {
        private const val TAG = "OnDeviceEngine"

        /** Name for a model imported without a usable file name, and of the single model before 0.2.2. */
        const val DEFAULT_MODEL_NAME = "parakeet.gguf"
        private const val ACTIVE_FILE_NAME = "active-model"
        private const val MODEL_EXTENSION = ".gguf"
        private const val MAX_NAME_CHARS = 120

        /** Holds the previous model during a [moveAsideFirst] promotion. */
        private const val ASIDE_SUFFIX = ".previous"
        private const val BUFFER_SIZE = 64 * 1024

        private fun isModelName(name: String) =
            !name.startsWith(".") && name.endsWith(MODEL_EXTENSION, ignoreCase = true)

        /**
         * A safe file name for a model from a user-visible [name] (an
         * imported document's display name): no path, only portable
         * characters, and a .gguf extension so it is listed as a model.
         */
        fun sanitizeModelName(name: String?): String {
            val base = name.orEmpty().substringAfterLast('/')
                .replace(Regex("[^A-Za-z0-9._-]"), "_")
                .trimStart('.')
                .take(MAX_NAME_CHARS)
            if (base.isEmpty() || base.equals(MODEL_EXTENSION, ignoreCase = true)) return DEFAULT_MODEL_NAME
            return if (isModelName(base)) base else base + MODEL_EXTENSION
        }

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
