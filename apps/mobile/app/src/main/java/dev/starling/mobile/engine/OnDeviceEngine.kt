package dev.starling.mobile.engine

import android.system.Os
import android.system.OsConstants
import android.util.Log
import dev.starling.mobile.transcription.InferenceResult
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.channels.FileChannel
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.util.UUID
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Owns the on-device speech engine: the installed GGUFs, which one is
 * active, the native model context, and its warmup. Each GGUF loads as the
 * [ModelKind] its metadata names (Parakeet or one of the LLM-decoder speech
 * models), detected at import and again at every load. Every `*.gguf` in
 * the directory [modelDirectory] returns is an installed model (imports
 * keep their file name, downloads their catalog name). The directory is
 * looked up per operation, so it can move (to shared storage once the app
 * may use it, see ModelStorage); models still in the install's own
 * [ownDir] stay installed until [adoptModels] moves them. Other Starling
 * builds may share the directory: choosing a name and renaming into it, and
 * deleting, happen under a lock file there ([withFolderLock]), download
 * partials carry this install's [partialTag], and only stale staging files
 * are swept. The active one is named in [ACTIVE_FILE_NAME] in [ownDir], and
 * falls back to the first installed model when that file is missing or
 * names a deleted model. The model loads on first use, or ahead of it
 * through [prepare] (the [ModelLifetime] preload), and then stays resident
 * (hundreds of MB for a 0.6B q4 model, GBs for the larger kinds) until
 * [releaseWhenIdle] drops it under memory pressure or [releaseIfIdle] drops
 * it after the idle policy's timeout; the next use reloads it. All calls are serialized; the engine
 * additionally mutexes inside the C API. [observer] learns about loads,
 * unloads, failures, and the end of each use, so the lifetime policy and the
 * UI never have to take the engine lock to know where the model stands.
 *
 * [memoryGate] is consulted before each load with the model's size and kind
 * (the kind sets the working set on top of the weights, see
 * [ModelKind.workingSetBytes]) and returns a reason to refuse it (not
 * enough free memory), so a load that cannot fit fails with an explanation
 * instead of the process being killed.
 * [nativeSupport] decides whether this CPU can run the native library at all
 * (see [NativeSupport]); it is checked before the library is first loaded.
 * [keepAwake] holds the CPU awake (a partial wake lock in the app) around
 * each native load, transcription call and free: kernel suspend with GPU
 * work outstanding wedges the Pixel's GPU driver.
 */
class OnDeviceEngine(
    private val modelDirectory: () -> File,
    private val ownDir: File,
    private val memoryGate: (modelBytes: Long, kind: ModelKind) -> String? = { _, _ -> null },
    private val nativeSupport: () -> String? = NativeSupport::unsupportedReason,
    private val keepAwake: () -> AutoCloseable = { AutoCloseable {} },
    private val partialTag: String = "local",
) : OnDeviceStreamSession.LiveEngine {
    /** An engine whose models and active-model marker all live in [modelDir]. */
    constructor(
        modelDir: File,
        memoryGate: (modelBytes: Long, kind: ModelKind) -> String? = { _, _ -> null },
        nativeSupport: () -> String? = NativeSupport::unsupportedReason,
        keepAwake: () -> AutoCloseable = { AutoCloseable {} },
    ) : this({ modelDir }, modelDir, memoryGate, nativeSupport, keepAwake)

    /**
     * What [adoptModels] did: models moved (an identical copy already in
     * place counts, its private duplicate removed), and models it had to
     * leave where they were.
     */
    data class Adoption(val moved: Int, val remaining: Int)

    /** Where in [importModel] a rejection happened; import failures report their stage. */
    enum class ImportStage { OPEN, COPY, VALIDATE, PROMOTE }

    sealed interface ImportResult {
        /** [model] names what was imported ([ModelKind.Detection.Supported.displayName]). */
        data class Imported(val sizeBytes: Long, val name: String, val model: String) : ImportResult
        data class Rejected(val reason: String, val stage: ImportStage) : ImportResult
    }

    data class InstalledModel(val name: String, val sizeBytes: Long, val active: Boolean)

    /**
     * Residency and use events, delivered on the thread that caused them and
     * with the engine lock held: implementations must return at once and must
     * never call back into the engine.
     */
    interface Observer {
        /** A load of [model] started (it may take seconds). */
        fun loading(model: String) = Unit

        /** [model] is resident and warmed up. */
        fun loaded(model: String) = Unit

        /** A load failed; [reason] is the user-facing explanation. */
        fun loadFailed(model: String?, reason: String) = Unit

        /** The resident model was freed. */
        fun unloaded() = Unit

        /** The resident model failed a request (not a load); [error] is the native reason. */
        fun engineFailed(error: String) = Unit

        /**
         * A use ended with the model resident and no live session running;
         * [generation] identifies this point for [releaseIfIdle].
         */
        fun idle(generation: Long) = Unit
    }

    @Volatile
    var observer: Observer = object : Observer {}

    private val activeFile = File(ownDir, ACTIVE_FILE_NAME)

    /** Guards engine state: the native handle and the model file's identity. */
    // A java.lang.Object (not Any) so prepare() can wait on it for a pinned
    // model to be released; Kotlin's Any has no wait/notify.
    @Suppress("PLATFORM_CLASS_MAPPED_TO_KOTLIN")
    private val lock = Object()

    /**
     * Serializes imports. The staging sweep at the start of an import deletes
     * every staging file, so a second import must never be mid-copy while it
     * runs — hence a dedicated lock, distinct from [lock]: the multi-hundred-MB
     * copy and the staged-file validation do not touch engine state and must
     * not block a concurrent transcription.
     */
    private val importLock = Any()

    /** Serializes active-model marker writes (they share one temporary file). */
    private val markerLock = Any()

    private var handle: Long = 0L

    // Counts model loads (written under [lock]): a live session reuses a
    // preview only while the load that produced it is still resident.
    // [residentGeneration] is the resident load's number, -1 when none;
    // volatile so a session at Stop never waits for another take's call.
    private var loadGeneration = 0L
    @Volatile
    private var residentGeneration = -1L

    /** The model file [handle] was loaded from; a different active model forces a reload. */
    private var loadedFile: File? = null

    /** The kind [loadedFile] loaded as: sets the transcription window length. */
    private var loadedKind: ModelKind? = null
    private var loadError: String? = null

    // Live sessions in progress. Counted without the engine lock: a session
    // starts on the main thread at the Record tap, which must never wait for
    // a preload that holds the lock through a multi-second load.
    private val liveSessions = AtomicInteger()

    // Guarded by [lock]: live sessions whose prepare() succeeded and that
    // therefore use the loaded model until they end, and a release (memory
    // pressure, model switch) that waits for them. A session that started
    // but has not loaded yet pins nothing: a release or a model switch then
    // goes ahead, and its prepare() loads the model it needs.
    private var pinnedSessions = 0
    private var releasePending = false

    // Bumped at the start and the end of every use, so an idle release armed
    // at one point never frees a model used since.
    private val useGeneration = AtomicLong()

    // When the native call in progress started (System.nanoTime), or
    // [NO_NATIVE_CALL]. Native calls run one at a time under [lock]; read
    // without it, so a transcription waiting for the lock can tell a busy
    // engine from a stuck one (#356).
    @Volatile
    private var nativeCallSince = NO_NATIVE_CALL

    fun hasModel(): Boolean = activeModelFile() != null

    fun modelSizeBytes(): Long = activeModelFile()?.length() ?: 0L

    /** File name of the model transcription uses, or null when none is installed. */
    fun activeModelName(): String? = activeModelFile()?.name

    /** Every installed model, sorted by name, with the active one marked. */
    fun installedModels(): List<InstalledModel> {
        val installed = installedFiles()
        val active = activeAmong(installed)
        return installed.map { InstalledModel(it.name, it.length(), it == active) }
    }

    /**
     * Makes the installed model [name] the active one; false when it is not
     * installed. The previous model is freed now, or when the live session
     * using it ends (see [releaseWhenIdle]).
     */
    fun selectModel(name: String): Boolean {
        val file = installedFile(name) ?: return false
        // The marker write (two fsyncs) stays outside the engine lock, so it
        // never stalls a transcription. Checked under the marker lock, which
        // deleteModel holds across its unlink and marker cleanup: either the
        // delete ran first (the file is gone) or it clears this marker.
        synchronized(markerLock) {
            if (!file.isFile) return false
            writeActiveName(file.name)
        }
        synchronized(lock) { if (loadedFile != file) releaseLocked() }
        return true
    }

    /**
     * Deletes the installed model [name]. Deleting the active model makes the
     * first remaining one active. Waits for an in-flight transcription; a
     * live session keeps its already-loaded model until it ends.
     */
    fun deleteModel(name: String): Boolean = synchronized(importLock) {
        val dir = installedFile(name)?.parentFile ?: return false
        // The unlink under the engine lock, so a load never races it, and
        // under the marker lock, so a concurrent selectModel either sees the
        // file gone or has written its marker before the cleanup below.
        // (Order: importLock, folder lock, lock, markerLock; no path takes
        // them the other way.)
        withFolderLock(dir) {
            synchronized(lock) {
                synchronized(markerLock) {
                    val file = installedFile(name) ?: return false
                    if (!file.delete()) return false
                    if (file == loadedFile) releaseLocked()
                    // Drops a marker naming this model — a not-yet-moved namesake
                    // in [ownDir] must not silently take its place — or one
                    // deleted earlier, so a later install under that name
                    // cannot silently become active.
                    val named = readActiveName()
                    if (named == name || (named != null && installedFiles().none { it.name == named })) {
                        activeFile.delete()
                    }
                }
            }
        }
        fsyncDirectory(dir)
        true
    }

    /** Frees the resident model now, or when the last live session ends. Caller holds [lock]. */
    private fun releaseLocked() {
        if (pinnedSessions > 0) releasePending = true else unload()
    }

    /**
     * The models in the current directory, plus any still in [ownDir] that
     * [adoptModels] could not move (a same-named one in the current
     * directory wins).
     */
    private fun installedFiles(): List<File> {
        val current = modelDirectory()
        val found = modelFilesIn(current).toMutableList()
        if (current.absoluteFile != ownDir.absoluteFile) {
            modelFilesIn(ownDir).filterTo(found) { leftover -> found.none { it.name == leftover.name } }
        }
        return found.sortedBy { it.name }
    }

    private fun modelFilesIn(dir: File): List<File> =
        dir.listFiles { file ->
            file.isFile && isModelName(file.name) && file.length() >= ModelFiles.MIN_MODEL_BYTES
        }.orEmpty().toList()

    private fun installedFile(name: String): File? =
        installedFiles().firstOrNull { it.name == name }

    private fun activeModelFile(): File? = activeAmong(installedFiles())

    private fun activeAmong(installed: List<File>): File? {
        val named = readActiveName()
        return installed.firstOrNull { it.name == named } ?: installed.firstOrNull()
    }

    /**
     * The sanitized [name] for an import; catalog file names are reserved
     * for verified downloads, so an import can never pose as (or be replaced
     * by) a catalog model.
     */
    private fun importName(name: String): String {
        val sanitized = sanitizeModelName(name)
        return if (ModelCatalog.ALL.any { it.fileName == sanitized }) "imported-$sanitized" else sanitized
    }

    /** [name] in [dir], or name-2.gguf, name-3.gguf, ... : the first that is not taken. */
    private fun freeModelFile(dir: File, name: String): File {
        val stem = name.removeSuffix(MODEL_EXTENSION)
        return generateSequence(1) { it + 1 }
            .map { n -> File(dir, if (n == 1) name else "$stem-$n$MODEL_EXTENSION") }
            .first { !it.exists() }
    }

    private fun readActiveName(): String? =
        runCatching { activeFile.readText(Charsets.UTF_8).trim() }.getOrNull()?.takeIf(String::isNotEmpty)

    // Bumped (under markerLock) by every marker write, so adoptModels can
    // tell whether the user picked a model while it was copying.
    private var markerGeneration = 0L

    /** Atomically records [name] as the active model. */
    private fun writeActiveName(name: String) = synchronized(markerLock) {
        markerGeneration++
        ownDir.mkdirs()
        val temporary = File(ownDir, "$ACTIVE_FILE_NAME.tmp")
        FileOutputStream(temporary).use { output ->
            output.write(name.toByteArray(Charsets.UTF_8))
            output.fd.sync()
        }
        if (!temporary.renameTo(activeFile)) {
            deleteFile(temporary, "active model marker")
            throw IOException("Unable to record the active model")
        }
        fsyncDirectory(ownDir)
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
     *    tensor-info parse including the data-section size, a supported
     *    [ModelKind]) BEFORE the active model is touched;
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
                val dir = modelDirectory()
                sweepStaleStaging(dir)
                val staged = stagingFile(dir)
                val copied = runCatching {
                    dir.mkdirs()
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
                        "The model could not be copied to the model folder" +
                            (detail?.let { ": $it" } ?: "."),
                        ImportStage.COPY,
                    )
                }
                // An import never replaces an installed model: a clash gets a
                // numbered name, so a same-named file cannot silently swap
                // out a model (or pose as the catalog model).
                publishStaged(staged, promote) { freeModelFile(dir, importName(name)) }
            }
        }

    /**
     * Imports a finished, checksum-verified download ([ModelDownloader]) as
     * the on-device model. The file is renamed into a staging name in its
     * own directory — no second multi-hundred-MB copy — and then goes
     * through the same validation and atomic promotion as [importModel],
     * there. [downloaded] must live in the model directory or [ownDir] (a
     * download that started before the models moved finishes in [ownDir];
     * [adoptModels] moves it later); a file elsewhere is refused and left
     * alone. Once staged it is consumed either way.
     */
    fun adoptDownloaded(downloaded: File, name: String = DEFAULT_MODEL_NAME): ImportResult =
        adoptDownloaded(downloaded, name, ::promoteByMove)

    internal fun adoptDownloaded(
        downloaded: File,
        name: String = DEFAULT_MODEL_NAME,
        promote: (staged: File, target: File) -> Boolean,
    ): ImportResult =
        synchronized(importLock) {
            // A file outside the model directories is not ours: refuse, keep it.
            val dir = downloaded.parentFile?.canonicalFile
                ?.takeIf { it == modelDirectory().canonicalFile || it == ownDir.canonicalFile }
                ?: return ImportResult.Rejected("The downloaded model is not in the model directory.", ImportStage.COPY)
            sweepStaleStaging(dir, keep = downloaded)
            val staged = stagingFile(dir)
            if (!downloaded.renameTo(staged)) {
                deleteFile(downloaded, "downloaded model")
                return ImportResult.Rejected("The downloaded model could not be staged.", ImportStage.COPY)
            }
            // A verified download replaces an earlier copy of itself.
            publishStaged(staged, promote) { File(dir, sanitizeModelName(name)) }
        }

    /**
     * Moves every model in [from] into the current model directory (once the
     * app may use shared storage, the models it already has go with it).
     * A model whose name is already there with the same contents (another
     * build downloaded it too) is just removed from [from]; another clash
     * gets a numbered name, and the active-model marker follows it. Each one
     * is copied to a staging file, fsynced and renamed into place before its
     * source is deleted, so a failure — including too little free space,
     * checked first — leaves it where it was, still installed (see
     * [installedFiles]). Blocking.
     */
    fun adoptModels(from: File): Adoption = synchronized(importLock) {
        val dir = modelDirectory()
        if (from.canonicalFile == dir.canonicalFile) return Adoption(0, 0)
        val sources = from.listFiles { file -> file.isFile && isModelName(file.name) }.orEmpty()
        var moved = 0
        var remaining = 0
        if (sources.isNotEmpty()) sweepStaleStaging(dir)
        for (source in sources) {
            // Compared outside the folder lock (it reads both files end to
            // end), then dropped under it once the shared copy is still the
            // one compared: another build may have deleted or replaced it.
            val existing = File(dir, source.name)
            val stamp = existing.length() to existing.lastModified()
            val same = runCatching { existing.isFile && sameContents(existing, source) }.getOrDefault(false)
            val duplicate = same && withFolderLock(dir) {
                val unchanged = existing.isFile && (existing.length() to existing.lastModified()) == stamp
                if (unchanged) dropAdopted(source, existing)
                unchanged
            }
            if (duplicate) {
                moved++
                continue
            }
            val (wasActive, generation) = synchronized(markerLock) { (readActiveName() == source.name) to markerGeneration }
            dir.mkdirs()
            if (dir.usableSpace < source.length() + ADOPT_SPACE_MARGIN_BYTES) {
                runCatching { Log.w(TAG, "Not enough space in $dir for ${source.name}") }
                remaining++
                continue
            }
            val staged = stagingFile(dir)
            val placed = runCatching {
                source.inputStream().use { input ->
                    FileOutputStream(staged).use { output ->
                        input.copyTo(output, BUFFER_SIZE)
                        output.fd.sync()
                    }
                }
                withFolderLock(dir) {
                    val target = freeModelFile(dir, source.name)
                    Files.move(staged.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE)
                    target
                }.also { fsyncDirectory(dir) }
            }
            val target = placed.getOrNull()
            if (target == null) {
                deleteFile(staged, "staging file")
                runCatching { Log.w(TAG, "Could not move ${source.name} to $dir", placed.exceptionOrNull()) }
                remaining++
                continue
            }
            // The moved model stays the active one, unless the user picked
            // a model meanwhile.
            if (wasActive && target.name != source.name) {
                synchronized(markerLock) {
                    if (markerGeneration == generation && readActiveName() == source.name) {
                        runCatching { writeActiveName(target.name) }
                    }
                }
            }
            dropAdopted(source, target)
            moved++
        }
        Adoption(moved, remaining)
    }

    /**
     * Deletes [source] once [copy] holds the same bytes. Under the engine
     * lock, like [deleteModel], so a load never opens a path that is going
     * away; a model resident from [source] counts as loaded from [copy].
     */
    private fun dropAdopted(source: File, copy: File) = synchronized(lock) {
        if (loadedFile == source) loadedFile = copy
        deleteFile(source, "moved model")
    }

    /** Whether [a] and [b] hold the same bytes. */
    private fun sameContents(a: File, b: File): Boolean {
        if (a.length() != b.length()) return false
        a.inputStream().buffered(BUFFER_SIZE).use { left ->
            b.inputStream().buffered(BUFFER_SIZE).use { right ->
                val x = ByteArray(BUFFER_SIZE)
                val y = ByteArray(BUFFER_SIZE)
                while (true) {
                    val n = left.readNBytesCompat(x)
                    if (n != right.readNBytesCompat(y)) return false
                    if (n == 0) return true
                    for (i in 0 until n) if (x[i] != y[i]) return false
                }
            }
        }
    }

    /** Fills [buffer] as far as the stream allows; 0 at its end. */
    private fun java.io.InputStream.readNBytesCompat(buffer: ByteArray): Int {
        var total = 0
        while (total < buffer.size) {
            val read = read(buffer, total, buffer.size - total)
            if (read < 0) break
            total += read
        }
        return total
    }

    /**
     * Runs [block] holding the lock of the model folder [dir], shared with
     * every other Starling build (a `.lock` file there, plus an in-process
     * monitor: Java file locks are per process). Guards what must not
     * interleave with another build: choosing a free name and renaming into
     * it, and deleting; it is only ever held for those short steps. Best
     * effort: a folder that refuses the lock file, or a lock another process
     * holds for over [FOLDER_LOCK_WAIT_MILLIS] (frozen in the background,
     * say), runs [block] unguarded, as before sharing existed.
     */
    private inline fun <T> withFolderLock(dir: File, block: () -> T): T {
        dir.mkdirs()
        synchronized(folderMonitor(dir)) {
            val channel = runCatching {
                FileChannel.open(File(dir, FOLDER_LOCK_NAME).toPath(), StandardOpenOption.CREATE, StandardOpenOption.WRITE)
            }.onFailure { e -> runCatching { Log.w(TAG, "model folder lock file unavailable in $dir", e) } }
                .getOrNull() ?: return block()
            try {
                val held = runCatching { lockWithin(channel, FOLDER_LOCK_WAIT_MILLIS) }
                    .onFailure { e -> runCatching { Log.w(TAG, "model folder lock unavailable in $dir", e) } }
                    .getOrNull()
                try {
                    return block()
                } finally {
                    runCatching { held?.release() }
                }
            } finally {
                runCatching { channel.close() }
            }
        }
    }

    /** [channel]'s exclusive lock, polled for up to [millis]; throws when it stays taken. */
    private fun lockWithin(channel: FileChannel, millis: Long): java.nio.channels.FileLock {
        val deadline = System.nanoTime() + millis * 1_000_000
        while (true) {
            channel.tryLock()?.let { return it }
            if (System.nanoTime() > deadline) throw IOException("held by another process for over $millis ms")
            Thread.sleep(FOLDER_LOCK_POLL_MILLIS)
        }
    }

    private fun stagingFile(dir: File) = File(dir, "$DEFAULT_MODEL_NAME.${UUID.randomUUID()}.importing")

    /**
     * Validates [staged], promotes it to the file [target] picks, and makes
     * that the active model; [staged] is gone afterwards. Holds [importLock],
     * so the target cannot be taken between choosing and promoting it.
     */
    private fun publishStaged(
        staged: File,
        promote: (staged: File, target: File) -> Boolean,
        target: () -> File,
    ): ImportResult {
        val size = staged.length()
        try {
            // Validation reads only the staging file, so it runs outside the
            // engine lock; a concurrent transcription keeps serving the
            // previous model meanwhile.
            val model = when (val detection = ModelFiles.inspect(staged)) {
                is ModelKind.Detection.Unsupported ->
                    return ImportResult.Rejected(detection.reason, ImportStage.VALIDATE)
                is ModelKind.Detection.Supported -> detection.displayName
            }

            // The name is chosen and taken under the folder lock, so another
            // build sharing the folder cannot pick it in between.
            val modelFile = withFolderLock(staged.parentFile!!) {
                val modelFile = target()
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
                    fsyncDirectory(modelFile.parentFile)
                    // The new file is already in place; a native failure here
                    // must not mask a completed import (the next transcription
                    // reloads from the new file anyway).
                    runCatching { unload() }
                }
                modelFile
            }
            // A fresh import is what the user wants to use next. The model is
            // installed even if this marker write fails.
            runCatching { writeActiveName(modelFile.name) }
            return ImportResult.Imported(size, modelFile.name, model)
        } finally {
            // Every path that reaches here without a rename-based promotion
            // leaves the staged file behind: rejection, failed promotion, a
            // copy-based test seam, or an unexpected throw. Delete it so at
            // most one staging file ever exists.
            deleteFile(staged, "staging file")
        }
    }

    /**
     * Where this install's in-progress download of [spec] lives (resumable
     * across restarts). Named for [partialTag], so builds sharing the folder
     * never write or clean up each other's. A partial already in [ownDir]
     * (paused before the models moved, or by a build that named partials
     * without the tag) is resumed there.
     */
    fun downloadFile(spec: ModelDownload): File {
        val stem = "download-${spec.sha256.take(16)}"
        val name = "$stem.$partialTag.part"
        return listOf(File(ownDir, name), File(ownDir, "$stem.part")).firstOrNull { it.isFile }
            ?: File(modelDirectory(), name)
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
    private fun fsyncDirectory(dir: File?) {
        val directory = dir?.takeIf { it.isDirectory } ?: return
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
     * Removes staging files left by interrupted imports in [dir], download
     * partials nobody resumed for [STALE_PARTIAL_MILLIS] (say, of a build
     * that was uninstalled), and a marker temporary left by a crash
     * mid-write.
     * Only ever called with [importLock] held, so none of this engine's
     * staging files can be in use. Another Starling build sharing [dir] may
     * be writing one, though, so only staging files untouched for
     * [STALE_STAGING_MILLIS] are removed.
     */
    private fun sweepStaleStaging(dir: File, keep: File? = null) {
        val cutoff = System.currentTimeMillis() - STALE_STAGING_MILLIS
        val partialCutoff = System.currentTimeMillis() - STALE_PARTIAL_MILLIS
        val stale = dir.listFiles { file ->
            file.isFile && (
                (file.name.endsWith(".importing") && file.lastModified() < cutoff) ||
                    (file.name.startsWith("download-") && file.name.endsWith(".part") && file.lastModified() < partialCutoff)
                )
        }.orEmpty().filter { it.absoluteFile != keep?.absoluteFile }
        // A marker write in flight holds markerLock; only an orphan is removed.
        synchronized(markerLock) { File(ownDir, "$ACTIVE_FILE_NAME.tmp").delete() }
        // Under the folder lock, like every other deletion in a folder other
        // builds may share.
        if (stale.isNotEmpty()) withFolderLock(dir) { for (file in stale) deleteFile(file, "stale staging file") }
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
    private fun ensureLoadedLocked(active: File? = activeModelFile()): String? {
        val modelFile = active
            ?: return "Download or import a speech model first to transcribe on this device."
        // The user picked another model since this one was loaded. A live
        // session keeps the loaded one: a mid-recording reload of hundreds
        // of MB would stall the stream (the switch follows the session).
        if (handle != 0L && loadedFile != modelFile && pinnedSessions == 0) unload()
        if (handle != 0L) return null
        val failed = try {
            awake { loadModelLocked(modelFile) }
        } catch (t: Throwable) {
            // A missing or broken native library throws; the state must not
            // stay "loading", and the failure counts for the backoff.
            observer.loadFailed(modelFile.name, t.message ?: t::class.java.simpleName)
            throw t
        }
        if (failed != null) observer.loadFailed(modelFile.name, failed)
        return failed
    }

    /** Loads and warms up [modelFile]; null when it is resident. Caller holds [lock]. */
    private fun loadModelLocked(modelFile: File): String? {
        nativeSupport()?.let { reason ->
            loadError = reason
            return "The on-device engine cannot run here: $reason"
        }
        // The kind picks the engine, so it is read from the file being
        // loaded (bounded, like the import's check) rather than remembered:
        // a file replaced since its import still loads as what it is.
        val kind = when (val detection = ModelFiles.inspect(modelFile)) {
            is ModelKind.Detection.Unsupported -> {
                loadError = detection.reason
                return detection.reason
            }
            is ModelKind.Detection.Supported -> detection.kind
        }
        memoryGate(modelFile.length(), kind)?.let { reason ->
            loadError = reason
            return "The on-device model was not loaded: $reason"
        }
        observer.loading(modelFile.name)
        NativeSupport.applyThreadDefault()
        // The fast engine's pipeline cache and GPU-wedge marker stay with
        // this install, not in a model folder other builds share.
        NativeSupport.applyFastEngineDefaults(ownDir)
        val abi = StarlingNative.abiVersion()
        if (abi != StarlingNative.EXPECTED_ABI_VERSION) {
            loadError = "engine ABI $abi, expected ${StarlingNative.EXPECTED_ABI_VERSION}"
            return "The on-device engine is incompatible: $loadError"
        }
        val loaded = StarlingNative.load(modelFile.absolutePath, kind.nativeKind)
        if (loaded == 0L) {
            val reason = StarlingNative.lastError(0L) ?: "the model could not be loaded"
            loadError = reason
            return "The on-device model failed to load: $reason"
        }
        handle = loaded
        loadedFile = modelFile
        loadedKind = kind
        loadGeneration++
        residentGeneration = loadGeneration
        loadError = null
        // Absorb lazy graph construction before the first real request,
        // mirroring starling-serve's warmup.
        if (StarlingNative.transcribe(handle, FloatArray(Warmup.SAMPLES), Warmup.SAMPLE_RATE) == null) {
            val error = StarlingNative.lastError(handle)
            // The same discrimination as ModelLifetime: an allocation that
            // did not fit (another model, or more free memory, may work) or
            // a preflight refusal is recoverable, so the warmup leaves the
            // model resident for the first real request to report; only a
            // lost/wedged device is fatal to the handle.
            if (error != null && ModelLifetime.isDriverFailure(error)) {
                // #325: a GPU driver that fails the warmup fails every request
                // after it. Report it instead of serving from a broken handle;
                // the caller decides whether a retry is wanted (speculative
                // preloads never retry it, see ModelLifetime).
                unload()
                loadError = error
                return "The on-device engine failed its warmup: $error"
            }
            // Any other warmup failure is left to the first real request,
            // which reports its own error (the pre-#229 behavior).
        }
        observer.loaded(modelFile.name)
        return null
    }

    /**
     * Brackets one use of the engine. Caller holds [lock]. When the use ends
     * with the model resident and no live session running, [Observer.idle]
     * hands the lifetime policy the point an idle release is measured from.
     */
    private inline fun <T> usingLocked(block: () -> T): T {
        useGeneration.incrementAndGet()
        try {
            return block()
        } finally {
            reportIdleLocked()
        }
    }

    /**
     * Runs one bounded native call with the CPU held awake (see
     * [keepAwake]), timed for [nativeCallAgeMillis]. A load frees a model
     * that fails its warmup inside its own call; the outer call keeps the
     * timing.
     */
    private inline fun <T> awake(call: () -> T): T = keepAwake().use {
        val outermost = nativeCallSince == NO_NATIVE_CALL
        if (outermost) nativeCallSince = System.nanoTime()
        try {
            call()
        } finally {
            if (outermost) nativeCallSince = NO_NATIVE_CALL
        }
    }

    /**
     * How long the native call in progress (load, transcription window,
     * free) has been running; 0 when none is. Never takes the engine lock.
     */
    fun nativeCallAgeMillis(): Long {
        val since = nativeCallSince
        return if (since == NO_NATIVE_CALL) 0L else (System.nanoTime() - since) / 1_000_000
    }

    /** Ends a use: a new generation, reported as idle when nothing else is using the model. Caller holds [lock]. */
    private fun reportIdleLocked() {
        val generation = useGeneration.incrementAndGet()
        if (handle != 0L && liveSessions.get() == 0) observer.idle(generation)
    }

    /**
     * Loads the model for a live session that just started and pins it to
     * the session (see [pinnedSessions]); null when ready. Blocking.
     */
    override fun prepare(cancelled: () -> Boolean): String? = synchronized(lock) {
        // A new take uses the model selected when it started, never the one
        // an earlier take (still finalizing) has pinned: wait, with the lock
        // released, for those takes to end so the switch can happen. The new
        // take's audio is being saved and backlogged meanwhile. The slice
        // re-checks the selection, which can change while waiting.
        while (!cancelled() && handle != 0L && pinnedSessions > 0 && loadedFile != activeModelFile()) {
            lock.wait(PIN_WAIT_SLICE_MILLIS)
        }
        // A closed session must not load (and so undo a deferred release).
        if (cancelled()) return "the live session was closed"
        usingLocked {
            ensureLoadedLocked().also {
                // Pins the loaded model to this session until
                // liveSessionEnded(prepared = true). Invariant: every path
                // that decrements [pinnedSessions] must lock.notifyAll(),
                // or a take waiting above strands for a full
                // PIN_WAIT_SLICE_MILLIS per slice.
                if (it == null) pinnedSessions++
            }
        }
    }

    /**
     * The [ModelLifetime] preload: loads the active model ahead of a
     * recording the user is about to start, or touches the resident one
     * (a use, so the idle release re-arms). [allowed] is checked once the
     * engine lock is held, so speculative work that queued behind a request
     * which then hit a fatal driver failure never runs. Pins nothing; null
     * when ready or skipped. Blocking.
     */
    fun preload(allowed: (model: String) -> Boolean): String? = synchronized(lock) {
        // Resolved once: the backoff check and the load see the same model,
        // even if a selection lands in between.
        val model = activeModelFile() ?: return null
        if (!allowed(model.name)) return null
        usingLocked { ensureLoadedLocked(model) }
    }

    override fun loadGeneration(): Long = residentGeneration

    /** The model file a prepared live session is using. */
    override fun loadedModelName(): String? = synchronized(lock) { loadedFile?.name }

    /** One live-stream window of 16 kHz mono samples. Blocking. */
    override fun transcribeWindow(samples: FloatArray): OnDeviceStreamSession.WindowResult =
        transcribeLive(samples, cancel = null)

    /** A live-tail preview that stops at the engine's next checkpoint once [cancel] holds. Blocking. */
    override fun transcribePreview(samples: FloatArray, cancel: () -> Boolean): OnDeviceStreamSession.WindowResult =
        transcribeLive(samples, StarlingNative.Cancel(cancel))

    private fun transcribeLive(
        samples: FloatArray,
        cancel: StarlingNative.Cancel?,
    ): OnDeviceStreamSession.WindowResult = synchronized(lock) {
        usingLocked {
            ensureLoadedLocked()?.let { return OnDeviceStreamSession.WindowResult.Failed(it) }
            val text = awake {
                if (cancel == null) {
                    StarlingNative.transcribe(handle, samples, ChunkStreamer.SAMPLE_RATE)
                } else {
                    StarlingNative.transcribeCancellable(handle, samples, ChunkStreamer.SAMPLE_RATE, cancel)
                }
            }
            if (text == null) {
                val error = StarlingNative.lastError(handle)
                // Stopped on request: nothing failed, and the model stays.
                if (cancel != null && error == StarlingNative.CANCELLED_ERROR) {
                    return OnDeviceStreamSession.WindowResult.Cancelled
                }
                error?.let(observer::engineFailed)
                if (error != null && ModelLifetime.isDriverFailure(error)) {
                    releaseDriverFailureLocked(error)
                }
                return OnDeviceStreamSession.WindowResult.Failed(
                    "the on-device engine returned an error: ${error ?: "unknown error"}",
                )
            }
            OnDeviceStreamSession.WindowResult.Text(text, loadedFile?.name, loadGeneration)
        }
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
    fun releaseWhenIdle() = synchronized(lock) { releaseLocked() }

    /**
     * The idle policy's release ([ModelLifetime]): frees the resident model
     * only when nothing used it since [generation] (see [Observer.idle]) and
     * no live session is running, so a timer that fires late can never pull
     * the model out from under a recording or a use that just started.
     * Returns whether the model was freed. Blocking.
     */
    fun releaseIfIdle(generation: Long): Boolean = synchronized(lock) {
        // A session starting right after this check (it never takes the lock)
        // finds the model gone and reloads it in its prepare(); its audio is
        // being saved meanwhile, so the race costs a reload, never audio.
        if (handle == 0L || useGeneration.get() != generation || pinnedSessions > 0 || liveSessions.get() > 0) {
            return false
        }
        unload()
        true
    }

    /** Never takes the engine lock: called on the main thread when a recording starts. */
    override fun liveSessionStarted() {
        liveSessions.incrementAndGet()
        useGeneration.incrementAndGet()
    }

    override fun liveSessionEnded(prepared: Boolean) {
        liveSessions.updateAndGet { maxOf(0, it - 1) }
        synchronized(lock) {
            if (prepared && pinnedSessions > 0) {
                pinnedSessions--
                // Tied to the decrement itself (see prepare()'s invariant):
                // a take waiting there for this pin to drop must be woken,
                // so any future path that decrements cannot strand it.
                lock.notifyAll()
            }
            if (pinnedSessions == 0 && releasePending) {
                releasePending = false
                unload()
            }
            // Reported as idle only when no session is live (see reportIdleLocked).
            reportIdleLocked()
        }
    }

    /**
     * Blocking transcription of a finalized WAV recording, with the active
     * model or the installed model named [modelName] (a retry with another
     * model; the active selection does not change). A GPU driver failure, in
     * the load's warmup or mid-transcription, has freed the model; it is
     * retried once, and the reload falls back to the CPU engine. A success
     * names the model file that produced it.
     *
     * With an [attempt] (#356, [BoundedTranscription]) the call is started
     * once it holds the engine, never when it was given up while waiting
     * for it, and stops at the engine's next checkpoint once the attempt is
     * stopped; the model stays loaded.
     */
    fun transcribe(audioFile: File, modelName: String? = null, attempt: CallAttempt? = null): InferenceResult =
        synchronized(lock) {
            if (attempt != null && !attempt.start()) return attempt.stopped()
            usingLocked {
                val result = transcribeLocked(audioFile, modelName, attempt)
                if (result is InferenceResult.Failure && ModelLifetime.isDriverFailure(result.message) &&
                    attempt?.cancelRequested() != true
                ) {
                    transcribeLocked(audioFile, modelName, attempt)
                } else {
                    result
                }
            }
        }

    /** Caller holds [lock] inside [usingLocked]. */
    private fun transcribeLocked(audioFile: File, modelName: String?, attempt: CallAttempt?): InferenceResult {
        val requested = if (modelName == null) {
            activeModelFile()
        } else {
            installedFile(modelName)
                ?: return InferenceResult.Failure("The model $modelName is no longer installed.", false)
        }
        ensureLoadedLocked(requested)?.let { return InferenceResult.Failure(it, false) }
        // A live take keeps its model loaded (see ensureLoadedLocked); a
        // retry that asked for another one must not silently use it.
        if (modelName != null && loadedFile != requested) {
            return InferenceResult.Failure(
                "The model $modelName cannot be loaded while a recording is using ${loadedFile?.name}; " +
                    "retry once it has finished.",
                false,
            )
        }

        val decoded = WavPcm.decodeMonoFloat(audioFile)
            ?: return InferenceResult.Failure("The recording audio could not be decoded.", false)
        if (decoded.sampleRate != WavWriterContract.SAMPLE_RATE) {
            return InferenceResult.Failure(
                "The recording sample rate (${decoded.sampleRate} Hz) is not supported.",
                false,
            )
        }
        // Bound each engine call like the serving layer (Parakeet: 30 s step,
        // 2 s overlap): one full-attention pass over the whole clip would
        // scale quadratically with the recording length, and the LLM-decoder
        // kinds' encoders see at most 30 s.
        val windowSeconds = (loadedKind ?: ModelKind.PARAKEET).maxWindowSeconds
        val windows = ChunkedTranscription.planWindows(decoded.samples.size, decoded.sampleRate, windowSeconds)
        val texts = ArrayList<String>(windows.size)
        val cancel = attempt?.let { StarlingNative.Cancel(it::cancelRequested) }
        for (window in windows) {
            if (attempt?.cancelRequested() == true) return attempt.stopped()
            val samples = if (window.start == 0 && window.endExclusive == decoded.samples.size) {
                decoded.samples
            } else {
                decoded.samples.copyOfRange(window.start, window.endExclusive)
            }
            val text = awake {
                if (cancel == null) {
                    StarlingNative.transcribe(handle, samples, decoded.sampleRate)
                } else {
                    StarlingNative.transcribeCancellable(handle, samples, decoded.sampleRate, cancel)
                }
            }
            if (text == null) {
                val error = StarlingNative.lastError(handle)
                // Stopped by its attempt (#356): not an engine failure.
                if (attempt != null && error == StarlingNative.CANCELLED_ERROR) return attempt.stopped()
                error?.let(observer::engineFailed)
                if (error != null && ModelLifetime.isDriverFailure(error)) {
                    releaseDriverFailureLocked(error)
                }
                return InferenceResult.Failure("The on-device engine returned an error: ${error ?: "unknown error"}", false)
            }
            texts.add(text)
        }
        // A single window is the direct path; joining would only normalize.
        val text = if (texts.size == 1) texts[0] else ChunkedTranscription.joinTexts(texts)
        return InferenceResult.Success(text, loadedFile?.name)
    }

    /**
     * A request that failed through the GPU driver (#325) leaves the
     * resident handle wedged: free it now, pinned sessions or not. Pins
     * protect a working model from a mid-take reload; a wedged one serves
     * nobody, and keeping it until the pins drop would let the batch
     * fallback of the failed take (which can win the lock before that take
     * unpins) run into the same broken handle. The next load falls back to
     * the CPU engine: a live window still returns its failure and the next
     * one reloads, [transcribe] retries at once, and speculative preloads
     * stay off through ModelLifetime's DriverFailed. Caller holds [lock].
     */
    private fun releaseDriverFailureLocked(error: String) {
        unload()
        loadError = error
    }

    private fun unload() {
        if (handle != 0L) {
            // Destroys the GPU device with the last engine: awake, like all GPU work.
            awake { StarlingNative.free(handle) }
            handle = 0L
            loadedFile = null
            loadedKind = null
            residentGeneration = -1L
            loadError = null
            observer.unloaded()
        }
    }

    companion object {
        private const val TAG = "OnDeviceEngine"

        /** Name for a model imported without a usable file name. */
        const val DEFAULT_MODEL_NAME = "parakeet.gguf"

        private const val NO_NATIVE_CALL = Long.MIN_VALUE

        /** How often a take waiting for a pinned model re-checks the selection. */
        private const val PIN_WAIT_SLICE_MILLIS = 1_000L
        private const val ACTIVE_FILE_NAME = "active-model"
        private const val MODEL_EXTENSION = ".gguf"
        private const val MAX_NAME_CHARS = 120
        private val UNSAFE_NAME_CHARS = Regex("[^A-Za-z0-9._-]")

        /** A staging file untouched this long belongs to no running import. */
        private const val STALE_STAGING_MILLIS = 60L * 60 * 1000

        /** A download partial untouched this long is given up. */
        private const val STALE_PARTIAL_MILLIS = 7L * 24 * 60 * 60 * 1000

        /** Free space [adoptModels] keeps beyond the model it moves. */
        private const val ADOPT_SPACE_MARGIN_BYTES = 256L * 1024 * 1024

        /** The lock file builds sharing a model folder coordinate on (hidden: not a model). */
        private const val FOLDER_LOCK_NAME = ".lock"
        private const val FOLDER_LOCK_WAIT_MILLIS = 30_000L
        private const val FOLDER_LOCK_POLL_MILLIS = 50L

        private val folderMonitors = java.util.concurrent.ConcurrentHashMap<String, Any>()

        private fun folderMonitor(dir: File): Any = folderMonitors.getOrPut(dir.absolutePath) { Any() }

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
                .replace(UNSAFE_NAME_CHARS, "_")
                .trimStart('.')
                .take(MAX_NAME_CHARS)
                // Truncation (or the source) can leave a stray separator
                // before the extension this appends.
                .trimEnd('.', '_', '-')
            // Leading dots are gone, so a bare ".gguf" is now "gguf".
            if (base.isEmpty() || base.equals(MODEL_EXTENSION.removePrefix("."), ignoreCase = true)) {
                return DEFAULT_MODEL_NAME
            }
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
