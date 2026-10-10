package dev.starling.mobile.storage

import android.content.Context
import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.AudioRemoval
import dev.starling.mobile.data.CaptureRecovery
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.data.RetentionClass
import dev.starling.mobile.data.RetireReason
import dev.starling.mobile.data.TranscriptRevision
import dev.starling.mobile.data.TranscriptSource
import dev.starling.mobile.data.TranscriptionProvenance
import org.json.JSONArray
import org.json.JSONObject
import java.io.EOFException
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.io.RandomAccessFile
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Small dependency-free durable queue for recordings.
 *
 * Each item has its own metadata file. This avoids rewriting a single index
 * when a response arrives and makes an interrupted write recoverable: the
 * previous metadata file remains in place until the replacement is complete.
 *
 * Audio outlives inference (#356): nothing here deletes audio except an
 * explicit [delete]. The store is opened once per process, before any
 * capture, so at open every capture the last process left unfinished is
 * recovered from its partial WAV ([recoverInterrupted]), and a transcription
 * it left running is marked failed and retryable. Recovery runs only there,
 * never on [list] or [get], so it cannot touch a WAV a capture of this
 * process is still writing.
 *
 * At rest (#342) a finished take is lossless FLAC ([compressAudio]); every
 * transcription reads it back as the exact WAV it replaced
 * ([withRequestAudio]). Audio a capture, transcription or retry is using
 * is pinned and never compressed or removed. The only other path that
 * removes audio is the retention policy the user set ([applyRetention]),
 * and it keeps the recording and its transcripts.
 */
class RecordingStore internal constructor(
    private val directory: File,
    private val move: (File, File) -> Unit = Durability::replace,
    private val clock: () -> Long = System::currentTimeMillis,
) {
    constructor(context: Context) : this(File(context.applicationContext.filesDir, "recordings"))

    private val lock = Any()

    // Guarded by [lock]: how many users hold each take's audio (pin), and a
    // counter every metadata write or delete moves, so a retention run
    // notices that its view of a class went stale.
    private val pins = HashMap<String, Int>()
    private var generation = 0L

    // Guarded by [lock]: takes a compression is encoding right now. A second
    // compression of the same take would write the same temporary.
    private val compressing = HashSet<String>()

    /** Test hook: runs at the named step of [compressAudio] (see [CompressionStep]). */
    internal var compressionHook: ((CompressionStep) -> Unit)? = null

    init {
        if (!directory.exists() && !directory.mkdirs()) {
            throw IOException("Unable to create private recording directory")
        }
        sweepOrphanedTemporaries()
        sweepEphemeral()
        recoverInterrupted()
        reconcileAtRest()
    }

    /**
     * [ephemeral] marks a take from a private field: it is hidden from
     * [list] and is the caller's to [delete] once the take settles.
     */
    fun create(ephemeral: Boolean = false): Recording = synchronized(lock) {
        val id = UUID.randomUUID().toString()
        val recording = Recording(
            id = id,
            createdAtMillis = clock(),
            wavName = "$id.wav",
            status = RecordingStatus.RECORDING,
            ephemeral = ephemeral,
        )
        save(recording)
        recording
    }

    /** The history: every recording except ephemeral ones. */
    fun list(): List<Recording> = synchronized(lock) {
        allRecordings()
            .filterNot { it.ephemeral }
            .sortedByDescending { it.createdAtMillis }
    }

    fun get(id: String): Recording = synchronized(lock) {
        requireValidId(id)
        val file = metadataFile(id)
        if (!file.isFile) throw IOException("Recording is no longer available")
        return decode(file)
    }

    fun partialFile(recording: Recording): File = File(directory, "${recording.id}.wav.part")

    /** The take's audio at rest: the FLAC once compressed, else the WAV. */
    fun audioFile(recording: Recording): File =
        flacFile(recording.id).takeIf(File::isFile) ?: wavFile(recording.id)

    private fun wavFile(id: String): File = File(directory, "$id.wav")

    private fun flacFile(id: String): File = File(directory, "$id.flac")

    /** Atomically promotes the completed WAV from its partial file. */
    fun commitAudio(recording: Recording, durationSeconds: Double): Recording = synchronized(lock) {
        val partial = partialFile(recording)
        if (!isFinalizedWav(partial)) {
            throw IOException("Recording did not produce a complete WAV")
        }
        move(partial, wavFile(recording.id))
        syncDirectory()
        val updated = recording.copy(
            status = RecordingStatus.PENDING,
            durationSeconds = durationSeconds,
            errorMessage = null,
        )
        save(updated)
        updated
    }

    /**
     * Settles a capture that did not complete cleanly (the microphone
     * failed, the disk filled up, the WAV could not be committed): whatever
     * audio its partial WAV holds becomes the recording, PENDING and marked
     * with a [CaptureRecovery] so it never reads as a complete take. Without
     * any audio the recording is FAILED with [message].
     */
    fun salvageCapture(id: String, message: String): Recording = synchronized(lock) {
        requireValidId(id)
        salvageLocked(get(id), message)
    }

    fun markFailed(id: String, message: String): Recording = synchronized(lock) {
        update(id) { it.copy(status = RecordingStatus.FAILED, errorMessage = message) }
    }

    fun markPending(id: String): Recording = synchronized(lock) {
        update(id) { it.copy(status = RecordingStatus.PENDING, errorMessage = null) }
    }

    fun markTranscribing(id: String): Recording = synchronized(lock) {
        update(id) {
            it.copy(
                status = RecordingStatus.TRANSCRIBING,
                attempts = it.attempts + 1,
                errorMessage = null,
            )
        }
    }

    /**
     * Stores a successful transcript as a new revision; earlier revisions
     * (from the first attempt or other models) are kept. An empty result
     * never displaces an earlier non-empty one: that attempt is recorded as
     * failed ([NO_SPEECH_ON_RETRY]) and the earlier transcript stays current.
     */
    fun markTranscribed(
        id: String,
        rawTranscript: String,
        provenance: TranscriptionProvenance = TranscriptionProvenance.BATCH_UPLOAD,
        source: TranscriptSource? = null,
        model: String? = null,
    ): Recording = synchronized(lock) {
        // Deliberately do not trim, normalize, or otherwise clean this value.
        update(id) {
            if (rawTranscript.isBlank() && it.revisions.any { revision -> revision.text.isNotBlank() }) {
                return@update it.copy(status = RecordingStatus.FAILED, errorMessage = NO_SPEECH_ON_RETRY)
            }
            it.copy(
                status = RecordingStatus.TRANSCRIBED,
                rawTranscript = rawTranscript,
                errorMessage = null,
                provenance = provenance,
                revisions = it.revisions + TranscriptRevision(rawTranscript, provenance, source, model, clock()),
            )
        }
    }

    /** Deletion is only called by an explicit user action. */
    fun delete(id: String) = synchronized(lock) {
        requireValidId(id)
        val metadata = metadataFile(id)
        val partial = File(directory, "$id.wav.part")
        val unrecognized = File(directory, "$id.wav.unrecognized")
        val temporary = File(directory, ".$id.json.tmp")
        val temporaries = directory.listFiles { file -> file.name.startsWith(".$id.") }.orEmpty().toList()
        generation++
        // Metadata goes last: until every payload file is gone it stays,
        // so an interrupted or failed delete (of an ephemeral take, too) is
        // found and finished again on the next open.
        val failures = (listOf(wavFile(id), flacFile(id), partial, unrecognized, temporary) + temporaries)
            .filter { it.exists() && !it.delete() }
        if (failures.isNotEmpty() || (metadata.exists() && !metadata.delete())) {
            throw IOException("Unable to delete recording files")
        }
    }

    /** Moves a take to another retention class; its limits apply from the next run. */
    fun setRetentionClass(id: String, retentionClass: RetentionClass): Recording = synchronized(lock) {
        update(id) { it.copy(retentionClass = retentionClass) }
    }

    /**
     * Keeps [id]'s audio where and as it is (no compression, no retention
     * removal) until the returned handle is closed. Closing twice is harmless.
     */
    fun pin(id: String): AutoCloseable = synchronized(lock) {
        requireValidId(id)
        pins[id] = (pins[id] ?: 0) + 1
        val open = AtomicBoolean(true)
        AutoCloseable {
            if (open.getAndSet(false)) {
                synchronized(lock) {
                    val left = (pins[id] ?: 1) - 1
                    if (left > 0) pins[id] = left else pins.remove(id)
                }
            }
        }
    }

    fun isPinned(id: String): Boolean = synchronized(lock) { (pins[id] ?: 0) > 0 }

    /**
     * Runs [block] with the take's audio as the request WAV every
     * transcription sends: the stored WAV, or the FLAC decoded into a
     * temporary WAV that is byte-identical to the WAV it replaced (the
     * header too). The audio stays pinned until [block] returns.
     */
    fun <T> withRequestAudio(id: String, block: (File) -> T): T {
        requireValidId(id)
        pin(id).use {
            val recording = get(id)
            if (recording.audioRemoved != null) throw IOException(AUDIO_REMOVED)
            val wav = wavFile(id)
            val flac = flacFile(id)
            // A pinned take is never compressed, so the WAV, once seen, stays.
            if (!flac.isFile) {
                if (!wav.isFile) throw IOException("The recording audio is missing")
                return block(wav)
            }
            val request = File.createTempFile(".$id.", REQUEST_SUFFIX, directory)
            try {
                FileOutputStream(request).use { output ->
                    FileInputStream(flac).use { input ->
                        Flac.decode(input, output) { info ->
                            if (info.sampleRate != WavWriter.SAMPLE_RATE) throw IOException("Unexpected sample rate in the stored audio")
                            output.write(WavWriter.header(info.totalSamples * WavWriter.BYTES_PER_SAMPLE))
                        }
                    }
                }
                return block(request)
            } finally {
                request.delete()
            }
        }
    }

    /** The take's audio at rest, opened (for playback and export); [flac] says which format it is. */
    class StoredAudio(val stream: FileInputStream, val flac: Boolean)

    /**
     * Opens the take's audio at rest under the store lock, so a compression
     * that publishes meanwhile cannot pull the file from under the caller:
     * an open stream keeps reading a file that is unlinked later.
     */
    fun openAudio(id: String): StoredAudio = synchronized(lock) {
        requireValidId(id)
        val flac = flacFile(id)
        if (flac.isFile) return StoredAudio(FileInputStream(flac), flac = true)
        StoredAudio(FileInputStream(wavFile(id)), flac = false)
    }

    /** The steps of [compressAudio] that [compressionHook] sees. */
    enum class CompressionStep {
        /** The FLAC is written, synced and verified as a temporary; nothing is published. */
        ENCODED,

        /** The FLAC is renamed into place and the directory synced; the WAV is still there. */
        PUBLISHED,
    }

    sealed interface Compression {
        data class Compressed(val wavBytes: Long, val flacBytes: Long) : Compression
        data class Skipped(val reason: String) : Compression
    }

    /** Takes [compressAudio] would compress now. */
    fun compressionCandidates(): List<String> = synchronized(lock) {
        allRecordings().filter { compressionBlocker(it) == null }.sortedBy { it.createdAtMillis }.map { it.id }
    }

    /**
     * Replaces a finished take's WAV with lossless FLAC. The encode runs off
     * the lock into a temporary that is synced and then decoded from
     * storage and compared with the WAV sample for sample. The publish runs
     * under the lock: it checks again that nothing uses the take (pins,
     * capture, transcription, a removal or a delete in between), renames
     * the FLAC into place, syncs the directory, and only then unlinks the
     * WAV. A crash leaves the WAV, the complete FLAC, or both; the next
     * open ([reconcileAtRest]) settles "both". Throws when the encode or the
     * check fails; the WAV is untouched then.
     */
    fun compressAudio(id: String): Compression {
        requireValidId(id)
        val wav = wavFile(id)
        val wavBytes = synchronized(lock) {
            val recording = runCatching { get(id) }.getOrElse { return Compression.Skipped("deleted") }
            compressionBlocker(recording)?.let { return Compression.Skipped(it) }
            compressing += id
            wav.length()
        }
        try {
            return encodeAndPublish(id, wav, wavBytes)
        } finally {
            synchronized(lock) { compressing -= id }
        }
    }

    private fun encodeAndPublish(id: String, wav: File, wavBytes: Long): Compression {
        val dataBytes = wavBytes - WAV_HEADER_BYTES
        val temporary = File(directory, ".$id$FLAC_TEMP_SUFFIX")
        try {
            FileInputStream(wav).use { input ->
                skipFully(input, WAV_HEADER_BYTES)
                FileOutputStream(temporary).use { output ->
                    Flac.encode(input, dataBytes / WavWriter.BYTES_PER_SAMPLE, WavWriter.SAMPLE_RATE, output)
                    output.fd.sync()
                }
            }
            FileInputStream(wav).use { input ->
                skipFully(input, WAV_HEADER_BYTES)
                val comparing = ComparingOutputStream(input)
                FileInputStream(temporary).use { flac -> Flac.decode(flac, comparing) }
                comparing.requireExhausted()
            }
            compressionHook?.invoke(CompressionStep.ENCODED)
        } catch (exception: Exception) {
            temporary.delete()
            throw exception
        }
        return synchronized(lock) {
            val recording = runCatching { get(id) }.getOrNull()
            val blocker = when {
                recording == null -> "deleted"
                wav.length() != wavBytes -> "changed"
                else -> compressionBlocker(recording, publishing = true)
            }
            if (blocker != null) {
                temporary.delete()
                return Compression.Skipped(blocker)
            }
            val flac = flacFile(id)
            try {
                move(temporary, flac)
            } catch (exception: Exception) {
                temporary.delete()
                throw IOException("Unable to publish the compressed audio", exception)
            }
            syncDirectory()
            compressionHook?.invoke(CompressionStep.PUBLISHED)
            if (!wav.delete()) throw IOException("Unable to remove the uncompressed audio")
            runCatching { syncDirectory() }
            Compression.Compressed(wavBytes, flac.length())
        }
    }

    /**
     * Why [recording] cannot be compressed now; null when it can. The
     * compression [publishing] it is the one in [compressing]. Caller holds
     * [lock].
     */
    private fun compressionBlocker(recording: Recording, publishing: Boolean = false): String? {
        val wav = wavFile(recording.id)
        val partial = partialFile(recording)
        return when {
            recording.ephemeral -> "private"
            recording.audioRemoved != null -> "removed"
            recording.status != RecordingStatus.PENDING &&
                recording.status != RecordingStatus.TRANSCRIBED &&
                recording.status != RecordingStatus.FAILED -> "in use"
            recording.errorMessage == UNRECOVERED_CAPTURE || partial.exists() || WavWriter.isOpen(partial) ->
                "not finalized"
            (pins[recording.id] ?: 0) > 0 -> "in use"
            !publishing && recording.id in compressing -> "compressing"
            flacFile(recording.id).exists() -> "compressed"
            // Only the app's own finalized WAV, whose header the request WAV
            // rebuilds byte for byte.
            !isFinalizedWav(wav) || !hasOwnHeader(wav) -> "not a finalized WAV"
            else -> null
        }
    }

    private fun hasOwnHeader(wav: File): Boolean = RandomAccessFile(wav, "r").use { file ->
        val header = ByteArray(WAV_HEADER_BYTES.toInt())
        file.readFully(header)
        header.contentEquals(WavWriter.header(file.length() - WAV_HEADER_BYTES))
    }

    /**
     * Applies the user's retention limits: the only path that removes a
     * take's audio without an explicit Delete, and only the audio — the
     * recording, its transcripts and revisions stay, and it lists as
     * removed. Off unless [gate]'s policy has a limit.
     *
     * Per class, takes are walked newest first. A take is due when it is
     * older than the age limit, or when it and every newer take of the
     * class together exceed the size limit. A due take keeps its audio, and
     * is reported, when it is younger than the grace, in use (pinned,
     * transcribing) or was never transcribed; it still counts toward the
     * class's size. Takes still recording have no audio yet and are not
     * considered.
     *
     * Every removal is decided under the store lock and inside
     * [PolicyGate.withPolicy]: the policy is read again (a change ends the
     * run, [RetentionReport.policyChanged]), [stop] is asked (a take
     * started, [RetentionReport.stopped]), and the class is walked again
     * with every size measured from the files now. The order is stamp,
     * then unlink; the next open finishes an unlink a crash cut off.
     */
    fun applyRetention(gate: PolicyGate, stop: () -> Boolean = { false }): RetentionReport {
        val policy = gate.withPolicy { it }
        if (!policy.isActive) return RetentionReport()
        val removed = mutableListOf<RemovedAudio>()
        val held = LinkedHashMap<String, HeldAudio>()
        val overLimit = LinkedHashMap<RetentionClass, Long>()
        for (retentionClass in RetentionClass.entries) {
            val limits = policy.limits(retentionClass)
            if (!limits.isActive) continue
            var members: List<Recording>? = null
            var seenGeneration = -1L
            while (true) {
                val step = synchronized(lock) {
                    gate.withPolicy { live ->
                        when {
                            live != policy -> RetentionStep.CHANGED
                            stop() -> RetentionStep.STOPPED
                            else -> {
                                if (members == null || seenGeneration != generation) {
                                    members = classMembers(retentionClass)
                                    seenGeneration = generation
                                }
                                val outcome = retireNext(members!!, retentionClass, limits, policy, held)
                                when (outcome) {
                                    is RetentionWalk.Removed -> {
                                        removed += outcome.audio
                                        seenGeneration = generation
                                        members = members!!.filterNot { it.id == outcome.audio.id }
                                        RetentionStep.CONTINUE
                                    }
                                    is RetentionWalk.Done -> {
                                        if (outcome.overLimitBytes > 0) overLimit[retentionClass] = outcome.overLimitBytes
                                        RetentionStep.DONE
                                    }
                                }
                            }
                        }
                    }
                }
                when (step) {
                    RetentionStep.CONTINUE -> continue
                    RetentionStep.DONE -> break
                    RetentionStep.CHANGED, RetentionStep.STOPPED -> return RetentionReport(
                        removed = removed,
                        held = held.values.toList(),
                        policyChanged = step == RetentionStep.CHANGED,
                        stopped = step == RetentionStep.STOPPED,
                    )
                }
            }
        }
        return RetentionReport(removed, held.values.toList(), overLimit)
    }

    private enum class RetentionStep { CONTINUE, DONE, CHANGED, STOPPED }

    private sealed interface RetentionWalk {
        data class Removed(val audio: RemovedAudio) : RetentionWalk
        data class Done(val overLimitBytes: Long) : RetentionWalk
    }

    /** The class's takes that still have audio, newest first. Caller holds [lock]. */
    private fun classMembers(retentionClass: RetentionClass): List<Recording> =
        allRecordings()
            .filter {
                it.retentionClass == retentionClass && !it.ephemeral && it.audioRemoved == null &&
                    it.status != RecordingStatus.RECORDING
            }
            .sortedWith(compareByDescending<Recording> { it.createdAtMillis }.thenByDescending { it.id })

    /**
     * One walk over [members] with sizes measured now: removes the first
     * due take nothing holds, or reports what the size limit still
     * exceeds. Caller holds [lock].
     */
    private fun retireNext(
        members: List<Recording>,
        retentionClass: RetentionClass,
        limits: ClassLimits,
        policy: RetentionPolicy,
        held: MutableMap<String, HeldAudio>,
    ): RetentionWalk {
        val now = clock()
        val ageCutoff = limits.maxAgeDays?.let { now - it * DAY_MILLIS }
        val maxBytes = limits.maxTotalBytes
        var kept = 0L
        for (recording in members) {
            val bytes = audioBytes(recording.id)
            if (bytes == 0L) continue
            kept += bytes
            val reason = when {
                ageCutoff != null && recording.createdAtMillis < ageCutoff -> RetireReason.AGE
                maxBytes != null && kept > maxBytes -> RetireReason.SIZE
                else -> continue
            }
            val hold = holdReason(recording, now, policy)
            if (hold != null) {
                held[recording.id] = HeldAudio(recording.id, retentionClass, bytes, hold)
                continue
            }
            held.remove(recording.id)
            save(recording.copy(audioRemoved = AudioRemoval(now, reason)))
            unlinkAudio(recording.id)
            return RetentionWalk.Removed(RemovedAudio(recording.id, retentionClass, bytes, reason))
        }
        return RetentionWalk.Done(if (maxBytes != null && kept > maxBytes) kept - maxBytes else 0L)
    }

    /** Why a due take keeps its audio, if it does. Caller holds [lock]. */
    private fun holdReason(recording: Recording, now: Long, policy: RetentionPolicy): HoldReason? = when {
        now - recording.createdAtMillis < policy.graceMillis -> HoldReason.RECENT
        (pins[recording.id] ?: 0) > 0 ||
            recording.status == RecordingStatus.TRANSCRIBING ||
            WavWriter.isOpen(partialFile(recording)) -> HoldReason.IN_USE
        recording.revisions.isEmpty() -> HoldReason.UNTRANSCRIBED
        else -> null
    }

    /** The bytes of a take's audio at rest (both files during a compression's publish). */
    private fun audioBytes(id: String): Long =
        listOf(wavFile(id), flacFile(id)).filter(File::isFile).sumOf(File::length)

    /** Removes a take's audio files; a failure is finished at the next open. Caller holds [lock]. */
    private fun unlinkAudio(id: String) {
        wavFile(id).delete()
        flacFile(id).delete()
        runCatching { syncDirectory() }
    }

    private fun allRecordings(): List<Recording> =
        directory.listFiles { file -> file.isFile && file.name.endsWith(".json") }
            .orEmpty()
            .mapNotNull { file -> runCatching { decode(file) }.getOrNull() }

    /**
     * At open, after [recoverInterrupted]: finishes what a crash left of
     * compression and retention. A take holding both its WAV and its FLAC
     * died between the FLAC's publish and the WAV's unlink: when the FLAC
     * decodes to exactly the WAV's samples the WAV goes, otherwise the FLAC
     * does (the WAV is the original). A take whose audio the policy already
     * stamped as removed loses the files the unlink missed.
     */
    private fun reconcileAtRest() {
        allRecordings().forEach { recording ->
            val wav = wavFile(recording.id)
            val flac = flacFile(recording.id)
            runCatching {
                when {
                    recording.audioRemoved != null -> if (wav.exists() || flac.exists()) unlinkAudio(recording.id)
                    wav.isFile && flac.isFile -> {
                        val same = runCatching {
                            FileInputStream(wav).use { input ->
                                skipFully(input, WAV_HEADER_BYTES)
                                val comparing = ComparingOutputStream(input)
                                FileInputStream(flac).use { Flac.decode(it, comparing) }
                                comparing.requireExhausted()
                            }
                        }.isSuccess
                        if (same && hasOwnHeader(wav)) wav.delete() else flac.delete()
                        syncDirectory()
                    }
                }
            }
        }
    }

    /** Fails the moment what is written differs from [expected]. */
    private class ComparingOutputStream(private val expected: InputStream) : OutputStream() {
        private val buffer = ByteArray(1 shl 16)

        override fun write(b: Int) = write(byteArrayOf(b.toByte()), 0, 1)

        override fun write(bytes: ByteArray, offset: Int, length: Int) {
            var done = 0
            while (done < length) {
                val read = expected.read(buffer, 0, minOf(buffer.size, length - done))
                if (read < 0) throw IOException("The compressed audio is longer than the original")
                for (i in 0 until read) {
                    if (buffer[i] != bytes[offset + done + i]) throw IOException("The compressed audio differs from the original")
                }
                done += read
            }
        }

        fun requireExhausted() {
            if (expected.read() >= 0) throw IOException("The compressed audio is shorter than the original")
        }
    }

    private fun skipFully(input: InputStream, count: Long) {
        var left = count
        while (left > 0) {
            val skipped = input.skip(left)
            if (skipped <= 0) {
                if (input.read() < 0) throw EOFException("The audio file is truncated")
                left--
            } else {
                left -= skipped
            }
        }
    }

    private fun update(id: String, transform: (Recording) -> Recording): Recording {
        val current = get(id)
        return transform(current).also(::save)
    }

    /**
     * Remove temporaries leaked by a crash: metadata between the write and
     * the rename in [save], a compression's unpublished FLAC, a decoded
     * request WAV. Nothing reads them back and the store is opened once per
     * process, before anything could be using one, so deleting them on open
     * is safe and keeps the directory from growing without bound on
     * crash-prone devices.
     */
    private fun sweepOrphanedTemporaries() {
        directory.listFiles { file ->
            file.isFile && file.name.startsWith(".") &&
                (file.name.endsWith(".json.tmp") || file.name.endsWith(FLAC_TEMP_SUFFIX) || file.name.endsWith(REQUEST_SUFFIX))
        }?.forEach { file -> file.delete() }
    }

    /**
     * An ephemeral take still on disk at open time outlived the process that
     * dictated it (the store is opened once, at process start, before any
     * capture). Its field was private, so it is deleted rather than offered
     * for retry.
     */
    private fun sweepEphemeral() {
        directory.listFiles { file -> file.isFile && file.name.endsWith(".json") }
            ?.forEach { file ->
                val id = runCatching { decode(file) }.getOrNull()?.takeIf { it.ephemeral }?.id
                if (id != null) runCatching { delete(id) }
            }
    }

    private fun save(recording: Recording) {
        generation++
        val target = metadataFile(recording.id)
        val temporary = File(directory, ".${recording.id}.json.tmp")
        val json = JSONObject()
            .put("id", recording.id)
            .put("created_at_ms", recording.createdAtMillis)
            .put("wav_name", recording.wavName)
            .put("status", recording.status.name)
            .put("duration_s", recording.durationSeconds)
            .put("attempts", recording.attempts)
            .put("raw_transcript", recording.rawTranscript ?: JSONObject.NULL)
            .put("error_message", recording.errorMessage ?: JSONObject.NULL)
            .put("provenance", recording.provenance?.name ?: JSONObject.NULL)
            .put("ephemeral", recording.ephemeral)
            .put("revisions", JSONArray(recording.revisions.map(::encodeRevision)))
            .put("recovery", recording.recovery?.let(::encodeRecovery) ?: JSONObject.NULL)
            .put("retention_class", recording.retentionClass.key)
            .put("audio_removed", recording.audioRemoved?.let(::encodeRemoval) ?: JSONObject.NULL)

        FileOutputStream(temporary).use { output ->
            output.write(json.toString().toByteArray(Charsets.UTF_8))
            output.fd.sync()
        }
        if (!temporary.renameTo(target)) {
            throw IOException("Unable to commit recording metadata")
        }
        syncDirectory()
    }

    /**
     * Makes this directory's renames durable, so after a power loss the
     * metadata and audio renames are on storage in the order they were
     * made. A failed sync throws, so nothing that relies on the rename (a
     * header repair after its recovery note) runs.
     */
    private fun syncDirectory() = Durability.syncDirectory(directory)

    private fun decode(file: File): Recording {
        val json = JSONObject(file.readText(Charsets.UTF_8))
        val id = json.getString("id")
        requireValidId(id)
        val status = runCatching {
            RecordingStatus.valueOf(json.getString("status"))
        }.getOrElse { RecordingStatus.FAILED }
        val wavName = json.getString("wav_name")
        require(wavName == "$id.wav") { "Invalid recording audio name" }
        val createdAtMillis = json.getLong("created_at_ms")
        val rawTranscript = json.optionalString("raw_transcript")
        val provenance = json.optionalString("provenance")
            ?.let { value ->
                runCatching { TranscriptionProvenance.valueOf(value) }.getOrNull()
            }
        val revisions = json.optJSONArray("revisions")
            ?.let { array -> (0 until array.length()).mapNotNull { decodeRevision(array.getJSONObject(it)) } }
            .orEmpty()
        return Recording(
            id = id,
            createdAtMillis = createdAtMillis,
            wavName = wavName,
            status = status,
            durationSeconds = json.optDouble("duration_s", 0.0),
            rawTranscript = rawTranscript,
            errorMessage = json.optionalString("error_message"),
            attempts = json.optInt("attempts", 0),
            provenance = provenance,
            ephemeral = json.optBoolean("ephemeral", false),
            // A transcript saved before revisions existed becomes the first
            // one, so a retry keeps it and an empty retry cannot erase it.
            revisions = if (revisions.isEmpty() && !rawTranscript.isNullOrBlank()) {
                listOf(
                    TranscriptRevision(
                        rawTranscript,
                        provenance ?: TranscriptionProvenance.BATCH_UPLOAD,
                        source = null,
                        model = null,
                        createdAtMillis = createdAtMillis,
                    ),
                )
            } else {
                revisions
            },
            recovery = json.optJSONObject("recovery")?.let(::decodeRecovery),
            retentionClass = json.optionalString("retention_class")
                ?.let { key -> RetentionClass.entries.firstOrNull { it.key == key } }
                ?: RetentionClass.STANDARD,
            audioRemoved = json.optJSONObject("audio_removed")?.let(::decodeRemoval),
        )
    }

    private fun encodeRemoval(removal: AudioRemoval): JSONObject = JSONObject()
        .put("at_ms", removal.atMillis)
        .put("reason", removal.reason.name)

    // A removal stays a removal even when its reason is unreadable.
    private fun decodeRemoval(json: JSONObject): AudioRemoval = AudioRemoval(
        atMillis = json.optLong("at_ms", 0L),
        reason = runCatching { RetireReason.valueOf(json.getString("reason")) }.getOrDefault(RetireReason.AGE),
    )

    private fun encodeRevision(revision: TranscriptRevision): JSONObject = JSONObject()
        .put("text", revision.text)
        .put("provenance", revision.provenance.name)
        .put("source", revision.source?.name ?: JSONObject.NULL)
        .put("model", revision.model ?: JSONObject.NULL)
        .put("created_at_ms", revision.createdAtMillis)

    private fun decodeRevision(json: JSONObject): TranscriptRevision? = runCatching {
        TranscriptRevision(
            text = json.getString("text"),
            provenance = TranscriptionProvenance.valueOf(json.getString("provenance")),
            source = json.optionalString("source")?.let { runCatching { TranscriptSource.valueOf(it) }.getOrNull() },
            model = json.optionalString("model"),
            createdAtMillis = json.getLong("created_at_ms"),
        )
    }.getOrNull()

    private fun encodeRecovery(recovery: CaptureRecovery): JSONObject = JSONObject()
        .put("reason", recovery.reason)
        .put("recovered_s", recovery.recoveredSeconds)
        .put("confirmed_s", recovery.confirmedSeconds)

    private fun decodeRecovery(json: JSONObject): CaptureRecovery = CaptureRecovery(
        reason = json.getString("reason"),
        recoveredSeconds = json.optDouble("recovered_s", 0.0),
        confirmedSeconds = json.optDouble("confirmed_s", 0.0),
    )

    /**
     * At open, before any capture of this process: settles what the previous
     * process left unfinished. A capture that never reached commitAudio (the
     * process was killed, the phone powered off) is recovered from its
     * partial WAV; a recording whose audio was promoted but whose metadata
     * still said RECORDING (the crash window inside commitAudio) becomes
     * PENDING as it was; a transcription that was running is failed with an
     * explanation, so the row offers Retry instead of claiming progress.
     */
    private fun recoverInterrupted() {
        directory.listFiles { file -> file.isFile && file.name.endsWith(".json") }
            ?.forEach { file ->
                val recording = runCatching { decode(file) }.getOrNull() ?: return@forEach
                if (recording.ephemeral) return@forEach
                val audioMissing = !audioFile(recording).exists() && partialFile(recording).exists()
                when {
                    recording.status == RecordingStatus.RECORDING || audioMissing ||
                        recording.errorMessage == UNRECOVERED_CAPTURE -> runCatching {
                        salvageLocked(
                            recording,
                            recording.recovery?.reason
                                ?: recording.errorMessage?.takeIf { it != UNRECOVERED_CAPTURE }
                                ?: INTERRUPTED_CAPTURE,
                        )
                    }.onFailure {
                        // Storage refused the repair or a rename: the row says
                        // so instead of "recording", and the next open tries
                        // again. A recovery note already saved is kept, since
                        // the header it was measured from may be repaired now.
                        runCatching {
                            val current = runCatching { decode(file) }.getOrDefault(recording)
                            save(current.copy(status = RecordingStatus.FAILED, errorMessage = UNRECOVERED_CAPTURE))
                        }
                    }
                    recording.status == RecordingStatus.TRANSCRIBING -> runCatching {
                        save(recording.copy(status = RecordingStatus.FAILED, errorMessage = INTERRUPTED_TRANSCRIPTION))
                    }
                }
            }
    }

    /**
     * Turns whatever audio a capture left into the recording (see
     * [salvageCapture]). The order makes a crash at any step safe to repeat:
     * the partial WAV is measured, the recovery note is saved, and only then
     * is the WAV's header repaired and synced, the WAV promoted and the
     * recording made PENDING. A repeat keeps the first note's (smaller)
     * confirmed size, since by then the header it was read from is repaired.
     * A WAV already promoted without a note is a clean Stop whose metadata
     * commit was cut off, so it is not marked as recovered. The promotion
     * replaces the destination in one rename, so no step leaves the
     * recording without the audio it had. A partial that is not this app's
     * WAV is neither repaired nor deleted: it is set aside unchanged and the
     * recording is FAILED with [UNRECOGNIZED_CAPTURE]. Caller holds [lock].
     */
    private fun salvageLocked(recording: Recording, reason: String): Recording {
        val destination = wavFile(recording.id)
        val partial = partialFile(recording)
        var recovery = recording.recovery
        if (WavWriter.isOpen(partial)) {
            // A capture of this process still holds the file (its worker
            // outlived the stop). Nothing is moved under it; the next start
            // recovers it like any interrupted capture.
            return recording.copy(status = RecordingStatus.FAILED, errorMessage = UNRECOVERED_CAPTURE)
                .also(::save)
        }
        if (partial.isFile) {
            when (val measured = measurePartial(partial)) {
                is PartialContent.Audio -> {
                    val confirmed = seconds(measured.confirmedBytes)
                    recovery = CaptureRecovery(
                        reason = recovery?.reason ?: reason,
                        recoveredSeconds = seconds(measured.dataBytes),
                        confirmedSeconds = recovery?.confirmedSeconds?.let { minOf(it, confirmed) } ?: confirmed,
                    )
                    save(recording.copy(recovery = recovery))
                    repairPartial(partial, measured.dataBytes)
                    move(partial, destination)
                    syncDirectory()
                }
                // A header without a single sample: nothing to keep.
                PartialContent.Empty -> partial.delete()
                PartialContent.Unrecognized -> {
                    val aside = unrecognizedFile(recording)
                    // Never replaced: whatever was set aside before is kept too.
                    if (aside.exists()) throw IOException("Unable to set aside the unrecognized recording audio")
                    // Its bytes may still be only in the page cache of the
                    // process that died; they are on storage before the
                    // recording says they are kept.
                    RandomAccessFile(partial, "rw").use { it.fd.sync() }
                    move(partial, aside)
                    syncDirectory()
                }
            }
        }
        val dataBytes = finalizedDataBytes(destination)
        val salvaged = when {
            dataBytes != null && dataBytes > 0 -> recording.copy(
                status = RecordingStatus.PENDING,
                durationSeconds = seconds(dataBytes),
                errorMessage = null,
                recovery = recovery,
            )
            unrecognizedFile(recording).exists() ->
                recording.copy(status = RecordingStatus.FAILED, errorMessage = UNRECOGNIZED_CAPTURE, recovery = null)
            else -> recording.copy(status = RecordingStatus.FAILED, errorMessage = reason, recovery = null)
        }
        save(salvaged)
        return salvaged
    }

    private sealed interface PartialContent {
        /**
         * The whole-sample payload size, and the part of it the capture had
         * confirmed on storage (the size its last header checkpoint
         * recorded, see WavWriter.checkpoint).
         */
        data class Audio(val dataBytes: Long, val confirmedBytes: Long) : PartialContent

        /** This app's header, or the start of it, without a single sample. */
        object Empty : PartialContent

        /** Not a WAV this app wrote: nothing in it can be read as its audio. */
        object Unrecognized : PartialContent
    }

    /**
     * What a partial WAV holds. WavWriter syncs a complete header before any
     * audio, so a partial whose header is not this app's (16 kHz mono PCM16)
     * is not taken for audio.
     */
    private fun measurePartial(partial: File): PartialContent = RandomAccessFile(partial, "r").use { file ->
        val length = file.length()
        val header = ByteArray(minOf(length, WAV_HEADER_BYTES).toInt())
        file.readFully(header)
        if (length < WAV_HEADER_BYTES) {
            // Cut off while WavWriter wrote its first header.
            val started = header.contentEquals(WavWriter.header(0).copyOf(header.size))
            return if (started) PartialContent.Empty else PartialContent.Unrecognized
        }
        if (!isOwnHeader(header)) return PartialContent.Unrecognized
        val dataBytes = minOf(length - WAV_HEADER_BYTES, WavWriter.MAX_DATA_BYTES) and 1L.inv()
        if (dataBytes <= 0) return PartialContent.Empty
        val headerData = littleEndianInt(header, 40).toLong() and 0xffffffffL
        val riffSize = littleEndianInt(header, 4).toLong() and 0xffffffffL
        // Both sizes come from one checkpoint write; sizes that disagree
        // (a torn write) confirm nothing.
        val confirmed = if (riffSize == headerData + WAV_HEADER_BYTES - 8) headerData.coerceAtMost(dataBytes) else 0L
        PartialContent.Audio(dataBytes, confirmed)
    }

    /** Makes a partial WAV a valid one over its first [dataBytes] of payload, and syncs it. */
    private fun repairPartial(partial: File, dataBytes: Long) = RandomAccessFile(partial, "rw").use { file ->
        file.setLength(WAV_HEADER_BYTES + dataBytes)
        file.seek(0)
        file.write(WavWriter.header(dataBytes))
        file.fd.sync()
    }

    /**
     * A RIFF/WAVE header with this app's format chunk and a data chunk; the
     * two size fields (bytes 4..8 and 40..44) may hold anything.
     */
    private fun isOwnHeader(header: ByteArray): Boolean {
        val expected = WavWriter.header(0)
        return (0 until 4).all { header[it] == expected[it] } && (8 until 40).all { header[it] == expected[it] }
    }

    /** The payload size of a finalized WAV, or null when [file] is not one. */
    private fun finalizedDataBytes(file: File): Long? =
        if (isFinalizedWav(file)) file.length() - WAV_HEADER_BYTES else null

    private fun seconds(dataBytes: Long): Double =
        dataBytes.toDouble() / (WavWriter.SAMPLE_RATE * WavWriter.BYTES_PER_SAMPLE)

    private fun isFinalizedWav(file: File): Boolean = runCatching {
        if (!file.isFile || file.length() < WAV_HEADER_BYTES) return false
        RandomAccessFile(file, "r").use { input ->
            val header = ByteArray(WAV_HEADER_BYTES.toInt())
            input.readFully(header)
            header.copyOfRange(0, 4).contentEquals("RIFF".toByteArray(Charsets.US_ASCII)) &&
                header.copyOfRange(8, 12).contentEquals("WAVE".toByteArray(Charsets.US_ASCII)) &&
                header.copyOfRange(36, 40).contentEquals("data".toByteArray(Charsets.US_ASCII)) &&
                littleEndianInt(header, 40) >= 0 &&
                WAV_HEADER_BYTES + littleEndianInt(header, 40).toLong() == file.length()
        }
    }.getOrDefault(false)

    private fun littleEndianInt(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xff) or
            ((bytes[offset + 1].toInt() and 0xff) shl 8) or
            ((bytes[offset + 2].toInt() and 0xff) shl 16) or
            ((bytes[offset + 3].toInt() and 0xff) shl 24)

    private fun JSONObject.optionalString(key: String): String? =
        if (isNull(key)) null else getString(key)

    private fun metadataFile(id: String): File = File(directory, "$id.json")

    /** Where a partial that is not this app's WAV is kept, unchanged, until the recording is deleted. */
    private fun unrecognizedFile(recording: Recording): File = File(directory, "${recording.id}.wav.unrecognized")

    private fun requireValidId(id: String) {
        require(UUID_PATTERN.matches(id)) { "Invalid recording id" }
    }

    companion object {
        private const val WAV_HEADER_BYTES = 44L
        private const val FLAC_TEMP_SUFFIX = ".flac.tmp"
        private const val REQUEST_SUFFIX = ".request.wav"
        private const val DAY_MILLIS = 24L * 60 * 60 * 1000
        private val UUID_PATTERN = Regex("[0-9a-fA-F-]{36}")

        const val INTERRUPTED_CAPTURE = "The recording was interrupted: the app or the phone stopped while it was recording."
        const val UNRECOVERED_CAPTURE =
            "The recording's audio could not be recovered yet; Starling tries again the next time it starts."
        const val UNRECOGNIZED_CAPTURE =
            "The recording did not finish cleanly, and its partial audio is not a WAV Starling recognizes, " +
                "so no audio could be recovered. The file is kept unchanged until this recording is deleted."
        const val NO_SPEECH_ON_RETRY = "This attempt recognized no speech; the earlier result is kept."
        const val AUDIO_REMOVED = "The audio was removed by your retention policy; the transcript is kept."
        const val INTERRUPTED_TRANSCRIPTION =
            "Transcription was interrupted when the app stopped. The audio is saved; retry it."
    }
}
