package dev.starling.mobile.storage

import android.content.Context
import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.CaptureRecovery
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.data.TranscriptRevision
import dev.starling.mobile.data.TranscriptSource
import dev.starling.mobile.data.TranscriptionProvenance
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.RandomAccessFile
import java.util.UUID

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
 */
class RecordingStore internal constructor(
    private val directory: File,
    private val move: (File, File) -> Unit = Durability::replace,
    private val clock: () -> Long = System::currentTimeMillis,
) {
    constructor(context: Context) : this(File(context.applicationContext.filesDir, "recordings"))

    private val lock = Any()

    init {
        if (!directory.exists() && !directory.mkdirs()) {
            throw IOException("Unable to create private recording directory")
        }
        sweepOrphanedTemporaries()
        sweepEphemeral()
        recoverInterrupted()
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
        directory.listFiles { file -> file.isFile && file.name.endsWith(".json") }
            .orEmpty()
            .mapNotNull { file -> runCatching { decode(file) }.getOrNull() }
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

    fun audioFile(recording: Recording): File = File(directory, recording.wavName)

    /** Atomically promotes the completed WAV from its partial file. */
    fun commitAudio(recording: Recording, durationSeconds: Double): Recording = synchronized(lock) {
        val partial = partialFile(recording)
        if (!isFinalizedWav(partial)) {
            throw IOException("Recording did not produce a complete WAV")
        }
        move(partial, audioFile(recording))
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
        val audio = File(directory, "$id.wav")
        val partial = File(directory, "$id.wav.part")
        val unrecognized = File(directory, "$id.wav.unrecognized")
        val temporary = File(directory, ".$id.json.tmp")
        // Metadata goes last: until every payload file is gone it stays,
        // so an interrupted or failed delete (of an ephemeral take, too) is
        // found and finished again on the next open.
        val failures = listOf(audio, partial, unrecognized, temporary)
            .filter { it.exists() && !it.delete() }
        if (failures.isNotEmpty() || (metadata.exists() && !metadata.delete())) {
            throw IOException("Unable to delete recording files")
        }
    }

    private fun update(id: String, transform: (Recording) -> Recording): Recording {
        val current = get(id)
        return transform(current).also(::save)
    }

    /**
     * Remove metadata temporaries leaked by a crash between the write and the
     * rename in [save]. Nothing ever reads them back (list() only considers
     * .json files), so deleting them on open is safe and keeps the directory
     * from growing without bound on crash-prone devices.
     */
    private fun sweepOrphanedTemporaries() {
        directory.listFiles { file ->
            file.isFile && file.name.startsWith(".") && file.name.endsWith(".json.tmp")
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
        )
    }

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
        val destination = audioFile(recording)
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
        private val UUID_PATTERN = Regex("[0-9a-fA-F-]{36}")

        const val INTERRUPTED_CAPTURE = "The recording was interrupted: the app or the phone stopped while it was recording."
        const val UNRECOVERED_CAPTURE =
            "The recording's audio could not be recovered yet; Starling tries again the next time it starts."
        const val UNRECOGNIZED_CAPTURE =
            "The recording did not finish cleanly, and its partial audio is not a WAV Starling recognizes, " +
                "so no audio could be recovered. The file is kept unchanged until this recording is deleted."
        const val NO_SPEECH_ON_RETRY = "This attempt recognized no speech; the earlier result is kept."
        const val INTERRUPTED_TRANSCRIPTION =
            "Transcription was interrupted when the app stopped. The audio is saved; retry it."
    }
}
