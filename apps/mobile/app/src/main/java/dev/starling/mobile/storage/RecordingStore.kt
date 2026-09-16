package dev.starling.mobile.storage

import android.content.Context
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
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
 */
class RecordingStore(private val directory: File) {
    constructor(context: Context) : this(
        File(context.applicationContext.filesDir, "recordings"),
    )

    private val lock = Any()

    init {
        if (!directory.exists() && !directory.mkdirs()) {
            throw IOException("Unable to create private recording directory")
        }
        // A crash between the temporary write and the rename in save() would
        // leave an unnamed .<id>.json.tmp behind forever: list() filters on
        // the .json suffix and delete() never looks for it. They are write
        // temporaries with no committed state, so sweep them on open.
        directory
            .listFiles { file -> file.isFile && file.name.startsWith(".") && file.name.endsWith(".json.tmp") }
            ?.forEach { temporary -> temporary.delete() }
    }

    fun create(): Recording = synchronized(lock) {
        val id = UUID.randomUUID().toString()
        val recording = Recording(
            id = id,
            createdAtMillis = System.currentTimeMillis(),
            wavName = "$id.wav",
            status = RecordingStatus.RECORDING,
        )
        save(recording)
        recording
    }

    fun list(): List<Recording> = synchronized(lock) {
        directory.listFiles { file -> file.isFile && file.name.endsWith(".json") }
            .orEmpty()
            .mapNotNull { file ->
                runCatching {
                    val decoded = decode(file)
                    recoverFinalizedAudio(decoded).also { recovered ->
                        if (recovered != decoded) runCatching { save(recovered) }
                    }
                }.getOrNull()
            }
            .sortedByDescending { it.createdAtMillis }
    }

    fun get(id: String): Recording = synchronized(lock) {
        requireValidId(id)
        val file = metadataFile(id)
        if (!file.isFile) throw IOException("Recording is no longer available")
        val decoded = decode(file)
        return recoverFinalizedAudio(decoded).also { recovered ->
            if (recovered != decoded) runCatching { save(recovered) }
        }
    }

    fun partialFile(recording: Recording): File = File(directory, "${recording.id}.wav.part")

    fun audioFile(recording: Recording): File = File(directory, recording.wavName)

    /** Atomically promotes the completed WAV from its partial file. */
    fun commitAudio(recording: Recording, durationSeconds: Double): Recording = synchronized(lock) {
        val partial = partialFile(recording)
        if (!isFinalizedWav(partial)) {
            throw IOException("Recording did not produce a complete WAV")
        }
        val destination = audioFile(recording)
        if (destination.exists() && !destination.delete()) {
            throw IOException("Unable to replace the recording audio")
        }
        if (!partial.renameTo(destination)) {
            throw IOException("Unable to finalize the recording audio")
        }
        val updated = recording.copy(
            status = RecordingStatus.PENDING,
            durationSeconds = durationSeconds,
            errorMessage = null,
        )
        save(updated)
        updated
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

    fun markTranscribed(id: String, rawTranscript: String): Recording = synchronized(lock) {
        // Deliberately do not trim, normalize, or otherwise clean this value.
        update(id) {
            it.copy(
                status = RecordingStatus.TRANSCRIBED,
                rawTranscript = rawTranscript,
                errorMessage = null,
            )
        }
    }

    /** Deletion is only called by an explicit user action. */
    fun delete(id: String) = synchronized(lock) {
        requireValidId(id)
        val metadata = metadataFile(id)
        val audio = File(directory, "$id.wav")
        val partial = File(directory, "$id.wav.part")
        // A crash between save()'s write and rename can orphan the metadata
        // temporary; the init sweep misses files created after this store
        // opened, so try to take it with the recording as well.
        val metadataTemporary = File(directory, ".$id.json.tmp")
        val failures = listOf(metadata, audio, partial, metadataTemporary)
            .filter { it.exists() && !it.delete() }
        if (failures.isNotEmpty()) {
            throw IOException("Unable to delete recording files")
        }
    }

    private fun update(id: String, transform: (Recording) -> Recording): Recording {
        val current = get(id)
        return transform(current).also(::save)
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

        FileOutputStream(temporary).use { output ->
            output.write(json.toString().toByteArray(Charsets.UTF_8))
            output.fd.sync()
        }
        if (!temporary.renameTo(target)) {
            throw IOException("Unable to commit recording metadata")
        }
    }

    private fun decode(file: File): Recording {
        val json = JSONObject(file.readText(Charsets.UTF_8))
        val id = json.getString("id")
        requireValidId(id)
        val status = runCatching {
            RecordingStatus.valueOf(json.getString("status"))
        }.getOrElse { RecordingStatus.FAILED }
        val wavName = json.getString("wav_name")
        require(wavName == "$id.wav") { "Invalid recording audio name" }
        return Recording(
            id = id,
            createdAtMillis = json.getLong("created_at_ms"),
            wavName = wavName,
            status = status,
            durationSeconds = json.optDouble("duration_s", 0.0),
            rawTranscript = json.optionalString("raw_transcript"),
            errorMessage = json.optionalString("error_message"),
            attempts = json.optInt("attempts", 0),
        )
    }

    /**
     * Recover the narrow crash window between audio promotion and metadata
     * commit. A process can leave a finalized WAV beside RECORDING metadata;
     * make it PENDING so the UI exposes Retry on the next read. A finalized
     * partial WAV is also promoted after a process exit before commitAudio.
     */
    private fun recoverFinalizedAudio(recording: Recording): Recording {
        if (recording.status != RecordingStatus.RECORDING) return recording
        val destination = audioFile(recording)
        if (isFinalizedWav(destination)) {
            return recording.copy(status = RecordingStatus.PENDING, errorMessage = null)
        }
        val partial = partialFile(recording)
        if (!isFinalizedWav(partial)) return recording
        if (destination.exists() && !destination.delete()) return recording
        if (!partial.renameTo(destination)) return recording
        return recording.copy(status = RecordingStatus.PENDING, errorMessage = null)
    }

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

    private fun requireValidId(id: String) {
        require(UUID_PATTERN.matches(id)) { "Invalid recording id" }
    }

    companion object {
        private const val WAV_HEADER_BYTES = 44L
        private val UUID_PATTERN = Regex("[0-9a-fA-F-]{36}")
    }
}
