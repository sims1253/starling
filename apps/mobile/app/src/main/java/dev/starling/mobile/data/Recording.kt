package dev.starling.mobile.data

/**
 * A recording is kept on disk until the user explicitly deletes it; only
 * a retention limit the user set can remove its audio (#342), and then
 * the recording and its transcripts stay. The raw transcript returned by
 * Starling is stored verbatim in [rawTranscript].
 */
enum class RecordingStatus {
    RECORDING,
    PENDING,
    TRANSCRIBING,
    TRANSCRIBED,
    FAILED,
}

/**
 * How the stored transcript was produced. Provenance only records what
 * happened; it never changes the text itself.
 */
enum class TranscriptionProvenance {
    /** Live `WS /stream` session: partials while recording, final on commit. */
    LIVE_STREAM,

    /** Multipart upload of the saved WAV after the recording finished. */
    BATCH_UPLOAD,
}

/** Where a transcript was computed. */
enum class TranscriptSource {
    ON_DEVICE,
    SERVER,
}

/**
 * One successful transcription of a recording's audio. A retry never
 * replaces an earlier revision: it appends a new one.
 */
data class TranscriptRevision(
    val text: String,
    val provenance: TranscriptionProvenance,
    val source: TranscriptSource?,
    /** The on-device model file, or the server model name; null when unknown. */
    val model: String?,
    val createdAtMillis: Long,
)

/**
 * The audio of a capture that did not end with a clean Stop (the process
 * died, the microphone failed, the disk filled up) and was recovered from
 * the partial WAV. [recoveredSeconds] is every sample found in the file;
 * [confirmedSeconds] the part the capture had confirmed on storage (fsync)
 * before it ended. Process death keeps everything the capture wrote; only
 * a power loss or OS crash can lose the unconfirmed tail.
 */
data class CaptureRecovery(
    val reason: String,
    val recoveredSeconds: Double,
    val confirmedSeconds: Double,
)

/**
 * Which retention limits govern a take's audio (#342). Archival takes
 * follow their own limits; they are still kept losslessly (no lossy
 * archival format yet).
 */
enum class RetentionClass(val key: String) {
    STANDARD("standard"),
    ARCHIVAL("archival"),
}

/** Which retention limit made a take's audio due. */
enum class RetireReason { AGE, SIZE }

/**
 * The retention policy removed this take's audio (#342); the take, its
 * transcripts and revisions stay.
 */
data class AudioRemoval(val atMillis: Long, val reason: RetireReason)

data class Recording(
    val id: String,
    val createdAtMillis: Long,
    val wavName: String,
    val status: RecordingStatus,
    val durationSeconds: Double = 0.0,
    /** The latest successful transcript (the last of [revisions]). */
    val rawTranscript: String? = null,
    val errorMessage: String? = null,
    val attempts: Int = 0,
    val provenance: TranscriptionProvenance? = null,
    /**
     * A take dictated into a private field (password, incognito). It never
     * appears in the history, and the keyboard deletes it as soon as the
     * take settles; any left behind by a crash are deleted on the next start.
     */
    val ephemeral: Boolean = false,
    /** Every successful transcript, oldest first. */
    val revisions: List<TranscriptRevision> = emptyList(),
    /** Set when the audio was recovered from an interrupted capture. */
    val recovery: CaptureRecovery? = null,
    val retentionClass: RetentionClass = RetentionClass.STANDARD,
    /** Set once the retention policy removed the audio. */
    val audioRemoved: AudioRemoval? = null,
)
