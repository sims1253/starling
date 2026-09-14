package dev.starling.mobile.data

/**
 * A recording is kept on disk until the user explicitly deletes it. The raw
 * transcript returned by Starling is stored verbatim in [rawTranscript].
 */
enum class RecordingStatus {
    RECORDING,
    PENDING,
    TRANSCRIBING,
    TRANSCRIBED,
    FAILED,
}

data class Recording(
    val id: String,
    val createdAtMillis: Long,
    val wavName: String,
    val status: RecordingStatus,
    val durationSeconds: Double = 0.0,
    val rawTranscript: String? = null,
    val errorMessage: String? = null,
    val attempts: Int = 0,
)
