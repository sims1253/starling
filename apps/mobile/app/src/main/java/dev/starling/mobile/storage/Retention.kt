package dev.starling.mobile.storage

import dev.starling.mobile.data.RetentionClass
import dev.starling.mobile.data.RetireReason

/**
 * Age and size limits for one retention class (#342); null is no limit,
 * both null leaves the class alone. Sizes are MiB, like the desktop's
 * `storage.<class>.maxTotalMb`.
 */
data class ClassLimits(
    val maxAgeDays: Int? = null,
    val maxTotalMb: Long? = null,
) {
    init {
        require(maxAgeDays == null || maxAgeDays in 0..MAX_AGE_DAYS) { "Age limit out of range" }
        require(maxTotalMb == null || maxTotalMb in 0..MAX_TOTAL_MB) { "Size limit out of range" }
    }

    val isActive: Boolean get() = maxAgeDays != null || maxTotalMb != null
    val maxTotalBytes: Long? get() = maxTotalMb?.let { it * 1024 * 1024 }

    companion object {
        /** A thousand years; the age arithmetic stays far from overflow. */
        const val MAX_AGE_DAYS = 365_000

        /** The largest MiB count whose byte count fits a Long. */
        const val MAX_TOTAL_MB = Long.MAX_VALUE / (1024 * 1024)
    }
}

/**
 * The user's retention policy (#342). The default is off: no class has a
 * limit and [RecordingStore.applyRetention] removes nothing.
 */
data class RetentionPolicy(
    val limits: Map<RetentionClass, ClassLimits> = emptyMap(),
    /** No take younger than this loses its audio. */
    val graceMillis: Long = DEFAULT_GRACE_MILLIS,
) {
    val isActive: Boolean get() = limits.values.any { it.isActive }

    fun limits(retentionClass: RetentionClass): ClassLimits = limits[retentionClass] ?: ClassLimits()

    companion object {
        /**
         * A day: a take whose transcription just failed always survives
         * long enough to retry (#356).
         */
        const val DEFAULT_GRACE_MILLIS = 24L * 60 * 60 * 1000
    }
}

/**
 * The live policy, as the retention run sees it before every removal.
 * [withPolicy] runs its block with the current policy and holds whatever
 * keeps it current (StorageSettings' lock) until the block returns, so a
 * save that lifts a limit waits for a removal already decided, and the
 * next check sees it.
 */
interface PolicyGate {
    fun <T> withPolicy(block: (RetentionPolicy) -> T): T
}

/** Why a due take keeps its audio. */
enum class HoldReason {
    /** Younger than the policy's grace. */
    RECENT,

    /** A transcription or retry is using it. */
    IN_USE,

    /** No attempt ever produced a transcript: the audio is all there is. */
    UNTRANSCRIBED,
}

data class RemovedAudio(val id: String, val retentionClass: RetentionClass, val bytes: Long, val reason: RetireReason)

data class HeldAudio(val id: String, val retentionClass: RetentionClass, val bytes: Long, val reason: HoldReason)

/** What one retention run removed and held. */
data class RetentionReport(
    val removed: List<RemovedAudio> = emptyList(),
    /** Due takes that kept their audio, with the reason. */
    val held: List<HeldAudio> = emptyList(),
    /** Bytes per class a size limit is still exceeded by (held takes count). */
    val overLimit: Map<RetentionClass, Long> = emptyMap(),
    /** The policy changed during the run; it stopped before the next removal. */
    val policyChanged: Boolean = false,
    /** The caller's stop condition (a take started recording) ended the run. */
    val stopped: Boolean = false,
    /** Due takes whose audio storage refused to unlink: they keep it, unstamped, for a later run. */
    val failed: List<RemovedAudio> = emptyList(),
) {
    val removedBytes: Long get() = removed.sumOf { it.bytes }
}
