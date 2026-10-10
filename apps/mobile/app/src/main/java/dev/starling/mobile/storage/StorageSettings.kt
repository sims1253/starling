package dev.starling.mobile.storage

import android.content.Context
import android.content.SharedPreferences
import dev.starling.mobile.data.RetentionClass
import java.io.IOException

/**
 * History-audio retention limits (#342), the desktop's
 * `storage.{standard,archival}.{maxAgeDays,maxTotalMb}` under the same key
 * names. Everything is off by default; finished takes are kept as
 * lossless FLAC regardless, which is not a setting.
 *
 * The policy in force is the last one that reached storage: a save whose
 * commit fails throws and changes nothing (SharedPreferences has already
 * taken the new values in memory then, so they are put back).
 */
class StorageSettings internal constructor(private val preferences: SharedPreferences) : PolicyGate {
    constructor(context: Context) : this(
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE),
    )

    private val lock = Any()

    // Guarded by [lock]: the last policy that reached storage.
    private var saved: RetentionPolicy = read()

    fun load(): RetentionPolicy = synchronized(lock) { saved }

    /**
     * Saves [limits] for [retentionClass]; returns the policy now in force.
     * Throws when storage refuses the save, and the earlier policy stays.
     */
    fun save(retentionClass: RetentionClass, limits: ClassLimits): RetentionPolicy = synchronized(lock) {
        val previous = saved.limits(retentionClass)
        // commit, not apply: the next retention check must see what was saved.
        if (!write(retentionClass, limits).commit()) {
            write(retentionClass, previous).commit()
            throw IOException("Unable to save the history audio limits")
        }
        saved = read()
        saved
    }

    override fun <T> withPolicy(block: (RetentionPolicy) -> T): T = synchronized(lock) { block(saved) }

    private fun write(retentionClass: RetentionClass, limits: ClassLimits): SharedPreferences.Editor {
        val editor = preferences.edit()
        limits.maxAgeDays?.let { editor.putLong(ageKey(retentionClass), it.toLong()) } ?: editor.remove(ageKey(retentionClass))
        limits.maxTotalMb?.let { editor.putLong(sizeKey(retentionClass), it) } ?: editor.remove(sizeKey(retentionClass))
        return editor
    }

    private fun read(): RetentionPolicy = RetentionPolicy(
        RetentionClass.entries.associateWith { retentionClass ->
            ClassLimits(
                maxAgeDays = stored(ageKey(retentionClass), ClassLimits.MAX_AGE_DAYS.toLong())?.toInt(),
                maxTotalMb = stored(sizeKey(retentionClass), ClassLimits.MAX_TOTAL_MB),
            )
        },
    )

    /**
     * A stored limit; anything unreadable, not positive or past [max] is no
     * limit (a huge value must never wrap into a tiny one).
     */
    private fun stored(key: String, max: Long): Long? =
        runCatching { preferences.getLong(key, 0L) }.getOrNull()?.takeIf { it in 1..max }

    companion object {
        private const val PREFS_NAME = "storage_settings"

        fun ageKey(retentionClass: RetentionClass) = "storage.${retentionClass.key}.maxAgeDays"

        fun sizeKey(retentionClass: RetentionClass) = "storage.${retentionClass.key}.maxTotalMb"
    }
}
