package dev.starling.mobile.storage

import android.content.Context
import android.content.SharedPreferences
import dev.starling.mobile.data.RetentionClass

/**
 * History-audio retention limits (#342), the desktop's
 * `storage.{standard,archival}.{maxAgeDays,maxTotalMb}` under the same key
 * names. Everything is off by default; finished takes are kept as
 * lossless FLAC regardless, which is not a setting.
 */
class StorageSettings internal constructor(private val preferences: SharedPreferences) : PolicyGate {
    constructor(context: Context) : this(
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE),
    )

    private val lock = Any()

    fun load(): RetentionPolicy = synchronized(lock) {
        RetentionPolicy(
            RetentionClass.entries.associateWith { retentionClass ->
                ClassLimits(
                    maxAgeDays = positive(ageKey(retentionClass))?.toInt(),
                    maxTotalMb = positive(sizeKey(retentionClass)),
                )
            },
        )
    }

    /** Saves [limits] for [retentionClass]; returns the policy now in force. */
    fun save(retentionClass: RetentionClass, limits: ClassLimits): RetentionPolicy = synchronized(lock) {
        val editor = preferences.edit()
        limits.maxAgeDays?.let { editor.putLong(ageKey(retentionClass), it.toLong()) } ?: editor.remove(ageKey(retentionClass))
        limits.maxTotalMb?.let { editor.putLong(sizeKey(retentionClass), it) } ?: editor.remove(sizeKey(retentionClass))
        // commit, not apply: the next retention check must read what was saved.
        editor.commit()
        load()
    }

    override fun <T> withPolicy(block: (RetentionPolicy) -> T): T = synchronized(lock) { block(load()) }

    /** A stored limit; anything unreadable or not positive is no limit. */
    private fun positive(key: String): Long? =
        runCatching { preferences.getLong(key, 0L) }.getOrNull()?.takeIf { it > 0 }

    companion object {
        private const val PREFS_NAME = "storage_settings"

        fun ageKey(retentionClass: RetentionClass) = "storage.${retentionClass.key}.maxAgeDays"

        fun sizeKey(retentionClass: RetentionClass) = "storage.${retentionClass.key}.maxTotalMb"
    }
}
