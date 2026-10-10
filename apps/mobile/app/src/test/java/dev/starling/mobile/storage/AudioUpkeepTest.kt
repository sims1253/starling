package dev.starling.mobile.storage

import android.content.SharedPreferences
import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RetentionClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

/** #342: the upkeep pass and the storage settings behind it. */
class AudioUpkeepTest {
    @get:Rule
    val folder = TemporaryFolder()

    private val day = 24L * 60 * 60 * 1000
    private var now = 1_000 * day

    private fun store() = RecordingStore(File(folder.root, "recordings")) { now }

    private fun take(store: RecordingStore, ageDays: Double): Recording {
        val recordedAt = now
        now -= (ageDays * day).toLong()
        val recording = store.create()
        now = recordedAt
        WavWriter(store.partialFile(recording)).apply {
            write(ByteArray(64_000), 64_000)
            finish()
        }
        store.commitAudio(recording, 2.0)
        return store.markTranscribed(recording.id, "text")
    }

    private val policyOff = object : PolicyGate {
        override fun <T> withPolicy(block: (RetentionPolicy) -> T): T = block(RetentionPolicy())
    }

    @Test
    fun aPassCompressesFinishedTakesThenAppliesThePolicy() {
        val store = store()
        val old = take(store, ageDays = 40.0)
        val young = take(store, ageDays = 2.0)
        val settings = StorageSettings(FakePreferences())
        settings.save(RetentionClass.STANDARD, ClassLimits(maxAgeDays = 30))
        val upkeep = AudioUpkeep(store, settings, Runnable::run, recording = { false })

        val report = upkeep.runPass()

        assertEquals(2, report.compressed)
        assertEquals(listOf(old.id), report.retention.removed.map { it.id })
        assertTrue(File(folder.root, "recordings/${young.id}.flac").isFile)
        assertEquals(report, upkeep.lastReport)
    }

    @Test
    fun noPassRunsWhileATakeRecordsAndARunningOneStops() {
        val store = store()
        take(store, ageDays = 40.0)
        take(store, ageDays = 50.0)
        val idle = AudioUpkeep(store, policyOff, Runnable::run, recording = { true })
        assertEquals(AudioUpkeep.Report(paused = true), idle.runPass())
        assertEquals(2, store.compressionCandidates().size)
        assertNull(idle.lastReport)

        // A take starts after the first compression.
        var asked = 0
        val interrupted = AudioUpkeep(store, policyOff, Runnable::run, recording = { ++asked > 2 })
        val report = interrupted.runPass()
        assertEquals(1, report.compressed)
        assertTrue(report.paused)
        assertEquals(1, store.compressionCandidates().size)
    }

    @Test
    fun aTakeThatFailsToCompressThreeTimesIsLeftAlone() {
        val store = store()
        val take = take(store, ageDays = 3.0)
        store.compressionHook = { throw IllegalStateException("encoder broke") }
        val upkeep = AudioUpkeep(store, policyOff, Runnable::run, recording = { false })

        repeat(AudioUpkeep.MAX_COMPRESSION_ATTEMPTS) { assertEquals(1, upkeep.runPass().failures) }
        assertEquals(0, upkeep.runPass().failures)
        assertTrue(File(folder.root, "recordings/${take.id}.wav").isFile)
        assertEquals(listOf(take.id), store.compressionCandidates())
    }

    @Test
    fun aChangedPolicyRunsAgain() {
        val store = store()
        take(store, ageDays = 40.0)
        take(store, ageDays = 50.0)
        var policy = RetentionPolicy(mapOf(RetentionClass.STANDARD to ClassLimits(maxAgeDays = 30)))
        var reads = 0
        val gate = object : PolicyGate {
            override fun <T> withPolicy(block: (RetentionPolicy) -> T): T {
                if (++reads == 3) policy = RetentionPolicy(mapOf(RetentionClass.STANDARD to ClassLimits(maxAgeDays = 45)))
                return block(policy)
            }
        }
        val passes = mutableListOf<Runnable>()
        val upkeep = AudioUpkeep(store, gate, { passes += it }, recording = { false })

        val first = upkeep.runPass()

        assertTrue(first.retention.policyChanged)
        assertEquals(1, first.retention.removed.size)
        assertEquals(1, passes.size)
    }

    @Test
    fun storageSettingsAreOffByDefaultAndUseTheDesktopKeys() {
        val preferences = FakePreferences()
        val settings = StorageSettings(preferences)
        assertFalse(settings.load().isActive)

        settings.save(RetentionClass.ARCHIVAL, ClassLimits(maxAgeDays = 365, maxTotalMb = 5120))
        settings.save(RetentionClass.STANDARD, ClassLimits(maxTotalMb = 1024))

        assertEquals(
            mapOf(
                "storage.archival.maxAgeDays" to 365L,
                "storage.archival.maxTotalMb" to 5120L,
                "storage.standard.maxTotalMb" to 1024L,
            ),
            preferences.values,
        )
        val policy = settings.load()
        assertEquals(ClassLimits(maxTotalMb = 1024), policy.limits(RetentionClass.STANDARD))
        assertEquals(1024L * 1024 * 1024, policy.limits(RetentionClass.STANDARD).maxTotalBytes)
        assertEquals(ClassLimits(maxAgeDays = 365, maxTotalMb = 5120), policy.limits(RetentionClass.ARCHIVAL))
        settings.save(RetentionClass.ARCHIVAL, ClassLimits())
        assertEquals(setOf("storage.standard.maxTotalMb"), preferences.values.keys)
        // Anything unreadable is no limit.
        preferences.values["storage.standard.maxAgeDays"] = -3L
        assertEquals(null, settings.load().limits(RetentionClass.STANDARD).maxAgeDays)
    }

    @Test
    fun aSaveWaitsForARemovalAlreadyDecided() {
        val settings = StorageSettings(FakePreferences())
        settings.save(RetentionClass.STANDARD, ClassLimits(maxAgeDays = 30))
        val inside = CountDownLatch(1)
        val release = CountDownLatch(1)
        val saved = CountDownLatch(1)
        val remover = thread {
            settings.withPolicy {
                inside.countDown()
                release.await(5, TimeUnit.SECONDS)
            }
        }
        inside.await(5, TimeUnit.SECONDS)
        thread {
            settings.save(RetentionClass.STANDARD, ClassLimits())
            saved.countDown()
        }
        assertFalse("the save went through mid-removal", saved.await(200, TimeUnit.MILLISECONDS))
        release.countDown()
        assertTrue(saved.await(5, TimeUnit.SECONDS))
        remover.join()
        assertFalse(settings.load().isActive)
    }

    /** Just enough SharedPreferences for StorageSettings (longs only). */
    private class FakePreferences : SharedPreferences {
        val values = LinkedHashMap<String, Long>()

        override fun getLong(key: String, defValue: Long): Long = values[key] ?: defValue
        override fun contains(key: String): Boolean = key in values
        override fun edit(): SharedPreferences.Editor = Editor()
        override fun getAll(): MutableMap<String, *> = values
        override fun getString(key: String?, defValue: String?): String? = throw UnsupportedOperationException()
        override fun getStringSet(key: String?, defValues: MutableSet<String>?): MutableSet<String> = throw UnsupportedOperationException()
        override fun getInt(key: String?, defValue: Int): Int = throw UnsupportedOperationException()
        override fun getFloat(key: String?, defValue: Float): Float = throw UnsupportedOperationException()
        override fun getBoolean(key: String?, defValue: Boolean): Boolean = throw UnsupportedOperationException()
        override fun registerOnSharedPreferenceChangeListener(listener: SharedPreferences.OnSharedPreferenceChangeListener?) = Unit
        override fun unregisterOnSharedPreferenceChangeListener(listener: SharedPreferences.OnSharedPreferenceChangeListener?) = Unit

        private inner class Editor : SharedPreferences.Editor {
            private val puts = LinkedHashMap<String, Long>()
            private val removes = mutableSetOf<String>()

            override fun putLong(key: String, value: Long) = apply { puts[key] = value }
            override fun remove(key: String) = apply { removes += key }
            override fun commit(): Boolean {
                removes.forEach(values::remove)
                values.putAll(puts)
                return true
            }
            override fun apply() {
                commit()
            }
            override fun putString(key: String?, value: String?) = throw UnsupportedOperationException()
            override fun putStringSet(key: String?, values: MutableSet<String>?) = throw UnsupportedOperationException()
            override fun putInt(key: String?, value: Int) = throw UnsupportedOperationException()
            override fun putFloat(key: String?, value: Float) = throw UnsupportedOperationException()
            override fun putBoolean(key: String?, value: Boolean) = throw UnsupportedOperationException()
            override fun clear() = throw UnsupportedOperationException()
        }
    }
}
