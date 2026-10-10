package dev.starling.mobile.storage

import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.data.RetentionClass
import dev.starling.mobile.data.RetireReason
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.IOException

/** #342: retention removes only audio, only what a limit requires, and never what is held. */
class RetentionTest {
    @get:Rule
    val folder = TemporaryFolder()

    private val day = 24L * 60 * 60 * 1000
    private var now = 1_000 * day

    private fun storeDir(): File = File(folder.root, "recordings")

    private fun store() = RecordingStore(storeDir()) { now }

    /** A fixed policy; [before] runs inside the gate, before every removal decision. */
    private class Gate(var policy: RetentionPolicy, var before: () -> Unit = {}) : PolicyGate {
        override fun <T> withPolicy(block: (RetentionPolicy) -> T): T {
            before()
            return block(policy)
        }
    }

    private fun policy(
        standard: ClassLimits = ClassLimits(),
        archival: ClassLimits = ClassLimits(),
    ) = RetentionPolicy(mapOf(RetentionClass.STANDARD to standard, RetentionClass.ARCHIVAL to archival))

    private val mib = 1024 * 1024

    /** A take recorded [ageDays] ago holding [bytes] of PCM, transcribed unless said otherwise. */
    private fun take(store: RecordingStore, ageDays: Double, bytes: Int = mib, transcribed: Boolean = true): Recording {
        val recordedAt = now
        now -= (ageDays * day).toLong()
        val recording = store.create()
        now = recordedAt
        WavWriter(store.partialFile(recording)).apply {
            write(ByteArray(bytes), bytes)
            finish()
        }
        store.commitAudio(recording, bytes / 32_000.0)
        return if (transcribed) store.markTranscribed(recording.id, "kept text") else store.get(recording.id)
    }

    private fun hasAudio(recording: Recording) =
        File(storeDir(), "${recording.id}.wav").exists() || File(storeDir(), "${recording.id}.flac").exists()

    @Test
    fun nothingIsRemovedByDefault() {
        val store = store()
        val old = take(store, ageDays = 3_000.0)
        val report = store.applyRetention(Gate(RetentionPolicy()))
        assertEquals(RetentionReport(), report)
        assertTrue(hasAudio(old))
    }

    @Test
    fun anAgeLimitRemovesOnlyTheAudioOfOldTakes() {
        val store = store()
        val old = take(store, ageDays = 40.0)
        val young = take(store, ageDays = 10.0)

        val report = store.applyRetention(Gate(policy(standard = ClassLimits(maxAgeDays = 30))))

        assertEquals(listOf(RemovedAudio(old.id, RetentionClass.STANDARD, 44L + mib, RetireReason.AGE)), report.removed)
        assertFalse(hasAudio(old))
        assertTrue(hasAudio(young))
        // The take, its transcript and revisions stay, marked as removed.
        val kept = store.get(old.id)
        assertEquals(RecordingStatus.TRANSCRIBED, kept.status)
        assertEquals("kept text", kept.rawTranscript)
        assertEquals(1, kept.revisions.size)
        assertEquals(now, kept.audioRemoved!!.atMillis)
        assertEquals(RetireReason.AGE, kept.audioRemoved!!.reason)
        assertTrue(store.list().any { it.id == old.id })
        try {
            store.withRequestAudio(old.id) { fail("removed audio was read") }
        } catch (exception: IOException) {
            assertEquals(RecordingStore.AUDIO_REMOVED, exception.message)
        }
        // A second run has nothing left to do.
        assertEquals(emptyList<RemovedAudio>(), store.applyRetention(Gate(policy(standard = ClassLimits(maxAgeDays = 30)))).removed)
    }

    @Test
    fun recentInUseAndUntranscribedTakesAreHeldAndReported() {
        val store = store()
        val recent = take(store, ageDays = 0.5)
        val untranscribed = take(store, ageDays = 50.0, transcribed = false)
        val pinned = take(store, ageDays = 60.0)
        val transcribing = take(store, ageDays = 70.0).also { store.markTranscribing(it.id) }
        val failedAfterTranscript = take(store, ageDays = 80.0).also { store.markFailed(it.id, "retry failed") }
        val pin = store.pin(pinned.id)

        // Everything is over a zero-day age limit; the grace still protects a day.
        val report = store.applyRetention(Gate(policy(standard = ClassLimits(maxAgeDays = 0))))

        assertEquals(listOf(failedAfterTranscript.id), report.removed.map { it.id })
        assertEquals(
            mapOf(
                recent.id to HoldReason.RECENT,
                untranscribed.id to HoldReason.UNTRANSCRIBED,
                pinned.id to HoldReason.IN_USE,
                transcribing.id to HoldReason.IN_USE,
            ),
            report.held.associate { it.id to it.reason },
        )
        listOf(recent, untranscribed, pinned, transcribing).forEach { assertTrue(hasAudio(it)) }
        pin.close()
        assertEquals(listOf(pinned.id), store.applyRetention(Gate(policy(standard = ClassLimits(maxAgeDays = 0)))).removed.map { it.id })
    }

    @Test
    fun aSizeLimitKeepsTheNewestAndCountsHeldTakes() {
        val store = store()
        val oldest = take(store, ageDays = 30.0)
        val untranscribed = take(store, ageDays = 20.0, transcribed = false)
        val newer = take(store, ageDays = 10.0)
        val newest = take(store, ageDays = 5.0)

        // Room for the two newest (each 1 MiB plus its 44-byte header).
        val report = store.applyRetention(Gate(policy(standard = ClassLimits(maxTotalMb = 3))))

        // The untranscribed take is due and held, but it still counts, so
        // the oldest goes; the class stays over by what the hold keeps.
        assertEquals(listOf(RemovedAudio(oldest.id, RetentionClass.STANDARD, 44L + mib, RetireReason.SIZE)), report.removed)
        assertEquals(listOf(HeldAudio(untranscribed.id, RetentionClass.STANDARD, 44L + mib, HoldReason.UNTRANSCRIBED)), report.held)
        assertEquals(listOf(true, true, true, false), listOf(newest, newer, untranscribed, oldest).map(::hasAudio))
        assertEquals(mapOf(RetentionClass.STANDARD to 3 * 44L), report.overLimit)
    }

    @Test
    fun aSizeLimitRemovesOldestFirstUntilTheClassFits() {
        val store = store()
        val takes = (5 downTo 1).map { take(store, ageDays = it * 10.0) } // oldest first

        val report = store.applyRetention(Gate(policy(standard = ClassLimits(maxTotalMb = 3))))

        // Three takes are 3 MiB plus three headers: only two fit.
        assertEquals(takes.take(3).map { it.id }.toSet(), report.removed.map { it.id }.toSet())
        assertEquals(listOf(false, false, false, true, true), takes.map(::hasAudio))
        assertEquals(emptyMap<RetentionClass, Long>(), report.overLimit)
        assertTrue(report.removed.all { it.reason == RetireReason.SIZE })
    }

    @Test
    fun eachClassFollowsItsOwnLimits() {
        val store = store()
        val standard = take(store, ageDays = 40.0)
        val archived = take(store, ageDays = 40.0).let { store.setRetentionClass(it.id, RetentionClass.ARCHIVAL) }

        val standardOnly = store.applyRetention(Gate(policy(standard = ClassLimits(maxAgeDays = 30))))
        assertEquals(listOf(standard.id), standardOnly.removed.map { it.id })
        assertTrue(hasAudio(archived))

        val archivalToo = store.applyRetention(Gate(policy(archival = ClassLimits(maxAgeDays = 30))))
        assertEquals(listOf(RemovedAudio(archived.id, RetentionClass.ARCHIVAL, 44L + mib, RetireReason.AGE)), archivalToo.removed)
        assertEquals(RetentionClass.ARCHIVAL, RecordingStore(storeDir()) { now }.get(archived.id).retentionClass)
    }

    @Test
    fun aTakeMovedToAnotherClassDuringTheRunIsJudgedByThatClass() {
        val store = store()
        val moved = take(store, ageDays = 40.0)
        val stays = take(store, ageDays = 45.0)
        var calls = 0
        // After the run read its policy, before its first removal.
        val gate = Gate(policy(standard = ClassLimits(maxAgeDays = 30))) {
            if (++calls == 2) store.setRetentionClass(moved.id, RetentionClass.ARCHIVAL)
        }

        val report = store.applyRetention(gate)

        assertEquals(listOf(stays.id), report.removed.map { it.id })
        assertTrue(hasAudio(moved))
    }

    @Test
    fun sizesAreMeasuredAgainAtTheRemoval() {
        val store = store()
        // Compressible audio: a quiet tone compresses far below its WAV.
        val quiet = ByteArray(2 * mib) { if (it % 2 == 0) (it / 2 % 7).toByte() else 0 }
        val older = take(store, ageDays = 20.0)
        val newer = store.create().also { recording ->
            WavWriter(store.partialFile(recording)).apply { write(quiet, quiet.size); finish() }
            store.commitAudio(recording, quiet.size / 32_000.0)
            store.markTranscribed(recording.id, "quiet")
        }
        var calls = 0
        // The run starts with 3 MiB in the class; the newer take is
        // compressed after that, before the removal is decided, and the
        // class fits.
        val gate = Gate(policy(standard = ClassLimits(maxTotalMb = 2))) {
            if (++calls == 2) assertTrue(store.compressAudio(newer.id) is RecordingStore.Compression.Compressed)
        }

        val report = store.applyRetention(gate)

        assertEquals(emptyList<RemovedAudio>(), report.removed)
        assertTrue(hasAudio(older))
    }

    @Test
    fun aTakeStillRecordingIsNotCountedAndAStartingTakeStopsTheRun() {
        val store = store()
        val old = take(store, ageDays = 40.0)
        val recording = store.create()
        val writer = WavWriter(store.partialFile(recording)).apply { write(ByteArray(4 * mib), 4 * mib) }

        // A size limit the live take alone would exceed: it is not a row yet.
        val sized = store.applyRetention(Gate(policy(standard = ClassLimits(maxTotalMb = 2))))
        assertEquals(RetentionReport(), sized)

        val stopped = store.applyRetention(Gate(policy(standard = ClassLimits(maxAgeDays = 30)))) { true }
        assertTrue(stopped.stopped)
        assertTrue(hasAudio(old))
        writer.finish()
        assertEquals(RecordingStatus.RECORDING, store.get(recording.id).status)
    }

    @Test
    fun aPolicyChangeStopsTheRunBeforeTheNextRemoval() {
        val store = store()
        take(store, ageDays = 50.0)
        take(store, ageDays = 40.0)
        val gate = Gate(policy(standard = ClassLimits(maxAgeDays = 30)))
        var calls = 0
        gate.before = {
            // The user lifts the limit right after the first removal.
            if (++calls == 3) gate.policy = RetentionPolicy()
        }

        val report = store.applyRetention(gate)

        assertTrue(report.policyChanged)
        assertEquals(1, report.removed.size)
    }

    @Test
    fun aCrashBetweenTheStampAndTheUnlinkIsFinishedOnTheNextOpen() {
        val store = store()
        val old = take(store, ageDays = 40.0)
        val audio = File(storeDir(), "${old.id}.wav").readBytes()
        store.applyRetention(Gate(policy(standard = ClassLimits(maxAgeDays = 30))))
        // As if the process died after the stamp was saved, before the unlink.
        File(storeDir(), "${old.id}.wav").writeBytes(audio)

        val reopened = RecordingStore(storeDir()) { now }

        assertFalse(hasAudio(old))
        assertNotNull(reopened.get(old.id).audioRemoved)
        assertEquals(emptyList<String>(), reopened.compressionCandidates())
    }

    /**
     * A partial that was not this app's WAV is kept aside as
     * `.wav.unrecognized` (#356): never compressed, never counted toward a
     * size limit and never removed by one; only Delete removes it.
     */
    @Test
    fun unrecognizedAudioIsNeitherCompressedCountedNorRemoved() {
        val first = store()
        now -= 40 * day
        val foreign = first.create()
        val mixed = first.create()
        now += 40 * day
        val foreignBytes = WavWriter.header(mib.toLong()).also { it[24] = 0x44; it[25] = 0xac.toByte() } + ByteArray(mib) { 3 }
        first.partialFile(foreign).writeBytes(foreignBytes)
        // A clean Stop whose metadata commit was cut off, beside an earlier
        // foreign partial set aside for the same take.
        WavWriter(first.partialFile(mixed)).apply { write(ByteArray(mib), mib); finish() }
        assertTrue(first.partialFile(mixed).renameTo(File(storeDir(), "${mixed.id}.wav")))
        val mixedAside = File(storeDir(), "${mixed.id}.wav.unrecognized").apply { writeBytes(foreignBytes) }

        val store = RecordingStore(storeDir()) { now }
        val foreignAside = File(storeDir(), "${foreign.id}.wav.unrecognized")
        assertEquals(RecordingStore.UNRECOGNIZED_CAPTURE, store.get(foreign.id).errorMessage)
        assertTrue(foreignAside.isFile)
        store.markTranscribed(mixed.id, "text")

        // Only the mixed take's WAV is compressed; its aside file stays as is.
        assertEquals(listOf(mixed.id), store.compressionCandidates())
        assertTrue(store.compressAudio(mixed.id) is RecordingStore.Compression.Compressed)
        assertTrue(store.compressAudio(foreign.id) is RecordingStore.Compression.Skipped)
        val mixedFlac = File(storeDir(), "${mixed.id}.flac").length()

        // A size limit the unrecognized bytes alone would exceed: they are not counted.
        val sized = store.applyRetention(Gate(policy(standard = ClassLimits(maxTotalMb = 1))))
        assertEquals(RetentionReport(), sized)
        // An age limit removes the mixed take's FLAC, never either aside file.
        val aged = store.applyRetention(Gate(policy(standard = ClassLimits(maxAgeDays = 30))))
        assertEquals(listOf(RemovedAudio(mixed.id, RetentionClass.STANDARD, mixedFlac, RetireReason.AGE)), aged.removed)
        assertTrue(foreignAside.readBytes().contentEquals(foreignBytes))
        assertTrue(mixedAside.readBytes().contentEquals(foreignBytes))
        // A reopen keeps them too; Delete is what removes them.
        RecordingStore(storeDir()) { now }.apply {
            delete(foreign.id)
            delete(mixed.id)
        }
        assertFalse(foreignAside.exists() || mixedAside.exists())
    }

    @Test
    fun compressedTakesAreRemovedByTheirFlacSize() {
        val store = store()
        val old = take(store, ageDays = 40.0)
        store.compressAudio(old.id)
        val flacBytes = File(storeDir(), "${old.id}.flac").length()

        val report = store.applyRetention(Gate(policy(standard = ClassLimits(maxAgeDays = 30))))

        assertEquals(flacBytes, report.removedBytes)
        assertFalse(hasAudio(old))
    }

    @Test
    fun anUnlinkStorageRefusesIsReportedAndTheTakeKeepsItsAudio() {
        var refuse = true
        val stuck = mutableSetOf<String>()
        val store = RecordingStore(
            storeDir(),
            unlink = { file -> if (refuse && file.name.substringBefore('.') in stuck) false else file.delete() },
        ) { now }
        val older = take(store, ageDays = 60.0)
        val old = take(store, ageDays = 40.0)
        stuck += old.id
        val gate = Gate(policy(standard = ClassLimits(maxAgeDays = 30)))

        val report = store.applyRetention(gate)

        // The refused one is reported, not counted as removed, and the run
        // goes on to the next due take instead of trying it again.
        assertEquals(listOf(older.id), report.removed.map { it.id })
        assertEquals(listOf(old.id), report.failed.map { it.id })
        assertTrue(hasAudio(old))
        assertEquals(null, store.get(old.id).audioRemoved)
        assertNotNull(store.get(older.id).audioRemoved)
        assertEquals(RetireReason.AGE, report.failed.single().reason)
        // A later run tries again.
        refuse = false
        val retried = store.applyRetention(gate)
        assertEquals(listOf(old.id), retried.removed.map { it.id })
        assertEquals(emptyList<RemovedAudio>(), retried.failed)
        assertFalse(hasAudio(old))
        assertNotNull(store.get(old.id).audioRemoved)
    }

    /**
     * A retention removal lands between a settle's verify and its unlink,
     * and storage refuses both its unlink and the take-back (the disk is
     * full): the take stays stamped with both files. The settle must not
     * unlink the WAV (leaving the FLAC alone on a removed take); the next
     * open finishes the removal.
     */
    @Test
    fun aSettleLeavesATakeRetentionStampedMeanwhileToTheRemoval() {
        var refuseWav = true
        var refuseRemoval = false
        val store = RecordingStore(
            storeDir(),
            unlink = { file ->
                when {
                    refuseWav && file.name.endsWith(".wav") -> false
                    refuseRemoval && file.name.endsWith(".flac") -> {
                        // The take-back save cannot write its temporary either.
                        File(storeDir(), ".${file.name.removeSuffix(".flac")}.json.tmp").mkdir()
                        false
                    }
                    else -> file.delete()
                }
            },
            clock = { now },
        )
        val old = take(store, ageDays = 40.0)
        assertTrue((store.compressAudio(old.id) as RecordingStore.Compression.Compressed).wavKept)
        refuseWav = false
        val wav = File(storeDir(), "${old.id}.wav")
        val flac = File(storeDir(), "${old.id}.flac")

        var interleaved = false
        val waiting = store.settleAtRest(stop = {
            if (!interleaved) {
                interleaved = true
                refuseRemoval = true
                try {
                    store.applyRetention(Gate(policy(standard = ClassLimits(maxAgeDays = 30))))
                    fail("the take-back did not fail")
                } catch (_: IOException) {
                }
                refuseRemoval = false
            }
            false
        })

        assertEquals(0, waiting)
        assertNotNull(store.get(old.id).audioRemoved)
        assertTrue(wav.isFile && flac.isFile)
        File(storeDir(), ".${old.id}.json.tmp").delete()
        val reopened = RecordingStore(storeDir()) { now }
        assertFalse(hasAudio(old))
        assertNotNull(reopened.get(old.id).audioRemoved)
    }
}
