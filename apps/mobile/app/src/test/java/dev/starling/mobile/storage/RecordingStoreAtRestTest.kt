package dev.starling.mobile.storage

import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.engine.WavPcm
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/** #342: finished takes at rest as FLAC; every crash window keeps the WAV or the whole FLAC. */
class RecordingStoreAtRestTest {
    @get:Rule
    val folder = TemporaryFolder()

    private fun storeDir(): File = File(folder.root, "recordings")

    private val speech: ByteArray by lazy {
        val fixture = generateSequence(File("").absoluteFile) { it.parentFile }
            .map { File(it, "tests/fixtures/2086-149220-0033.wav") }
            .first(File::isFile)
        WavPcm.decodePcm16(fixture)!!.pcm
    }

    /** A take committed the way every entry point commits one, transcribed once. */
    private fun committedTake(store: RecordingStore, pcm: ByteArray = speech, transcribed: Boolean = true): Recording {
        val recording = store.create()
        WavWriter(store.partialFile(recording)).apply {
            write(pcm, pcm.size)
            finish()
        }
        store.commitAudio(recording, pcm.size / 32_000.0)
        return if (transcribed) store.markTranscribed(recording.id, "text") else store.get(recording.id)
    }

    private fun requestAudio(store: RecordingStore, id: String): ByteArray =
        store.withRequestAudio(id) { it.readBytes() }

    private fun wav(id: String) = File(storeDir(), "$id.wav")

    private fun flac(id: String) = File(storeDir(), "$id.flac")

    private fun strays(): List<String> = storeDir().list().orEmpty().filter { it.startsWith(".") }

    @Test
    fun aCompressedTakeSendsByteIdenticalRequestAudio() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        val fromWav = requestAudio(store, take.id)
        assertArrayEquals(WavWriter.header(speech.size.toLong()) + speech, fromWav)

        val outcome = store.compressAudio(take.id) as RecordingStore.Compression.Compressed

        assertFalse(wav(take.id).exists())
        assertTrue(flac(take.id).isFile)
        assertEquals(fromWav.size.toLong(), outcome.wavBytes)
        assertTrue("${outcome.flacBytes} of ${outcome.wavBytes}", outcome.flacBytes < outcome.wavBytes * 0.65)
        assertArrayEquals(fromWav, requestAudio(store, take.id))
        assertEquals(flac(take.id), store.audioFile(take))
        assertTrue(store.openAudio(take.id).run { stream.close(); flac })
        // The decoded request WAV is a temporary of that attempt only.
        assertEquals(emptyList<String>(), strays())
        // And it survives a restart as it is.
        val reopened = RecordingStore(storeDir())
        assertArrayEquals(fromWav, requestAudio(reopened, take.id))
        assertEquals(RecordingStatus.TRANSCRIBED, reopened.get(take.id).status)
    }

    @Test
    fun aCrashBeforeThePublishLeavesTheWavAndNoTemporary() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        val expected = requestAudio(store, take.id)
        // The process died after the temporary FLAC was written.
        FileOutputStream(File(storeDir(), ".${take.id}.flac.tmp")).use { output ->
            Flac.encode(speech.inputStream(), speech.size / 2L, 16_000, output)
        }

        val reopened = RecordingStore(storeDir())

        assertEquals(emptyList<String>(), strays())
        assertTrue(wav(take.id).isFile)
        assertFalse(flac(take.id).exists())
        assertArrayEquals(expected, requestAudio(reopened, take.id))
        assertEquals(listOf(take.id), reopened.compressionCandidates())
    }

    @Test
    fun aCrashBetweenThePublishAndTheUnlinkKeepsTheFlacOnceSettledAfterTheOpen() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        val expected = requestAudio(store, take.id)
        store.compressionHook = { step -> if (step == RecordingStore.CompressionStep.PUBLISHED) throw IllegalStateException("killed") }
        try {
            store.compressAudio(take.id)
            fail("the injected crash did not happen")
        } catch (_: IllegalStateException) {
        }
        assertTrue(wav(take.id).isFile && flac(take.id).isFile)
        // Even before the next open, a retry reads the same audio.
        assertArrayEquals(expected, requestAudio(store, take.id))

        val reopened = RecordingStore(storeDir())

        // The open decodes nothing (it runs in Application.onCreate): both
        // files stay, and every reader uses the WAV, the original.
        assertTrue(wav(take.id).isFile && flac(take.id).isFile)
        assertArrayEquals(expected, requestAudio(reopened, take.id))
        assertFalse(reopened.openAudio(take.id).run { stream.close(); flac })
        assertEquals(wav(take.id), reopened.audioFile(take))
        assertEquals(emptyList<String>(), reopened.compressionCandidates())

        // The upkeep thread settles it.
        assertEquals(0, reopened.settleAtRest())
        assertFalse(wav(take.id).exists())
        assertTrue(flac(take.id).isFile)
        assertArrayEquals(expected, requestAudio(reopened, take.id))
    }

    @Test
    fun aFlacThatChangesAfterTheComparisonIsComparedAgainBeforeAnyUnlink() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        val expected = requestAudio(store, take.id)
        store.compressionHook = { step -> if (step == RecordingStore.CompressionStep.PUBLISHED) throw IllegalStateException("killed") }
        try {
            store.compressAudio(take.id)
            fail("the injected crash did not happen")
        } catch (_: IllegalStateException) {
        }
        // The FLAC matched its WAV when compared, and became other audio before the unlink.
        store.compressionHook = { step ->
            if (step == RecordingStore.CompressionStep.SETTLE_VERIFIED) {
                FileOutputStream(flac(take.id)).use { output ->
                    Flac.encode(ByteArray(speech.size).inputStream(), speech.size / 2L, 16_000, output)
                }
            }
        }

        assertEquals(1, store.settleAtRest())
        assertTrue(wav(take.id).isFile && flac(take.id).isFile)

        // The next call compares again: the WAV, the original, stays.
        store.compressionHook = null
        assertEquals(0, store.settleAtRest())
        assertTrue(wav(take.id).isFile)
        assertFalse(flac(take.id).exists())
        assertArrayEquals(expected, requestAudio(store, take.id))
    }

    @Test
    fun settlingPausesForARecordingAndADeleteNeverLeavesAnUnsettledFlacAlone() {
        var refuseFlac = false
        val store = RecordingStore(
            storeDir(),
            unlink = { file -> if (refuseFlac && file.name.endsWith(".flac")) false else file.delete() },
        )
        val take = committedTake(store)
        val expected = requestAudio(store, take.id)
        store.compressionHook = { step -> if (step == RecordingStore.CompressionStep.PUBLISHED) throw IllegalStateException("killed") }
        runCatching { store.compressAudio(take.id) }
        val reopened = RecordingStore(
            storeDir(),
            unlink = { file -> if (refuseFlac && file.name.endsWith(".flac")) false else file.delete() },
        )

        // A take started recording: nothing is settled.
        assertEquals(1, reopened.settleAtRest(stop = { true }))
        assertTrue(wav(take.id).isFile && flac(take.id).isFile)
        // A delete whose FLAC unlink is refused keeps the WAV too.
        refuseFlac = true
        try {
            reopened.delete(take.id)
            fail("a refused delete succeeded")
        } catch (_: IOException) {
        }
        assertTrue(wav(take.id).isFile && flac(take.id).isFile)
        assertArrayEquals(expected, requestAudio(reopened, take.id))
        refuseFlac = false
        reopened.delete(take.id)
        assertFalse(wav(take.id).exists() || flac(take.id).exists())
        assertEquals(0, reopened.settleAtRest())
    }

    @Test
    fun aPinnedTakeIsSettledOnlyOnceItIsReleased() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        store.compressionHook = { step -> if (step == RecordingStore.CompressionStep.PUBLISHED) throw IllegalStateException("killed") }
        runCatching { store.compressAudio(take.id) }
        val reopened = RecordingStore(storeDir())

        val pin = reopened.pin(take.id)
        // A retry reading the WAV keeps it.
        assertEquals(1, reopened.settleAtRest())
        assertTrue(wav(take.id).isFile && flac(take.id).isFile)
        pin.close()
        assertEquals(0, reopened.settleAtRest())
        assertFalse(wav(take.id).exists())
    }

    @Test
    fun aFailedSyncAfterThePublishKeepsTheWavUntilSettled() {
        var failSync = false
        val store = RecordingStore(
            storeDir(),
            syncDir = { if (failSync) throw IOException("injected fsync failure") },
        )
        val take = committedTake(store)
        val expected = requestAudio(store, take.id)
        // The rename landed, but whether it is durable is unknown: the WAV stays.
        failSync = true
        val outcome = store.compressAudio(take.id) as RecordingStore.Compression.Compressed
        assertTrue(outcome.wavKept)
        assertTrue(wav(take.id).isFile && flac(take.id).isFile)
        assertArrayEquals(expected, requestAudio(store, take.id))
        // Settling waits for a sync that works.
        assertEquals(1, store.settleAtRest())
        assertTrue(wav(take.id).isFile)
        failSync = false
        assertEquals(0, store.settleAtRest())
        assertFalse(wav(take.id).exists())
        assertArrayEquals(expected, requestAudio(store, take.id))
    }

    @Test
    fun aWavStorageWillNotUnlinkAfterThePublishStaysUntilSettled() {
        var refuse = true
        val store = RecordingStore(
            storeDir(),
            unlink = { file -> if (refuse && file.name.endsWith(".wav")) false else file.delete() },
        )
        val take = committedTake(store)
        val expected = requestAudio(store, take.id)

        // The FLAC is published: the take is compressed, nothing saved yet.
        val outcome = store.compressAudio(take.id) as RecordingStore.Compression.Compressed
        assertTrue(outcome.wavKept)
        assertTrue(wav(take.id).isFile && flac(take.id).isFile)
        assertArrayEquals(expected, requestAudio(store, take.id))
        assertEquals(emptyList<String>(), store.compressionCandidates())
        // Still refused: it waits; then the unlink goes through.
        assertEquals(1, store.settleAtRest())
        assertTrue(wav(take.id).isFile)
        refuse = false
        assertEquals(0, store.settleAtRest())
        assertFalse(wav(take.id).exists())
        assertArrayEquals(expected, requestAudio(store, take.id))
    }

    @Test
    fun aDamagedFlacBesideItsWavIsDroppedAndTheWavKept() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        val expected = requestAudio(store, take.id)
        store.compressionHook = { step -> if (step == RecordingStore.CompressionStep.PUBLISHED) throw IllegalStateException("killed") }
        runCatching { store.compressAudio(take.id) }
        val damaged = flac(take.id).readBytes()
        damaged[damaged.size / 2] = (damaged[damaged.size / 2].toInt() xor 0x10).toByte()
        flac(take.id).writeBytes(damaged)

        val reopened = RecordingStore(storeDir())
        assertEquals(0, reopened.settleAtRest())

        assertTrue(wav(take.id).isFile)
        assertFalse(flac(take.id).exists())
        assertArrayEquals(expected, requestAudio(reopened, take.id))
    }

    @Test
    fun aFlacWithADamagedSampleRateBesideItsWavIsDroppedAndTheWavKept() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        val expected = requestAudio(store, take.id)
        store.compressionHook = { step -> if (step == RecordingStore.CompressionStep.PUBLISHED) throw IllegalStateException("killed") }
        runCatching { store.compressAudio(take.id) }
        // STREAMINFO's sample rate (its 20 bits start at byte 18): the samples
        // and their MD5 still match, but the request path refuses the stream.
        val damaged = flac(take.id).readBytes()
        damaged[18] = (damaged[18].toInt() xor 0x01).toByte()
        flac(take.id).writeBytes(damaged)

        val reopened = RecordingStore(storeDir())
        assertEquals(0, reopened.settleAtRest())

        assertTrue(wav(take.id).isFile)
        assertFalse(flac(take.id).exists())
        assertArrayEquals(expected, requestAudio(reopened, take.id))
    }

    @Test
    fun aDamagedFlacAloneFailsTheAttemptInsteadOfSendingOtherAudio() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        store.compressAudio(take.id)
        val damaged = flac(take.id).readBytes()
        damaged[damaged.size / 2] = (damaged[damaged.size / 2].toInt() xor 0x01).toByte()
        flac(take.id).writeBytes(damaged)
        try {
            requestAudio(store, take.id)
            fail("damaged audio was sent")
        } catch (_: IOException) {
        }
        assertEquals(emptyList<String>(), strays())
    }

    @Test
    fun pinnedAudioIsNeverCompressed() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        val pin = store.pin(take.id)

        assertEquals(emptyList<String>(), store.compressionCandidates())
        assertEquals(RecordingStore.Compression.Skipped("in use"), store.compressAudio(take.id))
        pin.close()
        pin.close() // a second close does not unpin someone else
        assertTrue(store.compressAudio(take.id) is RecordingStore.Compression.Compressed)
    }

    @Test
    fun aRetryThatStartsDuringTheEncodeKeepsItsWav() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        val expected = requestAudio(store, take.id)
        var during: ByteArray? = null
        store.compressionHook = { step ->
            if (step == RecordingStore.CompressionStep.ENCODED) {
                // A retry reads its audio between the encode and the publish;
                // the publish runs while it still holds the pin.
                store.withRequestAudio(take.id) { file ->
                    during = file.readBytes()
                    store.compressionHook = null
                    val result = runCatching { store.compressAudio(take.id) }.getOrNull()
                    assertEquals(RecordingStore.Compression.Skipped("in use"), result)
                }
            }
        }

        // This compression's own publish sees nothing in use any more and goes ahead.
        assertTrue(store.compressAudio(take.id) is RecordingStore.Compression.Compressed)
        assertArrayEquals(expected, during)
        assertArrayEquals(expected, requestAudio(store, take.id))
    }

    @Test
    fun aPinTakenBeforeThePublishStopsIt() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        var pin: AutoCloseable? = null
        store.compressionHook = { step -> if (step == RecordingStore.CompressionStep.ENCODED) pin = store.pin(take.id) }

        assertEquals(RecordingStore.Compression.Skipped("in use"), store.compressAudio(take.id))
        assertTrue(wav(take.id).isFile)
        assertFalse(flac(take.id).exists())
        assertEquals(emptyList<String>(), strays())
        pin!!.close()
    }

    @Test
    fun aTakeIsCompressedByOneCompressionAtATime() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        var second: RecordingStore.Compression? = null
        store.compressionHook = { step ->
            if (step == RecordingStore.CompressionStep.ENCODED && second == null) {
                second = store.compressAudio(take.id)
            }
        }

        assertTrue(store.compressAudio(take.id) is RecordingStore.Compression.Compressed)
        assertEquals(RecordingStore.Compression.Skipped("compressing"), second)
        assertEquals(emptyList<String>(), store.compressionCandidates())
    }

    @Test
    fun aDeleteDuringTheEncodeLeavesNothingBehind() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        store.compressionHook = { step -> if (step == RecordingStore.CompressionStep.ENCODED) store.delete(take.id) }

        assertEquals(RecordingStore.Compression.Skipped("deleted"), store.compressAudio(take.id))
        assertEquals(emptyList<String>(), storeDir().list().orEmpty().toList())
    }

    @Test
    fun deleteRemovesTheFlac() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        store.compressAudio(take.id)
        store.delete(take.id)
        assertEquals(emptyList<String>(), storeDir().list().orEmpty().toList())
    }

    @Test
    fun onlyFinalizedSettledTakesAreCompressed() {
        val store = RecordingStore(storeDir())
        val second = 32_000
        // Still recording: its partial WAV is open.
        val recording = store.create()
        val writer = WavWriter(store.partialFile(recording)).apply { write(ByteArray(second), second) }
        // Transcribing right now.
        val transcribing = committedTake(store).also { store.markTranscribing(it.id) }
        // A private field's take.
        val private = store.create(ephemeral = true).also {
            WavWriter(store.partialFile(it)).apply { write(ByteArray(second), second); finish() }
            store.commitAudio(it, 1.0)
        }
        // Settled, never transcribed: compressed like any finished take.
        val pending = committedTake(store, transcribed = false)
        val failed = committedTake(store).also { store.markFailed(it.id, "server down") }

        assertEquals(setOf(pending.id, failed.id), store.compressionCandidates().toSet())
        listOf(recording, transcribing, private).forEach { take ->
            assertTrue(store.compressAudio(take.id) is RecordingStore.Compression.Skipped)
        }
        writer.finish()
    }

    @Test
    fun anInterruptedTakeIsCompressedOnlyOnceRecovered() {
        val first = RecordingStore(storeDir())
        val recording = first.create()
        val pcm = speech.copyOf(3 * 32_000)
        // Killed mid-take: the header confirms two of the three seconds.
        first.partialFile(recording).writeBytes(WavWriter.header(2L * 32_000) + pcm)
        assertEquals(emptyList<String>(), first.compressionCandidates())

        val reopened = RecordingStore(storeDir())
        val recovered = reopened.get(recording.id)
        assertEquals(RecordingStatus.PENDING, recovered.status)
        assertEquals(listOf(recording.id), reopened.compressionCandidates())
        reopened.compressAudio(recording.id)
        assertArrayEquals(WavWriter.header(pcm.size.toLong()) + pcm, requestAudio(reopened, recording.id))
        assertEquals(recovered.recovery, reopened.get(recording.id).recovery)
    }

    /**
     * A delete while a transcription or export reads the take: it is gone
     * for everyone else at once, and the files the reader holds stay until
     * it is done. Neither the decoded request WAV nor the FLAC behind it is
     * unlinked mid-read.
     */
    @Test
    fun aDeleteDuringARequestReadWaitsForTheReader() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        val expected = requestAudio(store, take.id)
        store.compressAudio(take.id)

        val read = store.withRequestAudio(take.id) { request ->
            store.delete(take.id)
            assertTrue(store.list().none { it.id == take.id })
            try {
                store.get(take.id)
                fail("a deleted take was still readable")
            } catch (_: IOException) {
            }
            // The attempt's result cannot bring it back.
            try {
                store.markTranscribed(take.id, "late")
                fail("a deleted take took a transcript")
            } catch (_: IOException) {
            }
            // Nor does playback or an export open it again.
            try {
                store.openAudio(take.id).stream.close()
                fail("a deleted take opened for playback")
            } catch (_: IOException) {
            }
            try {
                store.exportAudio(take.id, flac = true, output = java.io.ByteArrayOutputStream())
                fail("a deleted take was exported")
            } catch (_: IOException) {
            }
            assertTrue(request.isFile && flac(take.id).isFile)
            request.readBytes()
        }

        assertArrayEquals(expected, read)
        assertEquals(emptyList<String>(), storeDir().list().orEmpty().toList())
    }

    @Test
    fun aDeleteOfAPinnedWavTakeWaitsForTheLastPin() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        val first = store.pin(take.id)
        val second = store.pin(take.id)
        store.delete(take.id)
        assertTrue(wav(take.id).isFile)
        assertEquals(emptyList<String>(), store.compressionCandidates())
        first.close()
        assertTrue(wav(take.id).isFile)
        second.close()
        assertEquals(emptyList<String>(), storeDir().list().orEmpty().toList())
    }

    @Test
    fun aCommitCannotBringBackATakeDeletedWhilePinned() {
        val store = RecordingStore(storeDir())
        val recording = store.create()
        WavWriter(store.partialFile(recording)).apply {
            write(speech, speech.size)
            finish()
        }
        store.pin(recording.id)
        store.delete(recording.id)
        try {
            store.commitAudio(recording, speech.size / 32_000.0)
            fail("a deleted take was committed")
        } catch (_: IOException) {
        }
        assertEquals(emptyList<Recording>(), store.list())
        // The process dies with the pin held: the next open finishes the delete.
        RecordingStore(storeDir())
        assertEquals(emptyList<String>(), storeDir().list().orEmpty().toList())
    }

    @Test
    fun aDeleteHeldBackByAReaderIsFinishedAtTheNextOpen() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        store.pin(take.id)
        store.delete(take.id)
        // The process dies before the reader is done.
        val reopened = RecordingStore(storeDir())
        assertEquals(emptyList<Recording>(), reopened.list())
        assertEquals(emptyList<String>(), storeDir().list().orEmpty().toList())
    }

    @Test
    fun anExportTakesItsFormatAndItsBytesTogether() {
        val store = RecordingStore(storeDir())
        val take = committedTake(store)
        val expected = requestAudio(store, take.id)
        // Named .flac while the take is still a WAV: refused, never WAV bytes under that name.
        try {
            store.exportAudio(take.id, flac = true, output = java.io.ByteArrayOutputStream())
            fail("WAV bytes were exported as FLAC")
        } catch (_: IOException) {
        }
        store.compressAudio(take.id)
        val asFlac = java.io.ByteArrayOutputStream().also { store.exportAudio(take.id, flac = true, output = it) }
        assertArrayEquals(flac(take.id).readBytes(), asFlac.toByteArray())
        // Named .wav before the compression: still the original WAV.
        val asWav = java.io.ByteArrayOutputStream().also { store.exportAudio(take.id, flac = false, output = it) }
        assertArrayEquals(expected, asWav.toByteArray())
    }
}
