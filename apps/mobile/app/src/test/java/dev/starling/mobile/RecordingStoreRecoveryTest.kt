package dev.starling.mobile

import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.RecordingStatus
import dev.starling.mobile.data.TranscriptSource
import dev.starling.mobile.data.TranscriptionProvenance
import dev.starling.mobile.storage.RecordingStore
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/** #356: audio outlives the capture, the process and every inference outcome. */
class RecordingStoreRecoveryTest {
    @get:Rule
    val folder = TemporaryFolder()

    private fun storeDir(): File = File(folder.root, "recordings")

    /** One second of 16 kHz PCM16. */
    private val second = WavWriter.SAMPLE_RATE * WavWriter.BYTES_PER_SAMPLE

    private fun pcm(bytes: Int): ByteArray = ByteArray(bytes) { (it % 251).toByte() }

    /** A partial WAV as a killed capture leaves it: [confirmed] in the header, [written] on disk. */
    private fun killedCapture(file: File, written: Int, confirmed: Int): ByteArray {
        val payload = pcm(written)
        file.writeBytes(WavWriter.header(confirmed.toLong()) + payload)
        return payload
    }

    @Test
    fun aCaptureKilledMidRecordingIsRecoveredOnOpenWithItsDurableBoundary() {
        val first = RecordingStore(storeDir())
        val recording = first.create()
        val payload = killedCapture(first.partialFile(recording), written = 3 * second, confirmed = 2 * second)

        val reopened = RecordingStore(storeDir())
        val recovered = reopened.get(recording.id)

        assertEquals(RecordingStatus.PENDING, recovered.status)
        assertEquals(3.0, recovered.durationSeconds, 1e-9)
        val recovery = recovered.recovery!!
        assertEquals(RecordingStore.INTERRUPTED_CAPTURE, recovery.reason)
        assertEquals(3.0, recovery.recoveredSeconds, 1e-9)
        assertEquals(2.0, recovery.confirmedSeconds, 1e-9)
        assertFalse(reopened.partialFile(recording).exists())
        // Every sample the process wrote, as a valid WAV.
        val audio = reopened.audioFile(recovered).readBytes()
        assertArrayEquals(WavWriter.header(payload.size.toLong()), audio.copyOfRange(0, 44))
        assertArrayEquals(payload, audio.copyOfRange(44, audio.size))
        assertEquals(listOf(recording.id), reopened.list().map { it.id })
    }

    @Test
    fun aTornTrailingByteIsDroppedAndAnUnreadableHeaderConfirmsNothing() {
        val first = RecordingStore(storeDir())
        val recording = first.create()
        // Not a header this app wrote, plus half a sample at the end.
        first.partialFile(recording).writeBytes(ByteArray(44) + pcm(second + 1))

        val recovered = RecordingStore(storeDir()).get(recording.id)

        assertEquals(RecordingStatus.PENDING, recovered.status)
        assertEquals(1.0, recovered.recovery!!.recoveredSeconds, 1e-9)
        assertEquals(0.0, recovered.recovery!!.confirmedSeconds, 1e-9)
        assertEquals(44L + second, File(storeDir(), recovered.wavName).length())
    }

    @Test
    fun aCaptureKilledBeforeItsFirstSampleFailsWithoutPretendingToHaveAudio() {
        val first = RecordingStore(storeDir())
        val recording = first.create()
        first.partialFile(recording).writeBytes(WavWriter.header(0))

        val reopened = RecordingStore(storeDir())
        val failed = reopened.get(recording.id)

        assertEquals(RecordingStatus.FAILED, failed.status)
        assertEquals(RecordingStore.INTERRUPTED_CAPTURE, failed.errorMessage)
        assertNull(failed.recovery)
        assertFalse(reopened.partialFile(recording).exists())
        assertFalse(reopened.audioFile(recording).exists())
    }

    @Test
    fun aCleanStopCutOffBeforeItsMetadataCommitIsPendingAndNotMarkedRecovered() {
        val first = RecordingStore(storeDir())
        val recording = first.create()
        // commitAudio promoted the WAV, then the process died before save().
        File(storeDir(), recording.wavName).writeBytes(WavWriter.header(second.toLong()) + pcm(second))

        val recovered = RecordingStore(storeDir()).get(recording.id)

        assertEquals(RecordingStatus.PENDING, recovered.status)
        assertNull(recovered.recovery)
        assertEquals(1.0, recovered.durationSeconds, 1e-9)
    }

    @Test
    fun recoveryResumesAfterACrashBetweenTheNoteAndThePromotion() {
        val first = RecordingStore(storeDir())
        val recording = first.create()
        killedCapture(first.partialFile(recording), written = 2 * second, confirmed = second)
        RecordingStore(storeDir())
        // Simulate the earlier attempt dying after its note was saved but
        // before the WAV moved: put the partial back beside the note.
        val audio = File(storeDir(), recording.wavName)
        assertTrue(audio.renameTo(first.partialFile(recording)))

        val recovered = RecordingStore(storeDir()).get(recording.id)

        assertEquals(RecordingStatus.PENDING, recovered.status)
        assertEquals(2.0, recovered.recovery!!.recoveredSeconds, 1e-9)
        // The repaired header now covers everything; the first note's
        // boundary is the honest one and is kept.
        assertEquals(1.0, recovered.recovery!!.confirmedSeconds, 1e-9)
        assertTrue(audio.isFile)
    }

    @Test
    fun anInterruptedTranscriptionBecomesRetryableAndKeepsItsAudio() {
        val first = RecordingStore(storeDir())
        val recording = first.create()
        WavWriter(first.partialFile(recording)).apply { write(pcm(second), second); finish() }
        first.commitAudio(recording, 1.0)
        first.markTranscribing(recording.id)

        val reopened = RecordingStore(storeDir())
        val failed = reopened.get(recording.id)

        assertEquals(RecordingStatus.FAILED, failed.status)
        assertEquals(RecordingStore.INTERRUPTED_TRANSCRIPTION, failed.errorMessage)
        assertTrue(reopened.audioFile(failed).isFile)
    }

    @Test
    fun listAndGetNeverTouchACaptureThisProcessIsStillWriting() {
        val store = RecordingStore(storeDir())
        val recording = store.create()
        val partial = store.partialFile(recording)
        killedCapture(partial, written = second, confirmed = 0)

        assertEquals(RecordingStatus.RECORDING, store.list().single().status)
        assertEquals(RecordingStatus.RECORDING, store.get(recording.id).status)
        assertTrue(partial.isFile)
        assertEquals(44L + second, partial.length())
    }

    @Test
    fun aFailedCaptureKeepsWhatItRecorded() {
        val store = RecordingStore(storeDir())
        val recording = store.create()
        // The capture worker finalizes its WAV before it reports a failure.
        WavWriter(store.partialFile(recording)).apply { write(pcm(2 * second), 2 * second); finish() }

        val salvaged = store.salvageCapture(recording.id, "The microphone stopped")

        assertEquals(RecordingStatus.PENDING, salvaged.status)
        assertEquals(2.0, salvaged.durationSeconds, 1e-9)
        assertEquals("The microphone stopped", salvaged.recovery!!.reason)
        // A finished WAV was synced as a whole.
        assertEquals(2.0, salvaged.recovery!!.confirmedSeconds, 1e-9)
        assertTrue(store.audioFile(salvaged).isFile)
        assertEquals(salvaged, RecordingStore(storeDir()).get(recording.id))
    }

    @Test
    fun aFailedCaptureWithoutAudioIsFailed() {
        val store = RecordingStore(storeDir())
        val recording = store.create()

        val salvaged = store.salvageCapture(recording.id, "Unable to initialize the microphone")

        assertEquals(RecordingStatus.FAILED, salvaged.status)
        assertEquals("Unable to initialize the microphone", salvaged.errorMessage)
    }

    @Test
    fun retriesAddRevisionsAndFailuresKeepThem() {
        var now = 1_000L
        val store = RecordingStore(storeDir()) { now }
        val recording = store.create()
        WavWriter(store.partialFile(recording)).apply { write(pcm(second), second); finish() }
        store.commitAudio(recording, 1.0)

        store.markTranscribing(recording.id)
        store.markTranscribed(
            recording.id,
            "first",
            TranscriptionProvenance.LIVE_STREAM,
            TranscriptSource.ON_DEVICE,
            "a.gguf",
        )
        now = 2_000L
        store.markTranscribing(recording.id)
        store.markFailed(recording.id, "The server is unreachable")
        val afterFailure = store.get(recording.id)
        assertEquals(listOf("first"), afterFailure.revisions.map { it.text })
        assertEquals("first", afterFailure.rawTranscript)

        now = 3_000L
        store.markTranscribing(recording.id)
        store.markTranscribed(recording.id, "second", TranscriptionProvenance.BATCH_UPLOAD, TranscriptSource.SERVER, "parakeet")

        val reopened = RecordingStore(storeDir()).get(recording.id)
        assertEquals(RecordingStatus.TRANSCRIBED, reopened.status)
        assertEquals("second", reopened.rawTranscript)
        assertEquals(TranscriptionProvenance.BATCH_UPLOAD, reopened.provenance)
        assertEquals(listOf("first", "second"), reopened.revisions.map { it.text })
        assertEquals(listOf("a.gguf", "parakeet"), reopened.revisions.map { it.model })
        assertEquals(listOf(TranscriptSource.ON_DEVICE, TranscriptSource.SERVER), reopened.revisions.map { it.source })
        assertEquals(listOf(1_000L, 3_000L), reopened.revisions.map { it.createdAtMillis })
        assertEquals(3, reopened.attempts)
        assertTrue(File(storeDir(), reopened.wavName).isFile)
    }

    @Test
    fun salvageNeverTouchesAWavAWriterOfThisProcessStillHolds() {
        val store = RecordingStore(storeDir())
        val recording = store.create()
        val partial = store.partialFile(recording)
        // A worker that outlived its stop (CaptureStopPolicy.ZOMBIE_RESULT).
        val zombie = WavWriter(partial)
        zombie.write(pcm(second), second)

        val settled = store.salvageCapture(recording.id, "The microphone did not stop cleanly")

        assertEquals(RecordingStatus.FAILED, settled.status)
        assertEquals(RecordingStore.UNRECOVERED_CAPTURE, settled.errorMessage)
        assertTrue(partial.isFile)
        assertFalse(store.audioFile(recording).exists())
        // It keeps writing into its own file, which nobody moved.
        zombie.write(pcm(second), second)
        zombie.finish()

        val recovered = RecordingStore(storeDir()).get(recording.id)
        assertEquals(RecordingStatus.PENDING, recovered.status)
        assertEquals(2.0, recovered.recovery!!.recoveredSeconds, 1e-9)
    }

    @Test
    fun aFailedPromotionKeepsTheFirstNoteAndIsRetriedAtTheNextOpen() {
        val first = RecordingStore(storeDir())
        val recording = first.create()
        killedCapture(first.partialFile(recording), written = 3 * second, confirmed = 2 * second)
        // A non-empty directory where the WAV belongs cannot be replaced.
        val blocker = File(storeDir(), recording.wavName).apply { mkdirs() }
        File(blocker, "x").writeBytes(ByteArray(1))

        val failed = RecordingStore(storeDir()).get(recording.id)

        assertEquals(RecordingStatus.FAILED, failed.status)
        assertEquals(RecordingStore.UNRECOVERED_CAPTURE, failed.errorMessage)
        assertEquals(2.0, failed.recovery!!.confirmedSeconds, 1e-9)

        blocker.deleteRecursively()
        val recovered = RecordingStore(storeDir()).get(recording.id)

        assertEquals(RecordingStatus.PENDING, recovered.status)
        assertEquals(RecordingStore.INTERRUPTED_CAPTURE, recovered.recovery!!.reason)
        assertEquals(3.0, recovered.recovery!!.recoveredSeconds, 1e-9)
        // Measured from the repaired header this time, but the first note wins.
        assertEquals(2.0, recovered.recovery!!.confirmedSeconds, 1e-9)
    }

    @Test
    fun anEmptyRetryNeverDisplacesAnEarlierTranscript() {
        val store = RecordingStore(storeDir())
        val recording = store.create()

        // A first attempt that hears nothing is still its result.
        assertEquals(RecordingStatus.TRANSCRIBED, store.markTranscribed(recording.id, "").status)
        store.markTranscribed(recording.id, "hello")
        val retried = store.markTranscribed(recording.id, " ")

        assertEquals(RecordingStatus.FAILED, retried.status)
        assertEquals(RecordingStore.NO_SPEECH_ON_RETRY, retried.errorMessage)
        assertEquals("hello", retried.rawTranscript)
        assertEquals(listOf("", "hello"), retried.revisions.map { it.text })
    }

    @Test
    fun aTranscriptSavedBeforeRevisionsSurvivesRetries() {
        val id = "00000000-0000-4000-8000-000000000356"
        storeDir().mkdirs()
        // Metadata as written before revisions existed: no "revisions" key.
        File(storeDir(), "$id.json").writeText(
            """{"id":"$id","created_at_ms":500,"wav_name":"$id.wav","status":"TRANSCRIBED",""" +
                """"duration_s":1.0,"attempts":1,"raw_transcript":"old text","error_message":null,""" +
                """"provenance":"LIVE_STREAM","ephemeral":false}""",
        )
        File(storeDir(), "$id.wav").writeBytes(WavWriter.header(second.toLong()) + pcm(second))
        val store = RecordingStore(storeDir()) { 2_000L }

        val legacy = store.get(id).revisions.single()
        assertEquals("old text", legacy.text)
        assertEquals(TranscriptionProvenance.LIVE_STREAM, legacy.provenance)
        assertNull(legacy.source)
        assertNull(legacy.model)
        assertEquals(500L, legacy.createdAtMillis)

        val blank = store.markTranscribed(id, " ")
        assertEquals(RecordingStatus.FAILED, blank.status)
        assertEquals("old text", blank.rawTranscript)

        store.markTranscribed(id, "new text", TranscriptionProvenance.BATCH_UPLOAD, TranscriptSource.SERVER, "parakeet")
        val reopened = RecordingStore(storeDir()).get(id)
        assertEquals("new text", reopened.rawTranscript)
        assertEquals(listOf("old text", "new text"), reopened.revisions.map { it.text })
    }
}
