package dev.starling.mobile.network

import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.Recording
import dev.starling.mobile.storage.RecordingStore
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.IOException

/** #342: which failures of an attempt on the stored audio stay retryable. */
class TranscriptionCoordinatorTest {
    @get:Rule
    val folder = TemporaryFolder()

    private fun committedTake(store: RecordingStore): Recording {
        val recording = store.create()
        WavWriter(store.partialFile(recording)).apply {
            write(ByteArray(32_000), 32_000)
            finish()
        }
        return store.commitAudio(recording, 1.0)
    }

    @Test
    fun anEngineOrNetworkIoFailureStaysRetryable() {
        val store = RecordingStore(File(folder.root, "recordings"))
        val take = committedTake(store)

        val result = transcribeStoredAudio(store, take.id) { throw IOException("Connection reset") }

        assertEquals(InferenceResult.Failure("Connection reset", true), result)
    }

    @Test
    fun unreadableStoredAudioIsNotRetryable() {
        val directory = File(folder.root, "recordings")
        val store = RecordingStore(directory)
        val take = committedTake(store)
        assertEquals(true, File(directory, "${take.id}.wav").delete())

        val result = transcribeStoredAudio(store, take.id) { InferenceResult.Success("never sent") }

        assertEquals(InferenceResult.Failure("The recording audio is missing", false), result)
    }

    @Test
    fun theEngineResultIsReturnedAsItIs() {
        val store = RecordingStore(File(folder.root, "recordings"))
        val take = committedTake(store)

        val result = transcribeStoredAudio(store, take.id) { audio ->
            InferenceResult.Success("${audio.length()}")
        }

        assertEquals(InferenceResult.Success("${44 + 32_000}"), result)
    }
}
