package dev.starling.mobile

import dev.starling.mobile.storage.RecordingStore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class RecordingStoreEphemeralTest {
    @get:Rule
    val folder = TemporaryFolder()

    private fun storeDir(): File = File(folder.root, "recordings")

    @Test
    fun ephemeralTakesAreHiddenFromHistory() {
        val store = RecordingStore(storeDir())
        val kept = store.create()
        val private = store.create(ephemeral = true)

        assertEquals(listOf(kept.id), store.list().map { it.id })
        // The keyboard still reaches its own take by id until it settles.
        assertTrue(store.get(private.id).ephemeral)
        store.markTranscribed(private.id, "hunter2")
        assertTrue(store.get(private.id).ephemeral)
        assertEquals(listOf(kept.id), store.list().map { it.id })
    }

    @Test
    fun ephemeralLeftoversAreDeletedOnOpen() {
        val first = RecordingStore(storeDir())
        val kept = first.create()
        val private = first.create(ephemeral = true)
        File(storeDir(), "${private.id}.wav.part").writeBytes(ByteArray(64))

        val reopened = RecordingStore(storeDir())

        assertEquals(listOf(kept.id), reopened.list().map { it.id })
        assertFalse(File(storeDir(), "${private.id}.json").exists())
        assertFalse(File(storeDir(), "${private.id}.wav.part").exists())
        assertTrue(File(storeDir(), "${kept.id}.json").exists())
    }

    @Test
    fun aFailedDeleteKeepsTheMarkerForTheNextSweep() {
        val store = RecordingStore(storeDir())
        val private = store.create(ephemeral = true)
        // A non-empty directory where the WAV belongs cannot be deleted.
        val stuck = File(storeDir(), "${private.id}.wav").apply { mkdirs() }
        File(stuck, "x").writeBytes(ByteArray(1))

        assertTrue(runCatching { store.delete(private.id) }.isFailure)
        assertTrue(File(storeDir(), "${private.id}.json").exists())

        stuck.deleteRecursively()
        RecordingStore(storeDir())
        assertFalse(File(storeDir(), "${private.id}.json").exists())
    }
}
