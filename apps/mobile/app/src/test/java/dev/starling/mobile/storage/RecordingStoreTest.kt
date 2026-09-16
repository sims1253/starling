package dev.starling.mobile.storage

import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RecordingStoreTest {
    @Test
    fun openingTheStoreSweepsOrphanedMetadataTemporaries() {
        val directory = tempDirectory()
        try {
            val store = RecordingStore(directory)
            val recording = store.create()
            // A crash between save()'s temporary write and its rename leaves
            // exactly this unnamed file behind; nothing else ever removes it.
            val orphan = File(directory, ".${recording.id}.json.tmp")
            orphan.writeText("""{"interrupted":true}""")
            assertTrue(orphan.isFile)

            RecordingStore(directory)

            assertFalse(orphan.isFile)
            assertEquals(listOf(recording.id), store.list().map { it.id })
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun deleteRemovesTheMetadataTemporaryAlongsideTheRecording() {
        val directory = tempDirectory()
        try {
            val store = RecordingStore(directory)
            val recording = store.create()
            val orphan = File(directory, ".${recording.id}.json.tmp")
            orphan.writeText("""{"interrupted":true}""")
            assertTrue(orphan.isFile)

            store.delete(recording.id)

            assertFalse(orphan.isFile)
            assertFalse(File(directory, "${recording.id}.json").isFile)
            assertTrue(store.list().isEmpty())
        } finally {
            directory.deleteRecursively()
        }
    }

    private fun tempDirectory(): File = Files.createTempDirectory("starling-store-test").toFile()
}
