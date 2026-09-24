package dev.starling.mobile.engine

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Transactional model-import behavior (B08): a failed or interrupted import
 * must never leave the app without its last usable model, must reject
 * unrelated or corrupt GGUFs before replacing the active model, and must not
 * accumulate staging files.
 *
 * Fixtures are synthetic GGUFs: a real header plus string metadata KVs,
 * sparsely padded past ModelFiles.MIN_MODEL_BYTES (no multi-hundred-MB
 * payloads in the unit suite).
 */
class OnDeviceEngineImportTest {
    private fun tempDir(): File = Files.createTempDirectory("starling-model-test").toFile()

    private fun u32(value: Long): ByteArray = byteArrayOf(
        (value and 0xff).toByte(),
        ((value shr 8) and 0xff).toByte(),
        ((value shr 16) and 0xff).toByte(),
        ((value shr 24) and 0xff).toByte(),
    )

    private fun u64(value: Long): ByteArray {
        val out = ByteArray(8)
        for (i in 0 until 8) out[i] = ((value ushr (8 * i)) and 0xff).toByte()
        return out
    }

    private fun stringKv(key: String, value: String): ByteArray {
        val out = ByteArrayOutputStream()
        out.write(u64(key.length.toLong()))
        out.write(key.toByteArray(Charsets.UTF_8))
        out.write(u32(8)) // GGUF_TYPE_STRING
        out.write(u64(value.length.toLong()))
        out.write(value.toByteArray(Charsets.UTF_8))
        return out.toByteArray()
    }

    private fun ggufBytes(vararg kvs: Pair<String, String>): ByteArray {
        val out = ByteArrayOutputStream()
        out.write("GGUF".toByteArray(Charsets.US_ASCII))
        out.write(u32(2)) // version 2
        out.write(u64(0)) // tensor count (metadata check does not read tensors)
        out.write(u64(kvs.size.toLong()))
        for ((key, value) in kvs) out.write(stringKv(key, value))
        return out.toByteArray()
    }

    /** GGUFv2 header claiming [claimedKv] pairs but containing only [presentKv]. */
    private fun truncatedGgufBytes(claimedKv: Int, presentKv: List<Pair<String, String>>): ByteArray {
        val out = ByteArrayOutputStream()
        out.write("GGUF".toByteArray(Charsets.US_ASCII))
        out.write(u32(2))
        out.write(u64(0))
        out.write(u64(claimedKv.toLong()))
        for ((key, value) in presentKv) out.write(stringKv(key, value))
        return out.toByteArray()
    }

    /** Writes [prefix] then sparsely pads to [sizeBytes] (holes read as zeros). */
    private fun sparseModelFile(directory: File, name: String, prefix: ByteArray, sizeBytes: Long = ModelFiles.MIN_MODEL_BYTES): File {
        val file = File(directory, name)
        RandomAccessFile(file, "rw").use { out ->
            out.write(prefix)
            out.setLength(sizeBytes)
        }
        return file
    }

    /** Reads the first [count] bytes of [file] without materializing the whole file. */
    private fun prefix(file: File, count: Int): ByteArray =
        java.io.DataInputStream(file.inputStream().buffered()).use { input ->
            ByteArray(count).also { input.readFully(it) }
        }

    private fun parakeetPayload() = ggufBytes(
        "general.architecture" to "parakeet",
        "parakeet.preprocessor.normalize" to "per_feature",
    )

    private fun llamaPayload() = ggufBytes(
        "general.architecture" to "llama",
        "general.name" to "unrelated-chat-model",
        "llama.context_length" to "4096",
    )

    @Test
    fun unrelatedGgufIsRejectedBeforeReplacingTheActiveModel() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val parakeet = sparseModelFile(directory, "parakeet.gguf.source", parakeetPayload())
            val llama = sparseModelFile(directory, "llama.gguf.source", llamaPayload())
            engine.importModel(parakeet.inputStream()).let {
                assertTrue("seed import must succeed: $it", it is OnDeviceEngine.ImportResult.Imported)
            }
            val before = prefix(File(directory, "parakeet.gguf"), 256).copyOfRange(0, 64)
            val beforeLength = File(directory, "parakeet.gguf").length()

            val result = engine.importModel(llama.inputStream())

            assertTrue("unrelated GGUF must be rejected: $result", result is OnDeviceEngine.ImportResult.Rejected)
            val model = File(directory, "parakeet.gguf")
            assertTrue("the previous model must remain", model.isFile)
            assertEquals("the previous model must be untouched", beforeLength, model.length())
            assertTrue("the previous model bytes must be unchanged", before.contentEquals(prefix(model, 64)))
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun truncatedGgufMetadataIsRejected() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val truncated = sparseModelFile(
                directory,
                "truncated.gguf",
                truncatedGgufBytes(claimedKv = 3, presentKv = listOf("general.architecture" to "parakeet")),
            )

            val result = engine.importModel(truncated.inputStream())

            assertTrue("truncated metadata must be rejected: $result", result is OnDeviceEngine.ImportResult.Rejected)
            assertFalse("no model may be published from a corrupt file", File(directory, "parakeet.gguf").exists())
        } finally {
            directory.deleteRecursively()
        }
    }

    /**
     * The interrupted-download shape past the metadata: a parakeet-family
     * GGUF whose tensor records claim a data section the file does not hold.
     * The engine's own load would refuse it, so the import must refuse it
     * first — at VALIDATE, before the last usable model is replaced.
     */
    @Test
    fun dataSectionTruncationIsRejectedBeforeReplacingTheActiveModel() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val parakeet = sparseModelFile(directory, "parakeet.gguf.source", parakeetPayload())
            engine.importModel(parakeet.inputStream())
            val before = prefix(File(directory, "parakeet.gguf"), 256)

            // One F32 tensor of 50M elements claims a 200 MB data section in
            // a 50 MB file.
            val lying = ByteArrayOutputStream()
            lying.write("GGUF".toByteArray(Charsets.US_ASCII))
            lying.write(u32(2))
            lying.write(u64(1)) // one tensor
            lying.write(u64(1)) // one KV
            lying.write(u64("general.architecture".length.toLong()))
            lying.write("general.architecture".toByteArray(Charsets.UTF_8))
            lying.write(u32(8))
            lying.write(u64("parakeet".length.toLong()))
            lying.write("parakeet".toByteArray(Charsets.UTF_8))
            lying.write(u64("encoder.weight".length.toLong()))
            lying.write("encoder.weight".toByteArray(Charsets.UTF_8))
            lying.write(u32(1)) // n_dims
            lying.write(u64(50_000_000))
            lying.write(u32(0)) // F32
            lying.write(u64(0)) // offset
            val truncated = sparseModelFile(directory, "lying.gguf.source", lying.toByteArray())

            val result = engine.importModel(truncated.inputStream())

            val rejected = result as OnDeviceEngine.ImportResult.Rejected
            assertEquals(OnDeviceEngine.ImportStage.VALIDATE, rejected.stage)
            val model = File(directory, "parakeet.gguf")
            assertTrue("the previous model must survive", model.isFile)
            assertTrue("the previous model must be byte-identical", before.contentEquals(prefix(model, before.size)))
            assertEquals(
                "no staging files may remain",
                emptyList<File>(),
                directory.listFiles { f -> f.name.endsWith(".importing") }!!.toList(),
            )
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun failedImportSweepsStaleStagingFiles() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val staleUnique = File(directory, "parakeet.gguf.6f1c9af2.importing").apply { writeBytes(byteArrayOf(1)) }
            val staleLegacy = File(directory, "parakeet.gguf.importing").apply { writeBytes(byteArrayOf(2)) }

            val result = engine.importModel("XXXX-not-a-gguf".toByteArray().inputStream())

            assertTrue(result is OnDeviceEngine.ImportResult.Rejected)
            assertFalse("stale unique staging file must be swept", staleUnique.exists())
            assertFalse("legacy fixed-name staging file must be swept", staleLegacy.exists())
            assertEquals(
                "no staging files may remain after a failed import",
                emptyList<File>(),
                directory.listFiles { f -> f.name.endsWith(".importing") }!!.toList(),
            )
        } finally {
            directory.deleteRecursively()
        }
    }

    // ---- promotion / rollback (via the injected promotion seam) ------------

    @Test
    fun failedPromotionKeepsThePreviousModelUsable() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val parakeet = sparseModelFile(directory, "parakeet.gguf.source", parakeetPayload())
            engine.importModel(parakeet.inputStream())
            val before = prefix(File(directory, "parakeet.gguf"), 256)

            val replacement = sparseModelFile(
                directory,
                "replacement.gguf.source",
                ggufBytes("general.architecture" to "parakeet", "parakeet.vocab_size" to "8193"),
            )
            val result = engine.importModel(replacement.inputStream()) { _, _ -> false }

            val rejected = result as OnDeviceEngine.ImportResult.Rejected
            assertEquals(OnDeviceEngine.ImportStage.PROMOTE, rejected.stage)
            val model = File(directory, "parakeet.gguf")
            assertTrue("the previous model must survive a failed promotion", model.isFile)
            assertTrue("the previous model must be byte-identical", before.contentEquals(prefix(model, before.size)))
            assertTrue("the engine still reports a usable model", engine.hasModel())
            assertEquals(
                "a failed promotion must leave no staging file",
                emptyList<File>(),
                directory.listFiles { f -> f.name.endsWith(".importing") }!!.toList(),
            )
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun copyFailureKeepsThePreviousModelAndCleansStaging() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val parakeet = sparseModelFile(directory, "parakeet.gguf.source", parakeetPayload())
            engine.importModel(parakeet.inputStream())
            val before = prefix(File(directory, "parakeet.gguf"), 256)

            // Fails mid-copy (the same recovery path a failed fsync takes:
            // both are I/O failures inside the copy stage).
            val broken = object : InputStream() {
                var served = 0
                override fun read(): Int {
                    if (served++ >= 1024) throw java.io.IOException("injected copy failure")
                    return 0
                }
            }
            val result = engine.importModel(broken)

            val rejected = result as OnDeviceEngine.ImportResult.Rejected
            assertEquals(OnDeviceEngine.ImportStage.COPY, rejected.stage)
            val model = File(directory, "parakeet.gguf")
            assertTrue("the previous model must survive a failed copy", model.isFile)
            assertTrue("the previous model must be byte-identical", before.contentEquals(prefix(model, before.size)))
            assertEquals(
                "a failed copy must leave no staging file",
                emptyList<File>(),
                directory.listFiles { f -> f.name.endsWith(".importing") }!!.toList(),
            )
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun validationFailureReportsTheValidateStage() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val truncated = sparseModelFile(
                directory,
                "truncated.gguf",
                truncatedGgufBytes(claimedKv = 3, presentKv = listOf("general.architecture" to "parakeet")),
            )

            val rejected = engine.importModel(truncated.inputStream()) as OnDeviceEngine.ImportResult.Rejected

            assertEquals(OnDeviceEngine.ImportStage.VALIDATE, rejected.stage)
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun permissionFailureOnPromotionKeepsThePreviousModel() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val parakeet = sparseModelFile(directory, "parakeet.gguf.source", parakeetPayload())
            engine.importModel(parakeet.inputStream())
            val before = prefix(File(directory, "parakeet.gguf"), 256)

            val probe = File(directory, "probe.tmp").apply { writeBytes(byteArrayOf(1)) }
            val replacement = sparseModelFile(
                directory,
                "replacement.gguf.source",
                ggufBytes("general.architecture" to "parakeet", "parakeet.vocab_size" to "8193"),
            )
            directory.setWritable(false)
            val canaryFails = !probe.renameTo(File(directory, "probe2.tmp"))
            org.junit.Assume.assumeTrue("rename unexpectedly permitted (running as root?)", canaryFails)

            val result = engine.importModel(replacement.inputStream())

            directory.setWritable(true)
            val rejected = result as OnDeviceEngine.ImportResult.Rejected
            // A read-only directory can fail either at staging (COPY) or at
            // the rename (PROMOTE); both must keep the previous model.
            assertTrue(
                "unexpected stage ${rejected.stage}",
                rejected.stage == OnDeviceEngine.ImportStage.COPY ||
                    rejected.stage == OnDeviceEngine.ImportStage.PROMOTE,
            )
            val model = File(directory, "parakeet.gguf")
            assertTrue("the previous model must survive a permission failure", model.isFile)
            assertTrue("the previous model must be byte-identical", before.contentEquals(prefix(model, before.size)))
        } finally {
            directory.setWritable(true)
            directory.deleteRecursively()
        }
    }

    @Test
    fun concurrentImportsSerializeAndPublishExactlyOneModel() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val first = sparseModelFile(directory, "first.gguf.source", parakeetPayload())
            val second = sparseModelFile(
                directory,
                "second.gguf.source",
                ggufBytes(
                    "general.architecture" to "parakeet",
                    "parakeet.preprocessor.normalize" to "per_feature",
                    "parakeet.encoder.conv_norm_type" to "batch_norm",
                ),
            )

            val results = arrayOfNulls<OnDeviceEngine.ImportResult>(2)
            val threads = listOf(first, second).mapIndexed { index, source ->
                Thread { results[index] = engine.importModel(source.inputStream()) }
            }
            threads.forEach { it.start() }
            threads.forEach { it.join(30_000) }

            // Both imports must have fully succeeded — this only holds when
            // the import lock actually serializes them; if they raced, one
            // could fail (its staging file swept mid-copy by the other).
            for ((index, result) in results.withIndex()) {
                assertTrue(
                    "import ${if (index == 0) "first" else "second"} must succeed: $result",
                    result is OnDeviceEngine.ImportResult.Imported,
                )
            }

            val model = File(directory, "parakeet.gguf")
            assertTrue("exactly one model must be published", model.isFile)
            val published = prefix(model, 256)
            val matchesFirst = published.contentEquals(prefix(first, 256))
            val matchesSecond = published.contentEquals(prefix(second, 256))
            assertTrue(
                "the published model must be exactly one of the two imports",
                matchesFirst || matchesSecond,
            )
            assertEquals(
                "no staging files may survive concurrent imports",
                emptyList<File>(),
                directory.listFiles { f -> f.name.endsWith(".importing") }!!.toList(),
            )
        } finally {
            directory.deleteRecursively()
        }
    }

    // ---- promotion mechanics (Files.move ATOMIC_MOVE + fallback) ---------

    @Test
    fun atomicPromotionReplacesAnExistingModel() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val target = File(directory, "parakeet.gguf").apply { writeBytes(ByteArray(64) { 1 }) }
            val staged = File(directory, "parakeet.gguf.seed.importing").apply { writeBytes(ByteArray(64) { 2 }) }

            val promoted = engine.promoteByMove(staged, target)

            assertTrue("Files.move ATOMIC_MOVE must replace the existing target", promoted)
            assertTrue("the target now holds the staged bytes", target.readBytes().all { it == 2.toByte() })
            assertFalse("the staged file is gone after an atomic rename", staged.exists())
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun asideFallbackMovesTheStagedFileInAndDropsTheAside() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val target = File(directory, "parakeet.gguf").apply { writeBytes(ByteArray(64) { 1 }) }
            val staged = File(directory, "parakeet.gguf.seed.importing").apply { writeBytes(ByteArray(64) { 2 }) }

            val promoted = engine.moveAsideFirst(staged, target)

            assertTrue(promoted)
            assertTrue("the target now holds the staged bytes", target.readBytes().all { it == 2.toByte() })
            assertFalse("the aside backup must not linger", File(directory, "parakeet.gguf.previous").exists())
            assertFalse(staged.exists())
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun asideFallbackRestoresThePreviousModelWhenTheMoveInFails() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val target = File(directory, "parakeet.gguf").apply { writeBytes(ByteArray(64) { 1 }) }
            // A staged file that vanished underneath the import fails the
            // move-in step after the aside step already succeeded.
            val staged = File(directory, "parakeet.gguf.seed.importing")

            val promoted = engine.moveAsideFirst(staged, target)

            assertFalse(promoted)
            assertTrue(
                "the previous model must be restored byte-identical",
                target.readBytes().all { it == 1.toByte() },
            )
            assertFalse("no aside backup may linger after the restore", File(directory, "parakeet.gguf.previous").exists())
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun unexpectedFailureAfterCopySweepsTheStagedFileAndKeepsTheModel() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val parakeet = sparseModelFile(directory, "parakeet.gguf.source", parakeetPayload())
            engine.importModel(parakeet.inputStream())
            val before = prefix(File(directory, "parakeet.gguf"), 256)
            val replacement = sparseModelFile(
                directory,
                "replacement.gguf.source",
                ggufBytes("general.architecture" to "parakeet", "parakeet.vocab_size" to "8193"),
            )

            val thrown = runCatching {
                engine.importModel(replacement.inputStream()) { _, _ -> error("injected promote failure") }
            }.exceptionOrNull()

            assertTrue("the unexpected failure must propagate", thrown is IllegalStateException)
            val model = File(directory, "parakeet.gguf")
            assertTrue("the previous model must survive", model.isFile)
            assertTrue("the previous model must be byte-identical", before.contentEquals(prefix(model, before.size)))
            assertEquals(
                "an unexpected failure must still leave no staging file",
                emptyList<File>(),
                directory.listFiles { f -> f.name.endsWith(".importing") }!!.toList(),
            )
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun verifiedDownloadIsAdoptedWithoutACopy() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val download = sparseModelFile(directory, "download-test.part", parakeetPayload())
            val size = download.length()

            val result = engine.adoptDownloaded(download)

            assertTrue("a valid download must be imported: $result", result is OnDeviceEngine.ImportResult.Imported)
            assertTrue(engine.hasModel())
            assertEquals(size, File(directory, "parakeet.gguf").length())
            assertFalse("the download file is consumed", download.exists())
            assertTrue("no staging file may remain", directory.listFiles()!!.none { it.name.endsWith(".importing") })
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun unrelatedDownloadIsRejectedAndDiscarded() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val download = sparseModelFile(directory, "download-test.part", llamaPayload())

            val result = engine.adoptDownloaded(download)

            assertTrue("an unrelated GGUF must be rejected: $result", result is OnDeviceEngine.ImportResult.Rejected)
            assertFalse(engine.hasModel())
            assertFalse("a rejected download is discarded", download.exists())
            assertTrue("no staging file may remain", directory.listFiles()!!.none { it.name.endsWith(".importing") })
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun downloadOutsideTheModelDirectoryIsRefused() {
        val directory = tempDir()
        val elsewhere = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val download = sparseModelFile(elsewhere, "download-test.part", parakeetPayload())

            val result = engine.adoptDownloaded(download)

            assertTrue(result is OnDeviceEngine.ImportResult.Rejected)
            assertFalse(engine.hasModel())
            assertTrue("a file the engine does not own is left alone", download.exists())
        } finally {
            directory.deleteRecursively()
            elsewhere.deleteRecursively()
        }
    }

    @Test
    fun importsKeepTheirNamesAndTheLatestIsActive() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val source = sparseModelFile(directory, "source.part", parakeetPayload())

            engine.importModel(source.inputStream(), "first.gguf")
            val result = engine.importModel(source.inputStream(), "second.gguf")

            assertEquals(OnDeviceEngine.ImportResult.Imported(source.length(), "second.gguf"), result)
            assertEquals(listOf("first.gguf", "second.gguf"), engine.installedModels().map { it.name })
            assertEquals("second.gguf", engine.activeModelName())
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun selectionPersistsAcrossEngineInstances() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val source = sparseModelFile(directory, "source.part", parakeetPayload())
            engine.importModel(source.inputStream(), "first.gguf")
            engine.importModel(source.inputStream(), "second.gguf")

            assertTrue(engine.selectModel("first.gguf"))
            assertFalse("an unknown model cannot be selected", engine.selectModel("missing.gguf"))

            assertEquals("first.gguf", OnDeviceEngine(directory).activeModelName())
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun deletingTheActiveModelFallsBackToARemainingOne() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val source = sparseModelFile(directory, "source.part", parakeetPayload())
            engine.importModel(source.inputStream(), "first.gguf")
            engine.importModel(source.inputStream(), "second.gguf")

            assertTrue(engine.deleteModel("second.gguf"))
            assertEquals("first.gguf", engine.activeModelName())
            assertTrue(engine.deleteModel("first.gguf"))
            assertFalse(engine.hasModel())
            assertFalse("only installed models can be deleted", engine.deleteModel("source.part"))
            assertTrue(File(directory, "source.part").exists())
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun legacySingleModelIsStillTheActiveOne() {
        val directory = tempDir()
        try {
            sparseModelFile(directory, "parakeet.gguf", parakeetPayload())

            val engine = OnDeviceEngine(directory)

            assertEquals("parakeet.gguf", engine.activeModelName())
        } finally {
            directory.deleteRecursively()
        }
    }

    private fun catalogSpec(file: File, sha256: String = ModelDownloader.sha256(file)) = ModelDownload(
        url = "https://example.invalid/models/recommended.gguf",
        sizeBytes = file.length(),
        sha256 = sha256,
        label = "Recommended",
    )

    @Test
    fun legacyCopyOfTheCatalogModelGetsItsCatalogName() {
        val directory = tempDir()
        try {
            val legacy = sparseModelFile(directory, "parakeet.gguf", parakeetPayload())
            val engine = OnDeviceEngine(directory)

            assertTrue(engine.recognizeLegacyDownload(catalogSpec(legacy)))

            assertEquals("recommended.gguf", engine.activeModelName())
            assertEquals(listOf("recommended.gguf"), engine.installedModels().map { it.name })
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun legacyModelWithOtherBytesKeepsItsName() {
        val directory = tempDir()
        try {
            val legacy = sparseModelFile(directory, "parakeet.gguf", parakeetPayload())
            val engine = OnDeviceEngine(directory)

            assertFalse("a digest mismatch is not the catalog model", engine.recognizeLegacyDownload(catalogSpec(legacy, "0".repeat(64))))
            assertFalse(
                "a size mismatch is not even hashed",
                engine.recognizeLegacyDownload(catalogSpec(legacy).copy(sizeBytes = legacy.length() + 1)),
            )

            assertEquals(listOf("parakeet.gguf"), engine.installedModels().map { it.name })
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun downloadIsInstalledUnderItsCatalogName() {
        val directory = tempDir()
        try {
            val engine = OnDeviceEngine(directory)
            val download = sparseModelFile(directory, "download-test.part", parakeetPayload())

            engine.adoptDownloaded(download, ModelCatalog.RECOMMENDED_PARAKEET.fileName)

            assertEquals(ModelCatalog.RECOMMENDED_PARAKEET.fileName, engine.activeModelName())
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun importNamesAreSanitized() {
        assertEquals("model.gguf", OnDeviceEngine.sanitizeModelName("../x/model.gguf"))
        assertEquals("my_model_v2.gguf", OnDeviceEngine.sanitizeModelName("my model v2"))
        assertEquals("hidden.gguf", OnDeviceEngine.sanitizeModelName(".hidden.gguf"))
        assertEquals("parakeet.gguf", OnDeviceEngine.sanitizeModelName(null))
        assertEquals("parakeet.gguf", OnDeviceEngine.sanitizeModelName("..."))
        assertEquals("parakeet.gguf", OnDeviceEngine.sanitizeModelName(".gguf"))
        assertEquals("a.gguf.importing.gguf", OnDeviceEngine.sanitizeModelName("a.gguf.importing"))
    }
}
