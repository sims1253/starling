package dev.starling.mobile.engine

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.RandomAccessFile
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelFilesTest {
    @Test
    fun `accepts a gguf header of sufficient size`() {
        val header = "GGUF".toByteArray() + ByteArray(12)
        assertNull(ModelFiles.validate(ModelFiles.MIN_MODEL_BYTES, header))
    }

    @Test
    fun `rejects a wrong magic`() {
        val header = "GGUF".toByteArray().also { it[0] = 'X'.code.toByte() }
        assertNotNull(ModelFiles.validate(ModelFiles.MIN_MODEL_BYTES, header))
    }

    @Test
    fun `rejects undersized files even with correct magic`() {
        assertNotNull(ModelFiles.validate(1024, "GGUF".toByteArray()))
    }

    // ---- validateStaged (deep, staged-file contract) ----------------------

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

    /** A llama-family GGUF with an [architectureLength]-char architecture tag, sparsely padded to the minimum size. */
    private fun unrelatedGguf(architecture: String): File {
        val out = ByteArrayOutputStream()
        out.write("GGUF".toByteArray(Charsets.US_ASCII))
        out.write(u32(2))
        out.write(u64(0)) // tensor count
        out.write(u64(1)) // one KV
        out.write(u64("general.architecture".length.toLong()))
        out.write("general.architecture".toByteArray(Charsets.UTF_8))
        out.write(u32(8)) // string
        out.write(u64(architecture.length.toLong()))
        out.write(architecture.toByteArray(Charsets.UTF_8))
        val file = Files.createTempDirectory("starling-modelfiles").resolve("staged.gguf").toFile()
        RandomAccessFile(file, "rw").use { handled ->
            handled.write(out.toByteArray())
            handled.setLength(ModelFiles.MIN_MODEL_BYTES)
        }
        return file
    }

    @Test
    fun `caps the architecture echoed into the rejection message`() {
        val longTag = "l".repeat(200)

        val reason = ModelFiles.validateStaged(unrelatedGguf(longTag))

        assertTrue(reason!!.startsWith("This GGUF ("))
        val echoed = reason.removePrefix("This GGUF (").removeSuffix(") is not a Parakeet model.")
        assertEquals(
            "only the first 40 characters of a corruption-controlled tag may reach the message",
            40,
            echoed.length,
        )
        assertTrue(echoed.all { it == 'l' })
    }

    @Test
    fun `short architecture tags are echoed in full`() {
        assertEquals(
            "This GGUF (llama) is not a Parakeet model.",
            ModelFiles.validateStaged(unrelatedGguf("llama")),
        )
    }
}
