package dev.starling.mobile.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
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
}
