package dev.starling.mobile.engine

import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

/** Bounded GGUF metadata-section parsing (B08 pre-promotion validation). */
class GgufMetadataTest {
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

    private fun fileOf(bytes: ByteArray): File {
        val file = Files.createTempDirectory("starling-gguf-meta").resolve("model.gguf").toFile()
        file.writeBytes(bytes)
        return file
    }

    private fun header(version: Int = 2, tensors: Long = 0, kvs: Long): ByteArrayOutputStream {
        val out = ByteArrayOutputStream()
        out.write("GGUF".toByteArray(Charsets.US_ASCII))
        out.write(u32(version.toLong()))
        out.write(u64(tensors))
        out.write(u64(kvs))
        return out
    }

    private fun stringKv(out: ByteArrayOutputStream, key: String, value: String) {
        out.write(u64(key.length.toLong()))
        out.write(key.toByteArray(Charsets.UTF_8))
        out.write(u32(8))
        out.write(u64(value.length.toLong()))
        out.write(value.toByteArray(Charsets.UTF_8))
    }

    @Test
    fun parsesVersionTwoAndThreeWithKeysAndStrings() {
        for (version in intArrayOf(2, 3)) {
            val out = header(version, kvs = 2)
            stringKv(out, "general.architecture", "parakeet")
            stringKv(out, "parakeet.preprocessor.normalize", "per_feature")
            val parsed = GgufMetadata.parse(fileOf(out.toByteArray()))

            assertNotNull("version $version must parse", parsed)
            assertEquals(version, parsed!!.version)
            assertEquals(setOf("general.architecture", "parakeet.preprocessor.normalize"), parsed.keys.toSet())
            assertEquals("parakeet", parsed.strings["general.architecture"])
        }
    }

    @Test
    fun skipsScalarAndArrayValuesOfEveryType() {
        val out = header(kvs = 6)
        // u32 scalar (parakeet.vocab_size)
        out.write(u64("parakeet.vocab_size".length.toLong())); out.write("parakeet.vocab_size".toByteArray())
        out.write(u32(4)); out.write(u32(8193))
        // f32 scalar
        out.write(u64("k.f".length.toLong())); out.write("k.f".toByteArray())
        out.write(u32(6)); out.write(u32(0))
        // i64 scalar (8 bytes)
        out.write(u64("k.i".length.toLong())); out.write("k.i".toByteArray())
        out.write(u32(11)); out.write(u64(42))
        // u8 array of 3
        out.write(u64("k.a8".length.toLong())); out.write("k.a8".toByteArray())
        out.write(u32(9)); out.write(u32(0)); out.write(u64(3)); out.write(byteArrayOf(1, 2, 3))
        // f64 array of 2
        out.write(u64("k.a64".length.toLong())); out.write("k.a64".toByteArray())
        out.write(u32(9)); out.write(u32(12)); out.write(u64(2)); out.write(ByteArray(16))
        // string array of 2 (element type, count, then u64-length strings)
        out.write(u64("k.as".length.toLong())); out.write("k.as".toByteArray())
        out.write(u32(9)); out.write(u32(8)); out.write(u64(2))
        out.write(u64(1)); out.write("x".toByteArray())
        out.write(u64(2)); out.write("yy".toByteArray())

        val parsed = GgufMetadata.parse(fileOf(out.toByteArray()))

        assertNotNull(parsed)
        assertEquals(6, parsed!!.keys.size)
    }

    @Test
    fun rejectsWrongMagicVersionsAndCounts() {
        val badMagic = header(kvs = 0).toByteArray().also { it[0] = 'X'.code.toByte() }
        assertNull(GgufMetadata.parse(fileOf(badMagic)))
        assertNull("v1 is rejected by the engine's ggml", GgufMetadata.parse(fileOf(header(version = 1, kvs = 0).toByteArray())))
        assertNull("v4 is newer than the engine knows", GgufMetadata.parse(fileOf(header(version = 4, kvs = 0).toByteArray())))
        assertNull(
            "pathological KV count",
            GgufMetadata.parse(fileOf(header(kvs = GgufMetadata.MAX_KV_COUNT + 1).toByteArray())),
        )
    }

    @Test
    fun rejectsTruncatedSectionsAndUnknownTypes() {
        // Claims 3 pairs, holds 1: the next u64 length reads zeros/EOF.
        val truncated = header(kvs = 3)
        stringKv(truncated, "general.architecture", "parakeet")
        assertNull(GgufMetadata.parse(fileOf(truncated.toByteArray())))

        // Unknown value type 13.
        val unknownType = header(kvs = 1)
        unknownType.write(u64(1)); unknownType.write("k".toByteArray()); unknownType.write(u32(13))
        assertNull(GgufMetadata.parse(fileOf(unknownType.toByteArray())))

        // Nested arrays are invalid.
        val nested = header(kvs = 1)
        nested.write(u64(1)); nested.write("k".toByteArray()); nested.write(u32(9)); nested.write(u32(9))
        assertNull(GgufMetadata.parse(fileOf(nested.toByteArray())))

        // Empty keys are invalid.
        val emptyKey = header(kvs = 1)
        emptyKey.write(u64(0)); emptyKey.write(u32(8)); emptyKey.write(u64(0))
        assertNull(GgufMetadata.parse(fileOf(emptyKey.toByteArray())))

        // Duplicate keys are invalid.
        val duplicate = header(kvs = 2)
        stringKv(duplicate, "k", "a")
        stringKv(duplicate, "k", "b")
        assertNull(GgufMetadata.parse(fileOf(duplicate.toByteArray())))
    }

    @Test
    fun rejectsMetadataOverTheReadBudget() {
        // One string KV whose declared length blows the 4 MiB budget.
        val huge = header(kvs = 1)
        huge.write(u64(1)); huge.write("k".toByteArray())
        huge.write(u32(8)); huge.write(u64(GgufMetadata.MAX_METADATA_BYTES))
        assertNull(GgufMetadata.parse(fileOf(huge.toByteArray())))

        // A u8 array whose declared size blows the budget.
        val hugeArray = header(kvs = 1)
        hugeArray.write(u64(1)); hugeArray.write("k".toByteArray())
        hugeArray.write(u32(9)); hugeArray.write(u32(0)); hugeArray.write(u64(GgufMetadata.MAX_METADATA_BYTES))
        assertNull(GgufMetadata.parse(fileOf(hugeArray.toByteArray())))
    }
}
