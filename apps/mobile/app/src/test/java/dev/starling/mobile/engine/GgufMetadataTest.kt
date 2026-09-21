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
            assertEquals(setOf("general.architecture", "parakeet.preprocessor.normalize"), parsed!!.keys.toSet())
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
    fun retainsOnlyTheArchitectureString() {
        val out = header(kvs = 2)
        stringKv(out, "general.name", "some-chat-model")
        stringKv(out, "general.architecture", "parakeet")

        val parsed = GgufMetadata.parse(fileOf(out.toByteArray()))

        assertNotNull(parsed)
        // Tokenizer/chat-template-scale strings are dropped: only the one
        // consumed value is retained, but every key stays (family check).
        assertEquals(mapOf("general.architecture" to "parakeet"), parsed!!.strings)
        assertEquals(setOf("general.name", "general.architecture"), parsed.keys.toSet())
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

    // ---- tensor-info section (the engine's gguf_init_from_file rules) ----

    private fun tensorInfo(out: ByteArrayOutputStream, name: String, type: Long, offset: Long, vararg dims: Long) {
        out.write(u64(name.length.toLong()))
        out.write(name.toByteArray(Charsets.UTF_8))
        out.write(u32(dims.size.toLong()))
        for (dim in dims) out.write(u64(dim))
        out.write(u32(type))
        out.write(u64(offset))
    }

    /** A minimal parakeet header + one architecture KV + two F32 [100] tensors at engine-true offsets. */
    private fun twoTensorMetadata(): ByteArrayOutputStream {
        val out = header(tensors = 2, kvs = 1)
        stringKv(out, "general.architecture", "parakeet")
        // F32 x100 = 400 bytes; pad(400, 32) = 416 is the second offset.
        tensorInfo(out, "a", type = 0, offset = 0, dims = longArrayOf(100))
        tensorInfo(out, "b", type = 0, offset = 416, dims = longArrayOf(100))
        return out
    }

    private fun sparseFileOf(bytes: ByteArray, length: Long): File {
        val file = Files.createTempDirectory("starling-gguf-meta").resolve("model.gguf").toFile()
        java.io.RandomAccessFile(file, "rw").use { out ->
            out.write(bytes)
            out.setLength(length)
        }
        return file
    }

    @Test
    fun parsesTensorInfosWhenTheDataSectionIsFullyPresent() {
        val metadata = twoTensorMetadata().toByteArray()
        val dataStart = ((metadata.size + 31) / 32) * 32 // GGUF pad to default alignment 32
        // The engine's data section pads EVERY tensor, the last one included:
        // pad(400, 32) + pad(400, 32) = 416 + 416.
        val required = dataStart + 832L

        assertNotNull(GgufMetadata.parse(sparseFileOf(metadata, required)))
        assertNull(
            "a data section truncated by one byte is the interrupted-download shape",
            GgufMetadata.parse(sparseFileOf(metadata, required - 1)),
        )
        assertNull(
            "a file ending right after the tensor infos has no data section",
            GgufMetadata.parse(sparseFileOf(metadata, metadata.size.toLong())),
        )
    }

    @Test
    fun honorsTheAlignmentKVWhenSizingTheDataSection() {
        val out = header(tensors = 2, kvs = 2)
        stringKv(out, "general.architecture", "parakeet")
        // u32-typed general.alignment = 64 (non-default power of two).
        out.write(u64("general.alignment".length.toLong())); out.write("general.alignment".toByteArray())
        out.write(u32(4)); out.write(u32(64))
        // With alignment 64: pad(400, 64) = 448 per tensor, last included.
        tensorInfo(out, "a", type = 0, offset = 0, dims = longArrayOf(100))
        tensorInfo(out, "b", type = 0, offset = 448, dims = longArrayOf(100))
        val metadata = out.toByteArray()
        val dataStart = ((metadata.size + 63) / 64) * 64
        val required = dataStart.toLong() + 448 + 448

        assertNotNull(GgufMetadata.parse(sparseFileOf(metadata, required)))
        assertNull(GgufMetadata.parse(sparseFileOf(metadata, required - 1)))
    }

    @Test
    fun rejectsBadAlignmentKeyValue() {
        // Non-power-of-two alignment.
        val notPow2 = header(tensors = 0, kvs = 1)
        notPow2.write(u64("general.alignment".length.toLong())); notPow2.write("general.alignment".toByteArray())
        notPow2.write(u32(4)); notPow2.write(u32(3))
        assertNull(GgufMetadata.parse(fileOf(notPow2.toByteArray())))

        // Wrongly typed alignment (string instead of u32).
        val wrongType = header(tensors = 0, kvs = 1)
        stringKv(wrongType, "general.alignment", "32")
        assertNull(GgufMetadata.parse(fileOf(wrongType.toByteArray())))
    }

    @Test
    fun rejectsMalformedTensorInfos() {
        fun withTensors(build: (ByteArrayOutputStream) -> Unit): ByteArray {
            val out = header(tensors = 2, kvs = 1)
            stringKv(out, "general.architecture", "parakeet")
            build(out)
            return out.toByteArray()
        }

        assertNull(
            "duplicate tensor names",
            GgufMetadata.parse(
                sparseFileOf(
                    withTensors {
                        tensorInfo(it, "a", type = 0, offset = 0, dims = longArrayOf(100))
                        tensorInfo(it, "a", type = 0, offset = 416, dims = longArrayOf(100))
                    },
                    1 shl 20,
                ),
            ),
        )
        assertNull(
            "tensor name at ggml's GGML_MAX_NAME",
            GgufMetadata.parse(
                sparseFileOf(
                    withTensors {
                        tensorInfo(it, "a", type = 0, offset = 0, dims = longArrayOf(100))
                        tensorInfo(it, "x".repeat(64), type = 0, offset = 416, dims = longArrayOf(100))
                    },
                    1 shl 20,
                ),
            ),
        )
        assertNull(
            "more dimensions than ggml's GGML_MAX_DIMS",
            GgufMetadata.parse(
                sparseFileOf(
                    withTensors {
                        tensorInfo(it, "a", type = 0, offset = 0, dims = longArrayOf(1, 2, 3, 4, 5))
                    },
                    1 shl 20,
                ),
            ),
        )
        assertNull(
            "negative dimension (u64 with the sign bit set)",
            GgufMetadata.parse(
                sparseFileOf(
                    withTensors {
                        tensorInfo(it, "a", type = 0, offset = 0, dims = longArrayOf(-1L))
                    },
                    1 shl 20,
                ),
            ),
        )
        assertNull(
            "type outside [0, GGML_TYPE_COUNT)",
            GgufMetadata.parse(
                sparseFileOf(
                    withTensors {
                        tensorInfo(it, "a", type = 43, offset = 0, dims = longArrayOf(100))
                    },
                    1 shl 20,
                ),
            ),
        )
        assertNull(
            "removed ggml type (block size 0)",
            GgufMetadata.parse(
                sparseFileOf(
                    withTensors {
                        tensorInfo(it, "a", type = 31, offset = 0, dims = longArrayOf(100))
                    },
                    1 shl 20,
                ),
            ),
        )
        assertNull(
            "row size not a multiple of the type's block size (Q4_0, 7 elements)",
            GgufMetadata.parse(
                sparseFileOf(
                    withTensors {
                        tensorInfo(it, "a", type = 2, offset = 0, dims = longArrayOf(7))
                    },
                    1 shl 20,
                ),
            ),
        )
        assertNull(
            "offset not equal to the running padded total",
            GgufMetadata.parse(
                sparseFileOf(
                    withTensors {
                        tensorInfo(it, "a", type = 0, offset = 0, dims = longArrayOf(100))
                        tensorInfo(it, "b", type = 0, offset = 417, dims = longArrayOf(100))
                    },
                    1 shl 20,
                ),
            ),
        )
        assertNull(
            "tensor-info section truncated mid-record",
            GgufMetadata.parse(
                sparseFileOf(
                    withTensors {
                        tensorInfo(it, "a", type = 0, offset = 0, dims = longArrayOf(100))
                        it.write(u64(1)) // second record cut off right after its name length
                    },
                    1 shl 20,
                ),
            ),
        )
    }
}
