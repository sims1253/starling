package dev.starling.mobile.engine

import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.EOFException
import java.io.File
import java.io.FileInputStream
import java.io.IOException

/**
 * Bounded reader for the metadata (key-value) and tensor-info sections of a
 * GGUF file, used to validate a staged model before it is allowed to
 * replace the active one (B08). It mirrors the acceptance rules of the
 * engine's own reader (third_party/ggml/src/gguf.cpp) closely enough to
 * reject what the engine would reject — including every tensor-info record
 * and the sized, aligned data section, so a truncated download cannot
 * replace the last usable model with a file the engine then refuses to
 * load — while never reading more than [MAX_METADATA_BYTES]:
 *
 * - magic "GGUF", little-endian, version 2 or 3 (the engine's ggml rejects
 *   v1 and anything newer than it knows);
 * - scalar lengths and counts must be positive and fit the read budget;
 * - empty and duplicate keys, unknown value types, and nested arrays are
 *   malformed (the engine rejects all three);
 * - a truncated section (claims more pairs or tensors than the file holds)
 *   is malformed;
 * - general.alignment must be a u32 power of two (engine default 32);
 * - tensor names must be unique and shorter than ggml's GGML_MAX_NAME,
 *   with at most GGML_MAX_DIMS non-negative dimensions, a known type whose
 *   block size divides the row size, and an offset equal to the running
 *   GGML_PAD(nbytes, alignment) total;
 * - the file must actually hold the whole aligned data section.
 *
 * Only the value types needed for metadata inspection are decoded (strings);
 * everything else is skipped under the same budget. Of the decoded strings
 * only [ARCHITECTURE_KEY] is retained — the sole value any consumer reads;
 * tokenizer/chat-template blobs can fill the whole budget and must not be
 * held in memory for the lifetime of the parse result.
 */
object GgufMetadata {
    /** GGUF metadata + tensor infos are a few hundred KB at most (tokenizer pieces); 4 MiB bounds crafted files. */
    const val MAX_METADATA_BYTES = 4L * 1024 * 1024

    /** Bounds pathological KV counts before any per-pair work. */
    const val MAX_KV_COUNT = 1_000_000L

    private const val TYPE_UINT32 = 4L
    private const val TYPE_STRING = 8L
    private const val TYPE_ARRAY = 9L

    /** The one string value retained from the KV section (the rejection message's model family). */
    private const val ARCHITECTURE_KEY = "general.architecture"

    /** The engine sizes the data section on this KV when present; it must be u32. */
    private const val ALIGNMENT_KEY = "general.alignment"

    // Engine constants at the pinned ggml (third_party/ggml @ e91ded1).
    private const val GGML_TYPE_COUNT = 43
    private const val GGML_MAX_NAME = 64
    private const val GGML_MAX_DIMS = 4
    private const val GGUF_DEFAULT_ALIGNMENT = 32L

    /**
     * (block size, type size) per ggml type, mirroring ggml.c's type_traits
     * table at the pinned revision — including the removed types, whose
     * zero block size the engine itself refuses. Needed to compute each
     * tensor's byte size exactly as the engine does.
     */
    private val GGML_TYPES = longArrayOf(
        1, 4, // 0  F32
        1, 2, // 1  F16
        32, 18, // 2  Q4_0
        32, 20, // 3  Q4_1
        0, 0, // 4  (removed)
        0, 0, // 5  (removed)
        32, 22, // 6  Q5_0
        32, 24, // 7  Q5_1
        32, 34, // 8  Q8_0
        32, 36, // 9  Q8_1
        256, 84, // 10 Q2_K
        256, 110, // 11 Q3_K
        256, 144, // 12 Q4_K
        256, 176, // 13 Q5_K
        256, 210, // 14 Q6_K
        256, 292, // 15 Q8_K
        256, 66, // 16 IQ2_XXS
        256, 74, // 17 IQ2_XS
        256, 98, // 18 IQ3_XXS
        256, 50, // 19 IQ1_S
        256, 18, // 20 IQ4_NL
        256, 110, // 21 IQ3_S
        256, 82, // 22 IQ2_S
        256, 136, // 23 IQ4_XS
        1, 1, // 24 I8
        1, 2, // 25 I16
        1, 4, // 26 I32
        1, 8, // 27 I64
        1, 8, // 28 F64
        256, 56, // 29 IQ1_M
        1, 2, // 30 BF16
        0, 0, // 31 (removed)
        0, 0, // 32 (removed)
        0, 0, // 33 (removed)
        256, 54, // 34 TQ1_0
        256, 66, // 35 TQ2_0
        0, 0, // 36 (removed)
        0, 0, // 37 (removed)
        0, 0, // 38 (removed)
        32, 17, // 39 MXFP4
        64, 36, // 40 NVFP4
        128, 18, // 41 Q1_0
        64, 18, // 42 Q2_0
    )

    private val SCALAR_SIZES = longArrayOf(1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8)

    data class Parsed(
        /** All metadata keys, in file order. */
        val keys: List<String>,
        /** The retained string value: [ARCHITECTURE_KEY], when present. */
        val strings: Map<String, String>,
    )

    /** Null when [file]'s metadata section is malformed, unsupported, or over budget. */
    fun parse(file: File): Parsed? = try {
        DataInputStream(BufferedInputStream(FileInputStream(file))).use { input ->
            val reader = BudgetedReader(input, MAX_METADATA_BYTES)
            val magic = ByteArray(4)
            reader.readFully(magic)
            if (!magic.contentEquals("GGUF".toByteArray(Charsets.US_ASCII))) return null
            val version = reader.readU32().toInt()
            if (version !in 2..3) return null
            val tensorCount = reader.readU64()
            if (tensorCount < 0) return null
            val kvCount = reader.readU64()
            if (kvCount < 0 || kvCount > MAX_KV_COUNT) return null

            val keys = LinkedHashSet<String>()
            val strings = LinkedHashMap<String, String>()
            var alignment = GGUF_DEFAULT_ALIGNMENT
            var scratch = ByteArray(64)
            for (index in 0 until kvCount) {
                val key = reader.readUtf8String(scratch)
                scratch = key.first
                if (key.second.isEmpty() || !keys.add(key.second)) return null
                // Types are compared as the unsigned u32 Long before any
                // toInt() narrowing: values >= 0x80000000 narrow to negative
                // Ints, and the range check below must catch them.
                val type = reader.readU32()
                // The engine reads general.alignment as u32 and refuses any
                // other type for it, whatever the declared value.
                if (key.second == ALIGNMENT_KEY && type != TYPE_UINT32) return null
                when (type) {
                    TYPE_STRING -> {
                        val value = reader.readUtf8String(scratch)
                        scratch = value.first
                        if (key.second == ARCHITECTURE_KEY) strings[key.second] = value.second
                    }
                    TYPE_ARRAY -> {
                        val elementType = reader.readU32()
                        if (elementType >= SCALAR_SIZES.size || elementType == TYPE_ARRAY) return null
                        val count = reader.readU64()
                        if (count < 0) return null
                        if (elementType == TYPE_STRING) {
                            // Each element costs at least its u64 length field:
                            // reject padded huge counts in O(1) instead of
                            // skipping them one by one (the Int bound closes
                            // the wraparound hole before repeat()).
                            if (count > Int.MAX_VALUE || count > reader.remaining() / 8) return null
                            repeat(count.toInt()) {
                                val element = reader.readUtf8String(scratch)
                                // Reuse the largest buffer seen so a crafted
                                // token array churns at most one allocation.
                                scratch = element.first
                            }
                        } else {
                            val elementSize = SCALAR_SIZES[elementType.toInt()]
                            if (count > reader.remaining() / elementSize) return null
                            reader.skipFully(elementSize * count)
                        }
                    }
                    else -> {
                        if (type >= SCALAR_SIZES.size) return null
                        if (key.second == ALIGNMENT_KEY) {
                            // Nonzero power of two, as the engine requires.
                            alignment = reader.readU32()
                            if (alignment == 0L || alignment and (alignment - 1) != 0L) return null
                        } else {
                            reader.skipFully(SCALAR_SIZES[type.toInt()])
                        }
                    }
                }
            }

            // Tensor-info section: the engine parses every record and sizes
            // the aligned data section before it will load a model, so this
            // must reject here — before the file can replace the last usable
            // model — everything the engine's load would refuse.
            val tensorNames = HashSet<String>()
            var dataBytes = 0L
            for (index in 0 until tensorCount) {
                if (!tensorNames.add(reader.readTensorName())) return null
                val nDims = reader.readU32()
                if (nDims > GGML_MAX_DIMS) return null
                val dims = longArrayOf(1L, 1L, 1L, 1L)
                for (dimension in 0 until nDims.toInt()) {
                    val extent = reader.readU64()
                    if (extent < 0) return null
                    dims[dimension] = extent
                }
                val tensorType = reader.readU32()
                if (tensorType >= GGML_TYPE_COUNT) return null
                val traits = tensorType.toInt() * 2
                val blockSize = GGML_TYPES[traits]
                val typeSize = GGML_TYPES[traits + 1]
                // A block size of 0 is a removed type; the engine refuses
                // those and any row that is not a whole number of blocks.
                if (blockSize == 0L || dims[0] % blockSize != 0L) return null
                // nbytes with the engine's shape (undeclared dims are 1);
                // exact math so an unrepresentable size is rejected.
                val tensorBytes =
                    productExact(typeSize, dims[0] / blockSize, dims[1], dims[2], dims[3]) ?: return null
                val offset = reader.readU64()
                if (offset < 0 || offset != dataBytes) return null
                dataBytes = padExact(tensorBytes, alignment)?.let { sumExact(dataBytes, it) } ?: return null
            }
            if (tensorCount > 0) {
                // The data section starts at the next alignment boundary
                // after the tensor infos and the file must hold all of it —
                // the engine reads the whole blob (truncation fails there).
                val dataStart = padExact(reader.consumed(), alignment) ?: return null
                val requiredBytes = sumExact(dataStart, dataBytes) ?: return null
                if (file.length() < requiredBytes) return null
            }
            Parsed(keys.toList(), strings)
        }
    } catch (_: EOFException) {
        null
    } catch (_: IOException) {
        null
    }

    /** ggml's GGML_PAD: [value] rounded up to a multiple of [alignment]; null on overflow. */
    private fun padExact(value: Long, alignment: Long): Long? = runCatching {
        Math.multiplyExact(Math.addExact(value, alignment - 1) / alignment, alignment)
    }.getOrNull()

    private fun productExact(vararg factors: Long): Long? = runCatching {
        var product = 1L
        for (factor in factors) product = Math.multiplyExact(product, factor)
        product
    }.getOrNull()

    private fun sumExact(left: Long, right: Long): Long? = runCatching { Math.addExact(left, right) }.getOrNull()

    /**
     * Little-endian reader with a hard byte budget: any read that would
     * exceed [remaining] bytes fails as truncated instead of trusting the
     * file's self-described sizes. Scratch buffers are per-reader fields —
     * the KV loop runs up to [MAX_KV_COUNT] iterations and must not allocate
     * per read.
     */
    private class BudgetedReader(private val input: DataInputStream, budget: Long) {
        private var remainingBytes = budget
        private val startBudget = budget
        private val scratch4 = ByteArray(4)
        private val scratch8 = ByteArray(8)
        private val nameScratch = ByteArray(GGML_MAX_NAME)

        /** File bytes consumed so far (the read position; the data section starts after any padding). */
        fun consumed(): Long = startBudget - remainingBytes

        fun remaining(): Long = remainingBytes

        private fun take(count: Long) {
            if (count > remainingBytes) throw EOFException("metadata read exceeds budget")
            remainingBytes -= count
        }

        fun readFully(buffer: ByteArray) {
            take(buffer.size.toLong())
            input.readFully(buffer)
        }

        fun readU32(): Long {
            readFully(scratch4)
            return (scratch4[0].toLong() and 0xff) or
                ((scratch4[1].toLong() and 0xff) shl 8) or
                ((scratch4[2].toLong() and 0xff) shl 16) or
                ((scratch4[3].toLong() and 0xff) shl 24)
        }

        fun readU64(): Long {
            readFully(scratch8)
            var value = 0L
            for (i in 7 downTo 0) value = (value shl 8) or (scratch8[i].toLong() and 0xff)
            return value
        }

        /**
         * Reads a GGUF string (u64 length + UTF-8 bytes); returns the
         * (possibly grown) scratch buffer and the value. The length is
         * checked against the budget BEFORE the buffer allocation on
         * purpose: an adversarial length must fail without first
         * materializing a ByteArray of that size.
         */
        fun readUtf8String(scratch: ByteArray): Pair<ByteArray, String> {
            val length = readU64()
            if (length < 0 || length > remainingBytes) throw EOFException("string length exceeds budget")
            val buffer = if (length <= scratch.size) scratch else ByteArray(length.toInt())
            if (length > 0) {
                take(length)
                input.readFully(buffer, 0, length.toInt())
            }
            return buffer to String(buffer, 0, length.toInt(), Charsets.UTF_8)
        }

        /**
         * Reads a tensor name (u64 length + UTF-8 bytes). A length at or
         * above ggml's GGML_MAX_NAME is malformed — the engine refuses it —
         * and never allocates: names fit the fixed scratch.
         */
        fun readTensorName(): String {
            val length = readU64()
            if (length < 0 || length >= GGML_MAX_NAME || length > remainingBytes) {
                throw EOFException("tensor name length out of range")
            }
            if (length == 0L) return ""
            take(length)
            input.readFully(nameScratch, 0, length.toInt())
            return String(nameScratch, 0, length.toInt(), Charsets.UTF_8)
        }

        fun skipFully(count: Long) {
            var left = count
            take(left)
            while (left > 0) {
                val skipped = input.skip(left)
                if (skipped <= 0) {
                    if (input.read() == -1) throw EOFException("unexpected end of metadata")
                    left -= 1
                } else {
                    left -= skipped
                }
            }
        }
    }
}
