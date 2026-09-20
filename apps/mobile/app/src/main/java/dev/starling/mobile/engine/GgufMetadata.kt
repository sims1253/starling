package dev.starling.mobile.engine

import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.EOFException
import java.io.File
import java.io.FileInputStream
import java.io.IOException

/**
 * Bounded reader for the metadata (key-value) section of a GGUF file, used
 * to validate a staged model before it is allowed to replace the active one
 * (B08). It mirrors the acceptance rules of the engine's own reader
 * (third_party/ggml/src/gguf.cpp) closely enough to reject what the engine
 * would reject, while never reading more than [MAX_METADATA_BYTES]:
 *
 * - magic "GGUF", little-endian, version 2 or 3 (the engine's ggml rejects
 *   v1 and anything newer than it knows);
 * - scalar lengths and counts must be positive and fit the read budget;
 * - empty and duplicate keys, unknown value types, and nested arrays are
 *   malformed (the engine rejects all three);
 * - a truncated section (claims more pairs than the file holds) is malformed.
 *
 * Only the value types needed for metadata inspection are decoded (strings);
 * everything else is skipped under the same budget. Of the decoded strings
 * only [ARCHITECTURE_KEY] is retained — the sole value any consumer reads;
 * tokenizer/chat-template blobs can fill the whole budget and must not be
 * held in memory for the lifetime of the parse result.
 */
object GgufMetadata {
    /** GGUF metadata is a few hundred KB at most (tokenizer pieces); 4 MiB bounds crafted files. */
    const val MAX_METADATA_BYTES = 4L * 1024 * 1024

    /** Bounds pathological KV counts before any per-pair work. */
    const val MAX_KV_COUNT = 1_000_000L

    private const val TYPE_STRING = 8L
    private const val TYPE_ARRAY = 9L

    /** The one string value retained from the KV section (the rejection message's model family). */
    private const val ARCHITECTURE_KEY = "general.architecture"

    private val SCALAR_SIZES = longArrayOf(1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8)

    data class Parsed(
        val version: Int,
        val tensorCount: Long,
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
            var scratch = ByteArray(64)
            for (index in 0 until kvCount) {
                val key = reader.readUtf8String(scratch)
                scratch = key.first
                if (key.second.isEmpty() || !keys.add(key.second)) return null
                // Types are compared as the unsigned u32 Long before any
                // toInt() narrowing: values >= 0x80000000 narrow to negative
                // Ints, and the range check below must catch them.
                when (val type = reader.readU32()) {
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
                        reader.skipFully(SCALAR_SIZES[type.toInt()])
                    }
                }
            }
            Parsed(version, tensorCount, keys.toList(), strings)
        }
    } catch (_: EOFException) {
        null
    } catch (_: IOException) {
        null
    }

    /**
     * Little-endian reader with a hard byte budget: any read that would
     * exceed [remaining] bytes fails as truncated instead of trusting the
     * file's self-described sizes. Scratch buffers are per-reader fields —
     * the KV loop runs up to [MAX_KV_COUNT] iterations and must not allocate
     * per read.
     */
    private class BudgetedReader(private val input: DataInputStream, budget: Long) {
        private var remainingBytes = budget
        private val scratch4 = ByteArray(4)
        private val scratch8 = ByteArray(8)

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
