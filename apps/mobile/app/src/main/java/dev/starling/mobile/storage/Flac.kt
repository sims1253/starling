package dev.starling.mobile.storage

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.EOFException
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.security.MessageDigest

/**
 * Lossless at-rest audio (#342): a FLAC encoder and decoder for the app's
 * own recordings, 16-bit mono PCM (RFC 9639).
 *
 * The encoder uses only fixed predictors (orders 0–4) with partitioned
 * Rice residuals, constant and verbatim subframes; for speech that keeps
 * about half the size of the PCM. The decoder reads every subframe type
 * (LPC and wasted bits included) but only mono 16-bit streams, checks
 * each frame header's CRC-8, each frame's CRC-16, the sample count and the
 * STREAMINFO MD5, and refuses anything else, so damaged audio is an error
 * and never a different signal. Everything streams: a two-hour take never
 * sits in memory.
 *
 * Kotlin rather than the platform encoder (MediaCodec has one since API
 * 26): MediaCodec emits bare frames that MediaMuxer cannot put in a FLAC
 * file, its output depends on the device's codec, and it does not run in
 * the JVM unit tests that cover the crash windows around compression. The
 * device test decodes our files with the platform decoder as an
 * independent check.
 */
internal object Flac {
    const val BLOCK_SIZE = 4096
    private const val BITS = 16
    private const val MAX_FIXED_ORDER = 4
    private const val MAX_PARTITION_ORDER = 8
    private const val MAX_RICE_PARAMETER = 14
    private val MAGIC = "fLaC".toByteArray(Charsets.US_ASCII)
    private const val STREAMINFO_BYTES = 34

    /** What a stream's STREAMINFO says. */
    class Info(val sampleRate: Int, val totalSamples: Long, val md5: ByteArray)

    /**
     * Encodes [totalSamples] little-endian PCM16 mono samples from [input]
     * into [output], a file opened at its start: STREAMINFO is written last,
     * in place, once the frame sizes and the MD5 are known. Does not sync
     * or close [output].
     */
    fun encode(input: InputStream, totalSamples: Long, sampleRate: Int, output: FileOutputStream) {
        require(totalSamples in 0..MAX_TOTAL_SAMPLES) { "Too many samples for FLAC" }
        require(sampleRate in 1..MAX_SAMPLE_RATE) { "Unsupported sample rate" }
        val md5 = MessageDigest.getInstance("MD5")
        val buffered = BufferedOutputStream(output, 1 shl 16)
        buffered.write(MAGIC)
        buffered.write(0x80) // last metadata block, type 0 (STREAMINFO)
        buffered.write(byteArrayOf(0, 0, STREAMINFO_BYTES.toByte()))
        buffered.write(ByteArray(STREAMINFO_BYTES))

        val raw = ByteArray(BLOCK_SIZE * 2)
        val block = IntArray(BLOCK_SIZE)
        val writer = BitWriter()
        val scratch = Scratch()
        var remaining = totalSamples
        var frameNumber = 0L
        var minFrame = Int.MAX_VALUE
        var maxFrame = 0
        while (remaining > 0) {
            val count = minOf(remaining, BLOCK_SIZE.toLong()).toInt()
            readFully(input, raw, count * 2)
            md5.update(raw, 0, count * 2)
            for (i in 0 until count) {
                block[i] = ((raw[2 * i].toInt() and 0xff) or (raw[2 * i + 1].toInt() shl 8)).toShort().toInt()
            }
            writer.reset()
            encodeFrame(writer, block, count, frameNumber, sampleRate, scratch)
            buffered.write(writer.bytes, 0, writer.size)
            minFrame = minOf(minFrame, writer.size)
            maxFrame = maxOf(maxFrame, writer.size)
            remaining -= count
            frameNumber++
        }
        buffered.flush()
        val info = streamInfo(
            sampleRate,
            totalSamples,
            if (frameNumber == 0L) 0 else minFrame,
            maxFrame,
            md5.digest(),
        )
        output.channel.write(ByteBuffer.wrap(info), (MAGIC.size + 4).toLong())
    }

    /**
     * Decodes a FLAC stream into little-endian PCM16 written to [output];
     * throws when it is not a mono 16-bit stream this decoder reads, or when
     * any check (frame CRCs, sample count, MD5, trailing data) fails.
     * Returns its STREAMINFO; [onInfo] sees it before the first sample is
     * written (to put a header in front of them).
     */
    fun decode(input: InputStream, output: OutputStream, onInfo: (Info) -> Unit = {}): Info {
        val reader = BitReader(BufferedInputStream(input, 1 shl 16))
        val magic = ByteArray(4) { reader.readByte().toByte() }
        if (!magic.contentEquals(MAGIC)) throw IOException("Not a FLAC stream")
        var info: Info? = null
        var bitsPerSample = 0
        do {
            val header = reader.readByte()
            val last = header and 0x80 != 0
            val type = header and 0x7f
            val length = reader.readBits(24).toInt()
            if (info == null) {
                if (type != 0 || length != STREAMINFO_BYTES) throw IOException("FLAC stream without STREAMINFO")
                reader.readBits(16) // min block size
                reader.readBits(16) // max block size
                reader.readBits(24) // min frame size
                reader.readBits(24) // max frame size
                val sampleRate = reader.readBits(20).toInt()
                val channels = reader.readBits(3).toInt() + 1
                bitsPerSample = reader.readBits(5).toInt() + 1
                val total = reader.readBits(36)
                val md5 = ByteArray(16) { reader.readByte().toByte() }
                if (channels != 1 || bitsPerSample != BITS) throw IOException("Not a mono 16-bit FLAC stream")
                info = Info(sampleRate, total, md5)
            } else {
                repeat(length) { reader.readByte() }
            }
        } while (!last)
        val streamInfo = info ?: throw IOException("FLAC stream without STREAMINFO")
        onInfo(streamInfo)

        val md5 = MessageDigest.getInstance("MD5")
        val buffered = BufferedOutputStream(output, 1 shl 16)
        val samples = IntArray(MAX_BLOCK_SIZE)
        val pcm = ByteArray(MAX_BLOCK_SIZE * 2)
        var decoded = 0L
        var expectedFrame = 0L
        while (decoded < streamInfo.totalSamples) {
            val count = decodeFrame(reader, bitsPerSample, expectedFrame, decoded, samples)
            if (decoded + count > streamInfo.totalSamples) throw IOException("FLAC stream has more samples than it declares")
            for (i in 0 until count) {
                val sample = samples[i]
                if (sample < Short.MIN_VALUE || sample > Short.MAX_VALUE) throw IOException("FLAC sample out of range")
                pcm[2 * i] = sample.toByte()
                pcm[2 * i + 1] = (sample shr 8).toByte()
            }
            md5.update(pcm, 0, count * 2)
            buffered.write(pcm, 0, count * 2)
            decoded += count
            expectedFrame++
        }
        if (!reader.atEnd()) throw IOException("Unexpected data after the last FLAC frame")
        val digest = md5.digest()
        if (streamInfo.md5.any { it != 0.toByte() } && !digest.contentEquals(streamInfo.md5)) {
            throw IOException("FLAC audio does not match its MD5 signature")
        }
        buffered.flush()
        return streamInfo
    }

    // ---- Encoder ---------------------------------------------------------

    private class Scratch {
        val residual = IntArray(BLOCK_SIZE)
        val best = IntArray(BLOCK_SIZE)
        val sums = LongArray(1 shl MAX_PARTITION_ORDER)
        val parameters = IntArray(1 shl MAX_PARTITION_ORDER)
        val bestParameters = IntArray(1 shl MAX_PARTITION_ORDER)
    }

    private fun encodeFrame(
        writer: BitWriter,
        block: IntArray,
        count: Int,
        frameNumber: Long,
        sampleRate: Int,
        scratch: Scratch,
    ) {
        // Header: sync, fixed blocking strategy.
        writer.write(0xFFF8, 16)
        val blockSizeCode = if (count == BLOCK_SIZE) 0b1100 else 0b0111
        val sampleRateCode = if (sampleRate == 16_000) 0b0101 else 0b0000
        writer.write(blockSizeCode, 4)
        writer.write(sampleRateCode, 4)
        writer.write(0b0000, 4) // mono
        writer.write(0b100, 3) // 16 bits per sample
        writer.write(0, 1)
        writeUtf8Number(writer, frameNumber)
        if (blockSizeCode == 0b0111) writer.write(count - 1, 16)
        writer.write(crc8(writer.bytes, 0, writer.size), 8)

        encodeSubframe(writer, block, count, scratch)

        writer.alignToByte()
        writer.write(crc16(writer.bytes, 0, writer.size), 16)
    }

    private fun encodeSubframe(writer: BitWriter, block: IntArray, count: Int, scratch: Scratch) {
        if ((1 until count).all { block[it] == block[0] }) {
            writer.write(0b0_000000_0, 8)
            writer.writeSigned(block[0], BITS)
            return
        }
        var bestOrder = -1
        var bestPartitionOrder = 0
        var bestBits = 8L + BITS.toLong() * count // verbatim
        for (order in 0..minOf(MAX_FIXED_ORDER, count - 1)) {
            fixedResidual(block, count, order, scratch.residual)
            val (partitionOrder, bits) = bestPartitioning(scratch.residual, count, order, scratch)
            val total = 8L + BITS.toLong() * order + bits
            if (total < bestBits) {
                bestBits = total
                bestOrder = order
                bestPartitionOrder = partitionOrder
                System.arraycopy(scratch.residual, 0, scratch.best, 0, count - order)
                System.arraycopy(scratch.parameters, 0, scratch.bestParameters, 0, 1 shl partitionOrder)
            }
        }
        if (bestOrder < 0) {
            writer.write(0b0_000001_0, 8)
            for (i in 0 until count) writer.writeSigned(block[i], BITS)
            return
        }
        writer.write((0b001000 or bestOrder) shl 1, 8)
        for (i in 0 until bestOrder) writer.writeSigned(block[i], BITS)
        writer.write(0b00, 2) // Rice, 4-bit parameters
        writer.write(bestPartitionOrder, 4)
        val partitions = 1 shl bestPartitionOrder
        val perPartition = count shr bestPartitionOrder
        var index = 0
        for (partition in 0 until partitions) {
            val parameter = scratch.bestParameters[partition]
            writer.write(parameter, 4)
            val samples = if (partition == 0) perPartition - bestOrder else perPartition
            repeat(samples) {
                val value = scratch.best[index++]
                val folded = (value shl 1) xor (value shr 31)
                writer.writeUnary(folded ushr parameter)
                if (parameter > 0) writer.write(folded and ((1 shl parameter) - 1), parameter)
            }
        }
    }

    /** The residual of fixed predictor [order] for samples [order] until [count], from index 0. */
    private fun fixedResidual(block: IntArray, count: Int, order: Int, residual: IntArray) {
        for (i in order until count) {
            residual[i - order] = when (order) {
                0 -> block[i]
                1 -> block[i] - block[i - 1]
                2 -> block[i] - 2 * block[i - 1] + block[i - 2]
                3 -> block[i] - 3 * block[i - 1] + 3 * block[i - 2] - block[i - 3]
                else -> block[i] - 4 * block[i - 1] + 6 * block[i - 2] - 4 * block[i - 3] + block[i - 4]
            }
        }
    }

    /**
     * The partition order and the residual's estimated size in bits for
     * [order]'s residual; [Scratch.parameters] holds the Rice parameter of
     * each partition of the chosen order.
     */
    private fun bestPartitioning(residual: IntArray, count: Int, order: Int, scratch: Scratch): Pair<Int, Long> {
        var maxOrder = 0
        while (maxOrder < MAX_PARTITION_ORDER &&
            count % (1 shl (maxOrder + 1)) == 0 &&
            (count shr (maxOrder + 1)) > order
        ) {
            maxOrder++
        }
        // Sums of the folded residual per partition at the finest order;
        // coarser orders merge neighbours.
        val sums = scratch.sums
        val finest = 1 shl maxOrder
        val perFinest = count shr maxOrder
        var index = 0
        for (partition in 0 until finest) {
            val samples = if (partition == 0) perFinest - order else perFinest
            var sum = 0L
            repeat(samples) {
                val value = residual[index++]
                sum += ((value shl 1) xor (value shr 31)).toLong()
            }
            sums[partition] = sum
        }
        var bestOrder = 0
        var bestBits = Long.MAX_VALUE
        val parameters = IntArray(finest)
        var partitionOrder = maxOrder
        while (true) {
            val partitions = 1 shl partitionOrder
            val perPartition = count shr partitionOrder
            var bits = 6L
            for (partition in 0 until partitions) {
                val samples = (if (partition == 0) perPartition - order else perPartition).toLong()
                val (parameter, cost) = riceParameter(sums[partition], samples)
                parameters[partition] = parameter
                bits += 4 + cost
            }
            if (bits < bestBits) {
                bestBits = bits
                bestOrder = partitionOrder
                System.arraycopy(parameters, 0, scratch.parameters, 0, partitions)
            }
            if (partitionOrder == 0) break
            for (partition in 0 until partitions / 2) {
                sums[partition] = sums[2 * partition] + sums[2 * partition + 1]
            }
            partitionOrder--
        }
        return bestOrder to bestBits
    }

    /** The Rice parameter that codes [samples] values summing to [sum] in the fewest (estimated) bits. */
    private fun riceParameter(sum: Long, samples: Long): Pair<Int, Long> {
        var best = 0
        var bestBits = Long.MAX_VALUE
        for (parameter in 0..MAX_RICE_PARAMETER) {
            val bits = samples * (parameter + 1) + (sum shr parameter)
            if (bits < bestBits) {
                bestBits = bits
                best = parameter
            }
        }
        return best to bestBits
    }

    private fun writeUtf8Number(writer: BitWriter, value: Long) {
        when {
            value < 0x80 -> writer.write(value.toInt(), 8)
            else -> {
                var continuation = 1
                while (value >= (1L shl (5 * continuation + 6))) continuation++
                writer.write(((0xff00 shr (continuation + 1)) and 0xff) or (value ushr (6 * continuation)).toInt(), 8)
                for (shift in continuation - 1 downTo 0) {
                    writer.write(0x80 or ((value ushr (6 * shift)).toInt() and 0x3f), 8)
                }
            }
        }
    }

    private fun streamInfo(sampleRate: Int, totalSamples: Long, minFrame: Int, maxFrame: Int, md5: ByteArray): ByteArray {
        val writer = BitWriter()
        writer.write(BLOCK_SIZE, 16)
        writer.write(BLOCK_SIZE, 16)
        writer.write(minFrame, 24)
        writer.write(maxFrame, 24)
        writer.write(sampleRate, 20)
        writer.write(0, 3) // one channel
        writer.write(BITS - 1, 5)
        writer.write((totalSamples ushr 32).toInt(), 4)
        writer.write((totalSamples and 0xffffffffL).toInt(), 32)
        md5.forEach { writer.write(it.toInt() and 0xff, 8) }
        return writer.bytes.copyOf(writer.size)
    }

    private fun readFully(input: InputStream, buffer: ByteArray, count: Int) {
        var read = 0
        while (read < count) {
            val n = input.read(buffer, read, count - read)
            if (n < 0) throw EOFException("The audio ended before its declared length")
            read += n
        }
    }

    // ---- Decoder ---------------------------------------------------------

    private fun decodeFrame(
        reader: BitReader,
        streamBits: Int,
        expectedFrame: Long,
        decodedSamples: Long,
        samples: IntArray,
    ): Int {
        reader.startFrame()
        if (reader.readBits(15).toInt() != 0x7FFC) throw IOException("Lost FLAC frame sync")
        val variable = reader.readBits(1) == 1L
        val blockSizeCode = reader.readBits(4).toInt()
        val sampleRateCode = reader.readBits(4).toInt()
        val channelAssignment = reader.readBits(4).toInt()
        val sampleSizeCode = reader.readBits(3).toInt()
        if (reader.readBits(1) != 0L) throw IOException("Reserved FLAC header bit set")
        val number = readUtf8Number(reader)
        if (number != if (variable) decodedSamples else expectedFrame) throw IOException("FLAC frames out of order")
        val count = when (blockSizeCode) {
            0b0001 -> 192
            in 0b0010..0b0101 -> 576 shl (blockSizeCode - 2)
            0b0110 -> reader.readBits(8).toInt() + 1
            0b0111 -> reader.readBits(16).toInt() + 1
            in 0b1000..0b1111 -> 256 shl (blockSizeCode - 8)
            else -> throw IOException("Reserved FLAC block size")
        }
        when (sampleRateCode) {
            0b1100 -> reader.readBits(8)
            0b1101, 0b1110 -> reader.readBits(16)
            0b1111 -> throw IOException("Invalid FLAC sample rate")
        }
        if (channelAssignment != 0) throw IOException("Not a mono FLAC frame")
        val bits = when (sampleSizeCode) {
            0b000 -> streamBits
            0b100 -> 16
            else -> throw IOException("Not a 16-bit FLAC frame")
        }
        if (bits != streamBits) throw IOException("FLAC frame disagrees with STREAMINFO")
        val headerCrc = reader.crc8
        if (reader.readBits(8).toInt() != headerCrc) throw IOException("FLAC frame header is damaged")
        if (count > MAX_BLOCK_SIZE) throw IOException("FLAC block too large")

        decodeSubframe(reader, count, bits, samples)

        reader.alignToByte()
        val frameCrc = reader.crc16
        if (reader.readBits(16).toInt() != frameCrc) throw IOException("FLAC frame is damaged")
        return count
    }

    private fun decodeSubframe(reader: BitReader, count: Int, streamBits: Int, samples: IntArray) {
        if (reader.readBits(1) != 0L) throw IOException("FLAC subframe padding bit set")
        val type = reader.readBits(6).toInt()
        var wasted = 0
        if (reader.readBits(1) == 1L) wasted = reader.readUnary() + 1
        if (wasted >= streamBits) throw IOException("Invalid FLAC wasted bits")
        val bits = streamBits - wasted
        when {
            type == 0b000000 -> {
                val value = reader.readSigned(bits)
                samples.fill(value, 0, count)
            }
            type == 0b000001 -> for (i in 0 until count) samples[i] = reader.readSigned(bits)
            type in 0b001000..0b001100 -> {
                val order = type - 0b001000
                if (order > count) throw IOException("FLAC predictor order exceeds the block")
                for (i in 0 until order) samples[i] = reader.readSigned(bits)
                decodeResidual(reader, count, order, samples)
                for (i in order until count) {
                    samples[i] += when (order) {
                        0 -> 0
                        1 -> samples[i - 1]
                        2 -> 2 * samples[i - 1] - samples[i - 2]
                        3 -> 3 * samples[i - 1] - 3 * samples[i - 2] + samples[i - 3]
                        else -> 4 * samples[i - 1] - 6 * samples[i - 2] + 4 * samples[i - 3] - samples[i - 4]
                    }
                }
            }
            type >= 0b100000 -> {
                val order = type - 0b011111
                if (order > count) throw IOException("FLAC predictor order exceeds the block")
                for (i in 0 until order) samples[i] = reader.readSigned(bits)
                val precision = reader.readBits(4).toInt() + 1
                if (precision == 16) throw IOException("Invalid FLAC coefficient precision")
                val shift = reader.readSigned(5)
                if (shift < 0) throw IOException("Negative FLAC predictor shift")
                val coefficients = IntArray(order) { reader.readSigned(precision) }
                decodeResidual(reader, count, order, samples)
                for (i in order until count) {
                    var prediction = 0L
                    for (j in 0 until order) prediction += coefficients[j].toLong() * samples[i - 1 - j]
                    samples[i] += (prediction shr shift).toInt()
                }
            }
            else -> throw IOException("Reserved FLAC subframe type")
        }
        if (wasted > 0) for (i in 0 until count) samples[i] = samples[i] shl wasted
    }

    /** Reads the residual into samples [order] until [count] (the predictor adds to it). */
    private fun decodeResidual(reader: BitReader, count: Int, order: Int, samples: IntArray) {
        val method = reader.readBits(2).toInt()
        if (method > 1) throw IOException("Reserved FLAC residual coding")
        val parameterBits = if (method == 0) 4 else 5
        val escape = (1 shl parameterBits) - 1
        val partitionOrder = reader.readBits(4).toInt()
        val partitions = 1 shl partitionOrder
        if (count % partitions != 0 || (count shr partitionOrder) < order) {
            throw IOException("Invalid FLAC partition order")
        }
        var index = order
        for (partition in 0 until partitions) {
            val samplesInPartition = (count shr partitionOrder) - if (partition == 0) order else 0
            val parameter = reader.readBits(parameterBits).toInt()
            if (parameter == escape) {
                val raw = reader.readBits(5).toInt()
                repeat(samplesInPartition) { samples[index++] = if (raw == 0) 0 else reader.readSigned(raw) }
            } else {
                repeat(samplesInPartition) {
                    val quotient = reader.readUnary()
                    if (quotient > MAX_QUOTIENT) throw IOException("FLAC residual out of range")
                    val folded = (quotient.toLong() shl parameter) or reader.readBits(parameter)
                    samples[index++] = ((folded ushr 1) xor -(folded and 1)).toInt()
                }
            }
        }
    }

    private fun readUtf8Number(reader: BitReader): Long {
        val first = reader.readBits(8).toInt()
        if (first and 0x80 == 0) return first.toLong()
        var continuation = 0
        var mask = 0x40
        while (first and mask != 0) {
            continuation++
            mask = mask shr 1
        }
        if (continuation == 0 || continuation > 6) throw IOException("Invalid FLAC frame number")
        var value = (first and (mask - 1)).toLong()
        repeat(continuation) {
            val next = reader.readBits(8).toInt()
            if (next and 0xc0 != 0x80) throw IOException("Invalid FLAC frame number")
            value = (value shl 6) or (next and 0x3f).toLong()
        }
        return value
    }

    // ---- Bits and checksums ------------------------------------------------

    private class BitWriter {
        var bytes = ByteArray(BLOCK_SIZE * 2 + 64)
            private set
        var size = 0
            private set
        private var accumulator = 0L
        private var pending = 0

        fun reset() {
            size = 0
            accumulator = 0
            pending = 0
        }

        /** Writes the low [count] bits of [value] (count ≤ 32), most significant first. */
        fun write(value: Int, count: Int) {
            if (count == 0) return
            accumulator = (accumulator shl count) or (value.toLong() and ((1L shl count) - 1))
            pending += count
            while (pending >= 8) {
                pending -= 8
                put((accumulator ushr pending).toInt())
            }
            accumulator = accumulator and ((1L shl pending) - 1)
        }

        fun writeSigned(value: Int, count: Int) = write(value, count)

        fun writeUnary(zeros: Int) {
            var left = zeros
            while (left >= 31) {
                write(0, 31)
                left -= 31
            }
            write(1, left + 1)
        }

        fun alignToByte() {
            if (pending > 0) write(0, 8 - pending)
        }

        private fun put(byte: Int) {
            if (size == bytes.size) bytes = bytes.copyOf(bytes.size * 2)
            bytes[size++] = byte.toByte()
        }
    }

    private class BitReader(private val input: InputStream) {
        private var current = 0
        private var left = 0
        var crc8 = 0
            private set
        var crc16 = 0
            private set

        fun startFrame() {
            check(left == 0)
            crc8 = 0
            crc16 = 0
        }

        fun readByte(): Int {
            check(left == 0)
            return next()
        }

        private fun next(): Int {
            val byte = input.read()
            if (byte < 0) throw EOFException("The FLAC stream ended early")
            crc8 = CRC8[crc8 xor byte]
            crc16 = ((crc16 shl 8) xor CRC16[(crc16 ushr 8) xor byte]) and 0xffff
            return byte
        }

        /** [count] ≤ 36 bits, unsigned. */
        fun readBits(count: Int): Long {
            var value = 0L
            var needed = count
            while (needed > 0) {
                if (left == 0) {
                    current = next()
                    left = 8
                }
                val take = minOf(left, needed)
                val bits = (current ushr (left - take)) and ((1 shl take) - 1)
                value = (value shl take) or bits.toLong()
                left -= take
                needed -= take
            }
            return value
        }

        fun readSigned(count: Int): Int {
            val value = readBits(count)
            return ((value shl (64 - count)) shr (64 - count)).toInt()
        }

        fun readUnary(): Int {
            var zeros = 0
            while (true) {
                if (left == 0) {
                    current = next()
                    left = 8
                }
                val masked = current and ((1 shl left) - 1)
                if (masked == 0) {
                    zeros += left
                    left = 0
                    if (zeros > MAX_QUOTIENT) throw IOException("FLAC residual out of range")
                } else {
                    val leading = Integer.numberOfLeadingZeros(masked) - (32 - left)
                    zeros += leading
                    left -= leading + 1
                    return zeros
                }
            }
        }

        fun alignToByte() {
            left = 0
        }

        fun atEnd(): Boolean = left == 0 && input.read() < 0
    }

    private val CRC8 = IntArray(256) { index ->
        var crc = index
        repeat(8) { crc = if (crc and 0x80 != 0) ((crc shl 1) xor 0x07) and 0xff else (crc shl 1) and 0xff }
        crc
    }

    private val CRC16 = IntArray(256) { index ->
        var crc = index shl 8
        repeat(8) { crc = if (crc and 0x8000 != 0) ((crc shl 1) xor 0x8005) and 0xffff else (crc shl 1) and 0xffff }
        crc
    }

    private fun crc8(bytes: ByteArray, offset: Int, count: Int): Int {
        var crc = 0
        for (i in offset until offset + count) crc = CRC8[crc xor (bytes[i].toInt() and 0xff)]
        return crc
    }

    private fun crc16(bytes: ByteArray, offset: Int, count: Int): Int {
        var crc = 0
        for (i in offset until offset + count) {
            crc = ((crc shl 8) xor CRC16[(crc ushr 8) xor (bytes[i].toInt() and 0xff)]) and 0xffff
        }
        return crc
    }

    private const val MAX_TOTAL_SAMPLES = (1L shl 36) - 1
    private const val MAX_SAMPLE_RATE = (1 shl 20) - 1
    private const val MAX_BLOCK_SIZE = 65_536

    /** Bounds the unary run a damaged stream can make the decoder count. */
    private const val MAX_QUOTIENT = 1 shl 20
}
