package dev.starling.mobile.storage

import dev.starling.mobile.engine.WavPcm
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Random
import kotlin.math.PI
import kotlin.math.roundToInt
import kotlin.math.sin

/** #342: the at-rest codec is sample-exact and refuses damaged audio. */
class FlacTest {
    @get:Rule
    val folder = TemporaryFolder()

    private fun repoFile(path: String): File =
        generateSequence(File("").absoluteFile) { it.parentFile }
            .map { File(it, path) }
            .firstOrNull(File::isFile)
            ?: error("$path not found above ${File("").absolutePath}")

    private fun pcmBytes(samples: ShortArray): ByteArray = ByteArray(samples.size * 2).also { bytes ->
        samples.forEachIndexed { i, sample ->
            bytes[2 * i] = sample.toByte()
            bytes[2 * i + 1] = (sample.toInt() shr 8).toByte()
        }
    }

    private fun encode(pcm: ByteArray, sampleRate: Int = 16_000): File {
        val file = folder.newFile()
        FileOutputStream(file).use { output ->
            Flac.encode(ByteArrayInputStream(pcm), pcm.size / 2L, sampleRate, output)
        }
        return file
    }

    private fun decode(bytes: ByteArray): ByteArray =
        ByteArrayOutputStream().also { Flac.decode(ByteArrayInputStream(bytes), it) }.toByteArray()

    private fun assertRoundTrip(pcm: ByteArray, name: String, sampleRate: Int = 16_000): File {
        val file = encode(pcm, sampleRate)
        val decoded = ByteArrayOutputStream()
        val info = Flac.decode(file.inputStream(), decoded)
        assertArrayEquals(name, pcm, decoded.toByteArray())
        assertEquals(name, pcm.size / 2L, info.totalSamples)
        assertEquals(name, sampleRate, info.sampleRate)
        return file
    }

    /** Port of `tests/fidelity_audio.py::synthesize_samples`, as the desktop test uses it. */
    private fun synthesize(segments: org.json.JSONArray, rate: Int): ShortArray {
        val samples = ArrayList<Short>()
        for (index in 0 until segments.length()) {
            val segment = segments.getJSONObject(index)
            val count = (segment.getDouble("duration_seconds") * rate).roundToInt()
            if (segment.getString("kind") == "silence") {
                repeat(count) { samples.add(0) }
                continue
            }
            for (i in 0 until count) {
                val t = i.toDouble() / rate
                val envelope = 0.6 + 0.4 * sin(2 * PI * 10.0 * t)
                val value = 12_000.0 * envelope * sin(2 * PI * 220.0 * t)
                samples.add(value.coerceIn(-32_768.0, 32_767.0).toInt().toShort())
            }
        }
        return samples.toShortArray()
    }

    @Test
    fun theFidelityCorpusRoundTripsSampleExact() {
        val corpus = JSONObject(repoFile("packages/contracts/fidelity-corpus/corpus.json").readText())
        val fixtures = corpus.getJSONArray("fixtures")
        assertTrue("the whole corpus, long passage included", fixtures.length() >= 8)
        for (index in 0 until fixtures.length()) {
            val fixture = fixtures.getJSONObject(index)
            val synthesis = fixture.getJSONObject("synthesis")
            val pcm = pcmBytes(synthesize(synthesis.getJSONArray("segments"), synthesis.getInt("sample_rate")))
            assertRoundTrip(pcm, fixture.getString("id"), synthesis.getInt("sample_rate"))
        }
    }

    @Test
    fun realSpeechRoundTripsSampleExactAtAboutHalfTheSize() {
        val pcm = WavPcm.decodePcm16(repoFile("tests/fixtures/2086-149220-0033.wav"))!!.pcm
        val file = assertRoundTrip(pcm, "librispeech")
        assertTrue("${file.length()} of ${pcm.size}", file.length() < pcm.size * 0.65)
    }

    /**
     * Forms our encoder never writes but a file from elsewhere may hold,
     * decoded against the PCM they were made from. The fixtures are samples
     * 16000 until 26000 of the LibriSpeech fixture, written by reference
     * libFLAC 1.5.0 (`flac -8 --no-padding --no-seektable`, raw 16 kHz mono
     * s16le input): `lpc-order12.flac` as they are, LPC subframes of orders
     * 11 and 12; `wasted-bits3.flac` with the low 3 bits of every sample
     * cleared, the same LPC subframes with 3 wasted bits.
     */
    @Test
    fun referenceEncodedLpcAndWastedBitsDecodeSampleExact() {
        val speech = WavPcm.decodePcm16(repoFile("tests/fixtures/2086-149220-0033.wav"))!!.pcm
        val slice = speech.copyOfRange(16_000 * 2, 26_000 * 2)
        val cleared = slice.copyOf().also { bytes -> for (i in bytes.indices step 2) bytes[i] = (bytes[i].toInt() and 0xF8).toByte() }
        for ((name, expected) in listOf("lpc-order12.flac" to slice, "wasted-bits3.flac" to cleared)) {
            val flac = requireNotNull(javaClass.getResourceAsStream("/flac/$name")) { "$name is missing" }.use { it.readBytes() }
            val decoded = ByteArrayOutputStream()
            val info = Flac.decode(ByteArrayInputStream(flac), decoded)
            assertArrayEquals(name, expected, decoded.toByteArray())
            assertEquals(name, 10_000L, info.totalSamples)
            assertEquals(name, 16_000, info.sampleRate)
        }
    }

    @Test
    fun edgeCasesRoundTrip() {
        val random = Random(342)
        val cases = mapOf(
            "empty" to ShortArray(0),
            "one sample" to shortArrayOf(-7),
            "full scale square" to ShortArray(9_000) { if ((it / 37) % 2 == 0) Short.MAX_VALUE else Short.MIN_VALUE },
            "white noise" to ShortArray(20_000) { random.nextInt(65_536).minus(32_768).toShort() },
            "digital silence" to ShortArray(4_096 * 3),
            "a block and one" to ShortArray(4_097) { (it * 13 % 2_000).toShort() },
            "odd last block" to ShortArray(4_096 + 4_095) { (sin(it / 9.0) * 30_000).toInt().toShort() },
            "pathological residual" to ShortArray(8_192) { if (it % 2 == 0) Short.MAX_VALUE else Short.MIN_VALUE },
        )
        cases.forEach { (name, samples) -> assertRoundTrip(pcmBytes(samples), name) }
        // Other sample rates go through STREAMINFO instead of the 16 kHz code.
        assertRoundTrip(pcmBytes(ShortArray(5_000) { (it % 300).toShort() }), "44.1 kHz", 44_100)
    }

    @Test
    fun anyDamagedByteIsAnErrorNotADifferentSignal() {
        val pcm = WavPcm.decodePcm16(repoFile("tests/fixtures/2086-149220-0033.wav"))!!.pcm.copyOf(16_000 * 2 * 3)
        val good = encode(pcm).readBytes()
        val random = Random(7)
        repeat(200) {
            val damaged = good.copyOf()
            val at = 42 + random.nextInt(damaged.size - 42) // past STREAMINFO's sizes
            damaged[at] = (damaged[at].toInt() xor (1 shl random.nextInt(8))).toByte()
            try {
                val decoded = decode(damaged)
                fail("byte $at flipped decoded to ${if (decoded.contentEquals(pcm)) "the same" else "other"} audio")
            } catch (_: IOException) {
                // Refused, as it must be.
            }
        }
        val truncated = good.copyOf(good.size - 1)
        try {
            decode(truncated)
            fail("a truncated stream decoded")
        } catch (_: IOException) {
        }
        try {
            decode(good + byteArrayOf(0))
            fail("trailing data was accepted")
        } catch (_: IOException) {
        }
    }

    /** An independent decoder agrees, when the reference `flac` tool is installed. */
    @Test
    fun theReferenceDecoderAgreesWhenAvailable() {
        val tool = System.getenv("PATH").orEmpty().split(File.pathSeparator)
            .map { File(it, "flac") }.firstOrNull(File::canExecute)
        assumeTrue("flac is not installed", tool != null)
        val pcm = WavPcm.decodePcm16(repoFile("tests/fixtures/2086-149220-0033.wav"))!!.pcm
        val encoded = encode(pcm)
        val raw = File(folder.root, "decoded.raw")
        val process = ProcessBuilder(
            tool!!.path, "--silent", "--decode", "--force", "--force-raw-format",
            "--endian=little", "--sign=signed", "-o", raw.path, encoded.path,
        ).redirectErrorStream(true).start()
        val output = process.inputStream.readBytes().toString(Charsets.UTF_8)
        assertEquals(output, 0, process.waitFor())
        assertArrayEquals(pcm, raw.readBytes())
    }
}
