package dev.starling.mobile.engine

import java.io.ByteArrayOutputStream
import org.junit.Assert.assertEquals
import org.junit.Test

/** Model-kind detection from GGUF metadata, and the per-kind memory allowance. */
class ModelKindTest {
    private fun parsed(
        architecture: String? = null,
        keys: List<String> = emptyList(),
        uints: Map<String, Long> = emptyMap(),
    ) = GgufMetadata.Parsed(
        keys = listOfNotNull(architecture?.let { "general.architecture" }) + keys + uints.keys,
        strings = architecture?.let { mapOf("general.architecture" to it) }.orEmpty(),
        uints = uints,
    )

    private fun supported(metadata: GgufMetadata.Parsed): ModelKind.Detection.Supported {
        val detection = ModelKind.detect(metadata)
        return detection as? ModelKind.Detection.Supported ?: throw AssertionError("not supported: $detection")
    }

    private fun reason(metadata: GgufMetadata.Parsed): String {
        val detection = ModelKind.detect(metadata)
        return (detection as? ModelKind.Detection.Unsupported)?.reason ?: throw AssertionError("supported: $detection")
    }

    @Test
    fun eachArchitectureMapsToItsEngineKind() {
        val expected = mapOf(
            "moss_transcribe" to ModelKind.MOSS,
            "ark" to ModelKind.ARK,
            "granite" to ModelKind.GRANITE,
            "qwen3" to ModelKind.QWEN3,
            "voxtral" to ModelKind.VOXTRAL,
        )
        for ((architecture, kind) in expected) {
            assertEquals(architecture, kind, supported(parsed(architecture)).kind)
        }
    }

    @Test
    fun nativeKindsAreTheCApiEnumValues() {
        // cpp/include/starling_ggml.h enum starling_ggml_model.
        assertEquals(
            mapOf(
                ModelKind.PARAKEET to 1,
                ModelKind.MOSS to 2,
                ModelKind.ARK to 3,
                ModelKind.GRANITE to 6,
                ModelKind.QWEN3 to 7,
                ModelKind.VOXTRAL to 11,
            ),
            ModelKind.entries.associateWith { it.nativeKind },
        )
    }

    @Test
    fun parakeetIsDetectedByItsKeyFamiliesWhateverTheArchitecture() {
        for (metadata in listOf(
            parsed("parakeet", keys = listOf("parakeet.preprocessor.normalize")),
            parsed("parakeet_tdt", keys = listOf("parakeet.vocab_size")),
            parsed(keys = listOf("stt.parakeet.encoder.n_layers")),
            // A community file whose architecture tag says something else.
            parsed("llama", keys = listOf("parakeet.encoder.d_model")),
        )) {
            assertEquals(
                ModelKind.Detection.Supported(ModelKind.PARAKEET, "Parakeet"),
                ModelKind.detect(metadata),
            )
        }
    }

    @Test
    fun aParakeetArchitectureTagWithoutItsKeysIsNotEnough() {
        assertEquals("This GGUF (parakeet) is not a supported speech model.", reason(parsed("parakeet")))
    }

    @Test
    fun s1IsRejectedAsATextModel() {
        assertEquals("S1-mini is a text model, not a speech model.", reason(parsed("s1")))
    }

    @Test
    fun unknownArchitecturesAreRejectedByName() {
        assertEquals("This GGUF (llama) is not a supported speech model.", reason(parsed("llama")))
        assertEquals("This GGUF (whisper) is not a supported speech model.", reason(parsed("whisper")))
        assertEquals(
            "This GGUF (no architecture tag) is not a supported speech model.",
            reason(parsed(keys = listOf("general.name"))),
        )
        // Architectures are compared exactly, as the native loaders do.
        assertEquals("This GGUF (Qwen3) is not a supported speech model.", reason(parsed("Qwen3")))
    }

    @Test
    fun capsTheArchitectureEchoedIntoTheRejection() {
        val echoed = reason(parsed("x".repeat(200)))
            .removePrefix("This GGUF (")
            .removeSuffix(") is not a supported speech model.")
        assertEquals("x".repeat(40), echoed)
    }

    @Test
    fun displayNamesTellTheSizesOfSharedEnginesApart() {
        fun name(architecture: String, uints: Map<String, Long> = emptyMap()) =
            supported(parsed(architecture, uints = uints)).displayName

        assertEquals("ARK-ASR 3B", name("ark", mapOf(ModelKind.ARK_HIDDEN_SIZE_KEY to 2048L)))
        assertEquals("ARK-ASR 0.6B", name("ark", mapOf(ModelKind.ARK_HIDDEN_SIZE_KEY to 896L)))
        assertEquals("ARK-ASR", name("ark", mapOf(ModelKind.ARK_HIDDEN_SIZE_KEY to 1234L)))
        assertEquals("ARK-ASR", name("ark"))
        assertEquals("Qwen3-ASR 1.7B", name("qwen3", mapOf(ModelKind.QWEN3_HIDDEN_SIZE_KEY to 2048L)))
        assertEquals("Qwen3-ASR 0.6B", name("qwen3", mapOf(ModelKind.QWEN3_HIDDEN_SIZE_KEY to 1024L)))
        assertEquals("Qwen3-ASR", name("qwen3", mapOf(ModelKind.QWEN3_HIDDEN_SIZE_KEY to 1234L)))
        assertEquals("Qwen3-ASR", name("qwen3"))
        // The kind, never the size, picks the engine.
        assertEquals(ModelKind.QWEN3, supported(parsed("qwen3", uints = mapOf(ModelKind.QWEN3_HIDDEN_SIZE_KEY to 1024L))).kind)
        assertEquals("Granite Speech", name("granite"))
        assertEquals("MOSS-Transcribe", name("moss_transcribe"))
        assertEquals("Voxtral Realtime", name("voxtral"))
    }

    @Test
    fun theMemoryAllowanceDependsOnTheKind() {
        val mib = 1024L * 1024
        val weights = 600L * mib
        assertEquals(weights + 128 * mib, ModelKind.PARAKEET.requiredMemoryBytes(weights))
        for (kind in ModelKind.entries - ModelKind.PARAKEET) {
            assertEquals(kind.name, weights + 768 * mib, kind.requiredMemoryBytes(weights))
        }
    }

    @Test
    fun llmDecoderWindowsStayWithinTheirThirtySecondEncoders() {
        assertEquals(32.0, ModelKind.PARAKEET.maxWindowSeconds, 0.0)
        for (kind in ModelKind.entries - ModelKind.PARAKEET) {
            assertEquals(kind.name, 30.0, kind.maxWindowSeconds, 0.0)
        }
    }

    companion object {
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

        /** A tensorless GGUFv3 header with string [kvs], for fixtures that pad it to a model's size. */
        fun gguf(vararg kvs: Pair<String, String>): ByteArray {
            val out = ByteArrayOutputStream()
            out.write("GGUF".toByteArray(Charsets.US_ASCII))
            out.write(u32(3))
            out.write(u64(0))
            out.write(u64(kvs.size.toLong()))
            for ((key, value) in kvs) {
                out.write(u64(key.length.toLong()))
                out.write(key.toByteArray(Charsets.UTF_8))
                out.write(u32(8))
                out.write(u64(value.length.toLong()))
                out.write(value.toByteArray(Charsets.UTF_8))
            }
            return out.toByteArray()
        }
    }
}
