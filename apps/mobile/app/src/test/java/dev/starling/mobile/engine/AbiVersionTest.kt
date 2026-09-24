package dev.starling.mobile.engine

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The on-device library refuses to load when its ABI differs from
 * [StarlingNative.EXPECTED_ABI_VERSION], so a header bump without a matching
 * app bump breaks every on-device recording. Catch that drift on the JVM.
 */
class AbiVersionTest {
    @Test
    fun expectedAbiMatchesNativeHeader() {
        val header = generateSequence(File("").absoluteFile) { it.parentFile }
            .map { File(it, "cpp/include/starling_ggml.h") }
            .firstOrNull { it.isFile }
            ?: error("cpp/include/starling_ggml.h not found above ${File("").absolutePath}")
        val abi = Regex("""^#define STARLING_GGML_ABI_VERSION (\d+)""", RegexOption.MULTILINE)
            .find(header.readText())?.groupValues?.get(1)?.toInt()
            ?: error("STARLING_GGML_ABI_VERSION missing from $header")
        assertEquals(abi, StarlingNative.EXPECTED_ABI_VERSION)
    }
}
