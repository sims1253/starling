package dev.starling.mobile.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeSupportTest {
    private val modern = """
        processor	: 0
        Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
        processor	: 1
        Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
    """.trimIndent()

    private val required = listOf("asimddp", "asimdhp")

    private val cortexA53 = """
        processor	: 0
        Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid
    """.trimIndent()

    @Test
    fun aModernArm64CoreIsSupported() {
        assertNull(NativeSupport.unsupportedReason("aarch64", modern, required))
    }

    @Test
    fun anArmv80CoreIsRefusedWithTheMissingFeatures() {
        val reason = NativeSupport.unsupportedReason("aarch64", cortexA53, required)

        assertNotNull(reason)
        assertTrue(reason!!.contains("asimddp"))
        assertTrue(reason.contains("asimdhp"))
    }

    @Test
    fun oneCoreWithoutTheFeaturesIsEnoughToRefuse() {
        val mixed = "$modern\nprocessor\t: 2\nFeatures\t: fp asimd cpuid"

        assertNotNull(NativeSupport.unsupportedReason("aarch64", mixed, required))
    }

    @Test
    fun otherArchitecturesAndUnreadableCpuinfoAreNotBlocked() {
        assertNull(NativeSupport.unsupportedReason("x86_64", cortexA53, required))
        assertNull(NativeSupport.unsupportedReason("aarch64", null, required))
        assertNull(NativeSupport.unsupportedReason("aarch64", "processor\t: 0\n", required))
    }

    @Test
    fun theI8mmBuildIsRefusedOnADotprodOnlyCoreAndPointsAtTheStandardApk() {
        val reason = NativeSupport.unsupportedReason("aarch64", modern, required + "i8mm")

        assertNotNull(reason)
        assertTrue(reason!!.contains("i8mm"))
        assertTrue(reason.contains("standard Starling Mobile APK"))
        assertNull(NativeSupport.unsupportedReason("aarch64", modern.replace("asimddp", "asimddp i8mm"), required + "i8mm"))
    }

    @Test
    fun tensorG5UsesThePrimeAndPerformanceCores() {
        // 1 Cortex-X4 + 5 Cortex-A725 + 2 Cortex-A520.
        val frequencies = listOf(2_250_000L, 2_250_000L) + List(5) { 3_050_000L } + 3_780_000L

        assertEquals(6, NativeSupport.performanceCores(frequencies, 8))
    }

    @Test
    fun aSingleClusterUsesEveryCore() {
        assertEquals(8, NativeSupport.performanceCores(List(8) { 2_400_000L }, 8))
    }

    @Test
    fun performanceCoresExcludeTheEfficiencyCluster() {
        // 1 prime + 3 performance + 4 efficiency (a typical 1+3+4 phone).
        val frequencies = listOf(2_000_000L, 2_000_000L, 2_000_000L, 2_000_000L, 2_850_000L, 2_850_000L, 2_850_000L, 3_200_000L)

        assertEquals(4, NativeSupport.performanceCores(frequencies, 8))
    }

    @Test
    fun unreadableFrequenciesFallBackToHalfTheCores() {
        assertEquals(4, NativeSupport.performanceCores(emptyList(), 8))
        assertEquals(1, NativeSupport.performanceCores(emptyList(), 1))
        assertEquals(4, NativeSupport.performanceCores(listOf(3_000_000L), 8))
    }
}
