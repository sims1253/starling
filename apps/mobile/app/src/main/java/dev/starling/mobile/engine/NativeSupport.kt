package dev.starling.mobile.engine

import android.system.Os
import dev.starling.mobile.BuildConfig
import java.io.File

/**
 * Device checks for the native engine, kept out of the JNI layer so they run
 * before libstarling_jni is ever loaded.
 *
 * The arm64 build targets armv8.2-a with the dot-product and FP16 vector
 * extensions by default (apps/mobile/app/src/main/cpp/CMakeLists.txt): the
 * quantized matmuls are several times faster with them, and every mainstream
 * arm64 phone core since 2018 (Cortex-A55/A75 and newer) has both. The i8mm
 * APK variant additionally requires the int8 matrix-multiply extension.
 * A core without a required feature would crash with SIGILL inside the
 * engine, so on those devices the on-device option reports why it cannot
 * run instead; the server path is unaffected.
 */
object NativeSupport {
    /** CPU features (as named in /proc/cpuinfo) this build's arm64 library requires. */
    private val REQUIRED_ARM64_FEATURES = BuildConfig.ARM64_REQUIRED_CPU_FEATURES.toList()

    private const val THREADS_ENV = "STARLING_GGML_THREADS"

    @Volatile
    private var threadsApplied = false

    /** Null when this device can run the native engine, else the reason it cannot. */
    fun unsupportedReason(): String? = unsupportedReason(
        System.getProperty("os.arch"),
        runCatching { File("/proc/cpuinfo").readText() }.getOrNull(),
    )

    internal fun unsupportedReason(
        osArch: String?,
        cpuinfo: String?,
        required: List<String> = REQUIRED_ARM64_FEATURES,
    ): String? {
        when (osArch) {
            "aarch64" -> Unit
            // The x86_64 library (emulators, Chromebooks) is built without
            // extra ISA requirements.
            "x86_64", "amd64" -> return null
            // The APK ships arm64-v8a and x86_64 only; anything else cannot
            // load the engine at all.
            else -> return "the on-device engine is built for 64-bit ARM and x86_64 only " +
                "(this device reports $osArch). Use a Starling server instead."
        }
        // Unreadable cpuinfo proves nothing either way; do not block on it.
        val text = cpuinfo ?: return null
        val featureLines = text.lineSequence()
            .filter { it.trimStart().startsWith("Features", ignoreCase = true) }
            .map { line -> line.substringAfter(':').trim().split(Regex("\\s+")).toSet() }
            .toList()
        if (featureLines.isEmpty()) return null
        // Big.LITTLE parts list every core; the engine's threads can land on
        // any of them, so every core must have the features.
        val missing = required.filter { feature -> featureLines.any { feature !in it } }
        if (missing.isEmpty()) return null
        val remedy = if ("i8mm" in missing && missing.size == 1) {
            "Install the standard Starling Mobile APK instead of the i8mm build, or use a Starling server."
        } else {
            "Use a Starling server instead."
        }
        return "this phone's CPU lacks instructions this engine build requires " +
            "(missing: ${missing.joinToString()}). $remedy"
    }

    /**
     * Sets the engine's CPU thread count to the number of performance cores
     * unless [THREADS_ENV] is already set. The engine's own default counts
     * every core, and on big.LITTLE phones the efficiency cores then gate
     * each parallel step. Must run before the first model load.
     */
    fun applyThreadDefault() {
        if (threadsApplied) return
        threadsApplied = true
        if (!System.getenv(THREADS_ENV).isNullOrEmpty()) return
        val maxFrequencies = (0 until Runtime.getRuntime().availableProcessors()).mapNotNull { cpu ->
            runCatching {
                File("/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq").readText().trim().toLong()
            }.getOrNull()
        }
        val threads = performanceCores(maxFrequencies, Runtime.getRuntime().availableProcessors())
        runCatching { Os.setenv(THREADS_ENV, threads.toString(), false) }
    }

    /**
     * Every core except the slowest cluster: the prime and performance
     * clusters, not the efficiency cores that would gate each parallel step
     * (Tensor G5: 1 X4 + 5 A725 -> 6 of 8). A single-cluster CPU uses all of
     * its cores. Falls back to half the cores when frequencies are unreadable.
     */
    internal fun performanceCores(maxFrequenciesKhz: List<Long>, availableProcessors: Int): Int {
        val available = maxOf(1, availableProcessors)
        val slowest = maxFrequenciesKhz.minOrNull()
        if (slowest == null || slowest <= 0L || maxFrequenciesKhz.size < available) return maxOf(1, available / 2)
        val faster = maxFrequenciesKhz.count { it > slowest }
        return (if (faster == 0) maxFrequenciesKhz.size else faster).coerceIn(1, available)
    }
}
