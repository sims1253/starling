package dev.starling.mobile.engine

/**
 * Recognizes the native engine's GPU-driver failures (#325): a fence that
 * timed out, a lost device, a driver OOM, or a wedge recorded earlier (by
 * this process or, via the marker file, a previous one). After one of these
 * the Vulkan engine refuses further work, so the caller frees the model and
 * reloads it: the reload falls back to the CPU engine instead of feeding a
 * failing GPU driver. The phrases come from `cpp/fast/vk_runtime.cpp`.
 */
internal object GpuFailure {
    private val SIGNATURES = listOf(
        "the driver may be wedged",   // fence timeout / degradation watchdog
        "GPU driver",                 // device lost, driver OOM, recorded wedge
    )

    fun matches(error: String?): Boolean =
        error != null && SIGNATURES.any { error.contains(it, ignoreCase = true) }
}
