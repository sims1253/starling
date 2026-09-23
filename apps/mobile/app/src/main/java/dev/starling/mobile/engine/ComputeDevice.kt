package dev.starling.mobile.engine

import java.io.File

/** Where the on-device engine computes. [ggmlName] is the STARLING_GGML_DEVICE value. */
enum class ComputeDevice(val ggmlName: String) {
    CPU("cpu"),
    GPU("Vulkan0"),
}

/** Persisted device preference (SharedPreferences in the app, a field in tests). */
interface DevicePreferenceStore {
    fun get(): ComputeDevice
    fun set(device: ComputeDevice)
}

/** Hooks [OnDeviceEngine] runs around its native calls. */
interface NativeCallPolicy {
    /** Once, before the first model load of the process. */
    fun beforeFirstLoad() = Unit

    /** Around every native load/transcribe call. */
    fun <T> guard(block: () -> T): T = block()

    object None : NativeCallPolicy
}

/**
 * Chooses the on-device engine's compute device for this process and guards
 * GPU calls against crash loops.
 *
 * The native backend is process-global and picks its device once, at the
 * first model load, from STARLING_GGML_DEVICE (unset, it would auto-pick any
 * GPU). So the device is always set explicitly, CPU unless the user opted in
 * and this build has the Vulkan backend, and a later change of preference
 * only takes effect after the process restarts ([restartNeeded]). On the CPU,
 * GGML_DISABLE_VULKAN also keeps ggml from registering its Vulkan backend,
 * which would otherwise create a Vulkan instance and enumerate the GPU even
 * when it is not used: CPU mode never touches the GPU driver.
 *
 * A GPU driver bug can take the whole process down, and the voice keyboard
 * would then crash again on every dictation. While the GPU is in use, every
 * native call runs with [markerFile] present; a crash leaves it behind, and
 * the next process start finds it, switches the preference back to CPU, and
 * reports it through [recoveredFromGpuCrash].
 */
class ComputeDeviceSelector(
    private val gpuBuild: Boolean,
    private val preferences: DevicePreferenceStore,
    private val markerFile: File,
    private val setEnv: (name: String, value: String) -> Unit,
) : NativeCallPolicy {
    @Volatile
    var applied: ComputeDevice? = null
        private set

    /** True when this process started after a crash inside a GPU call. */
    val recoveredFromGpuCrash: Boolean

    init {
        recoveredFromGpuCrash = markerFile.exists()
        if (recoveredFromGpuCrash) {
            preferences.set(ComputeDevice.CPU)
            markerFile.delete()
        }
    }

    /** Whether the GPU option exists in this build at all. */
    val gpuAvailable: Boolean get() = gpuBuild

    fun preferred(): ComputeDevice = if (gpuBuild) preferences.get() else ComputeDevice.CPU

    fun setPreferred(device: ComputeDevice) {
        preferences.set(if (gpuBuild) device else ComputeDevice.CPU)
    }

    /** The preference differs from the device this process already committed to. */
    fun restartNeeded(): Boolean = applied.let { it != null && it != preferred() }

    @Synchronized
    override fun beforeFirstLoad() {
        if (applied != null) return
        val device = preferred()
        setEnv(DEVICE_ENV, device.ggmlName)
        if (device == ComputeDevice.CPU) setEnv(DISABLE_VULKAN_ENV, "1")
        applied = device
    }

    override fun <T> guard(block: () -> T): T {
        if (applied != ComputeDevice.GPU) return block()
        // Written before the call and removed after it returns or throws; a
        // process that dies inside the call leaves it for the next start.
        runCatching { markerFile.writeText("native GPU call in progress\n") }
        try {
            return block()
        } finally {
            markerFile.delete()
        }
    }

    companion object {
        const val DEVICE_ENV = "STARLING_GGML_DEVICE"
        const val DISABLE_VULKAN_ENV = "GGML_DISABLE_VULKAN"
    }
}
