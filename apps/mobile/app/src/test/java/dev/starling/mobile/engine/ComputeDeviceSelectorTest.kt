package dev.starling.mobile.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files

class ComputeDeviceSelectorTest {
    private class MemoryStore(var device: ComputeDevice = ComputeDevice.CPU) : DevicePreferenceStore {
        override fun get() = device
        override fun set(device: ComputeDevice) {
            this.device = device
        }
    }

    private val env = mutableMapOf<String, String>()
    private fun marker(): File = File(Files.createTempDirectory("starling-gpu").toFile(), "gpu_call.marker")

    private fun selector(gpuBuild: Boolean, store: DevicePreferenceStore, marker: File) =
        ComputeDeviceSelector(gpuBuild, store, marker) { name, value -> env[name] = value }

    @Test
    fun theDeviceIsAlwaysSetExplicitlySoAGpuIsNeverAutoPicked() {
        val selector = selector(gpuBuild = true, store = MemoryStore(ComputeDevice.CPU), marker = marker())

        selector.beforeFirstLoad()

        assertEquals("cpu", env[ComputeDeviceSelector.DEVICE_ENV])
        // CPU mode must not even register ggml's Vulkan backend.
        assertEquals("1", env[ComputeDeviceSelector.DISABLE_VULKAN_ENV])
        assertEquals(ComputeDevice.CPU, selector.applied)
    }

    @Test
    fun theGpuPreferenceAppliesVulkanOnceAndLaterChangesNeedARestart() {
        val store = MemoryStore(ComputeDevice.GPU)
        val selector = selector(gpuBuild = true, store = store, marker = marker())

        selector.beforeFirstLoad()
        assertEquals("Vulkan0", env[ComputeDeviceSelector.DEVICE_ENV])
        assertFalse(ComputeDeviceSelector.DISABLE_VULKAN_ENV in env)
        assertFalse(selector.restartNeeded())

        selector.setPreferred(ComputeDevice.CPU)
        selector.beforeFirstLoad()
        assertEquals("Vulkan0", env[ComputeDeviceSelector.DEVICE_ENV])
        assertTrue(selector.restartNeeded())
    }

    @Test
    fun aBuildWithoutVulkanIgnoresAStoredGpuPreference() {
        val selector = selector(gpuBuild = false, store = MemoryStore(ComputeDevice.GPU), marker = marker())

        selector.beforeFirstLoad()

        assertEquals(ComputeDevice.CPU, selector.preferred())
        assertEquals("cpu", env[ComputeDeviceSelector.DEVICE_ENV])
    }

    @Test
    fun aCrashInsideAGpuCallFallsBackToTheCpuOnTheNextStart() {
        val marker = marker()
        val store = MemoryStore(ComputeDevice.GPU)
        val crashed = selector(gpuBuild = true, store = store, marker = marker)
        crashed.beforeFirstLoad()
        // Simulate the process dying inside the call: the marker is written
        // but the block never returns normally.
        var sawMarker = false
        crashed.guard { sawMarker = marker.exists() }
        assertTrue(sawMarker)
        assertFalse(marker.exists())
        marker.writeText("left behind by a crash")

        val next = selector(gpuBuild = true, store = store, marker = marker)

        assertTrue(next.recoveredFromGpuCrash)
        assertEquals(ComputeDevice.CPU, store.device)
        assertFalse(marker.exists())
    }

    @Test
    fun anExceptionFromAGpuCallIsNotMistakenForACrash() {
        val marker = marker()
        val selector = selector(gpuBuild = true, store = MemoryStore(ComputeDevice.GPU), marker = marker)
        selector.beforeFirstLoad()

        runCatching { selector.guard { throw IllegalStateException("load failed") } }

        assertFalse(marker.exists())
    }

    @Test
    fun cpuCallsNeverTouchTheMarker() {
        val marker = marker()
        val selector = selector(gpuBuild = true, store = MemoryStore(ComputeDevice.CPU), marker = marker)
        selector.beforeFirstLoad()

        var sawMarker = true
        selector.guard { sawMarker = marker.exists() }

        assertFalse(sawMarker)
    }
}
