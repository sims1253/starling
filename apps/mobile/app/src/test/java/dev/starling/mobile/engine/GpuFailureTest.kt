package dev.starling.mobile.engine

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GpuFailureTest {
    @Test
    fun theNativeGpuFailureMessagesMatch() {
        listOf(
            "vkWaitForFences failed (VkResult 2) (GPU work did not finish within 20000 ms; the driver may be wedged)",
            "vkWaitForFences failed (VkResult -4) (device lost: the GPU driver has failed; a device restart is required)",
            "vkWaitForFences failed (VkResult -2) (out of device memory: the GPU driver is likely wedged; " +
                "a device restart clears it — do not keep retrying)",
            "fast engine: a previous process observed GPU driver failure (x) less than 15 minutes ago; " +
                "restart the app/device before retrying",
            "fast engine: GPU driver failure earlier in this process (x); restart the app/device before retrying",
        ).forEach { assertTrue(it, GpuFailure.matches(it)) }
    }

    @Test
    fun otherErrorsDoNotMatch() {
        assertFalse(GpuFailure.matches(null))
        assertFalse(GpuFailure.matches("audio is empty"))
        assertFalse(GpuFailure.matches("fast moss: decode did not terminate"))
        assertFalse(GpuFailure.matches("the model could not be loaded"))
    }
}
