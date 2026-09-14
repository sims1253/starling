package dev.starling.mobile

import dev.starling.mobile.ui.RequestGenerationGuard
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RequestGenerationGuardTest {
    @Test
    fun newerCaptureInvalidatesOlderCallback() {
        val guard = RequestGenerationGuard()
        val first = guard.begin()
        val second = guard.begin()

        assertFalse(guard.isCurrent(first))
        assertTrue(guard.isCurrent(second))
    }
}
