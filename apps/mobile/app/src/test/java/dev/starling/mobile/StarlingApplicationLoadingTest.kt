package dev.starling.mobile

import dev.starling.mobile.engine.ModelLifetime
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The "loading" classification behind every voice surface's status (#229):
 * Loading counts, and so does Unloaded — a cold take loads in its own
 * prepare() right after the status is first read — while Failed and
 * DriverFailed must not claim loading forever: nothing retries them
 * speculatively.
 */
class StarlingApplicationLoadingTest {
    @Test
    fun loadingAndUnloadedCountAsLoading() {
        assertTrue(StarlingApplication.isLoading(ModelLifetime.State.Loading("parakeet.gguf")))
        assertTrue(StarlingApplication.isLoading(ModelLifetime.State.Unloaded))
    }

    @Test
    fun failedDriverFailedAndReadyAreNotLoading() {
        assertFalse(StarlingApplication.isLoading(ModelLifetime.State.Ready("parakeet.gguf")))
        assertFalse(StarlingApplication.isLoading(ModelLifetime.State.Failed("parakeet.gguf", "out of memory")))
        assertFalse(StarlingApplication.isLoading(ModelLifetime.State.DriverFailed("device lost")))
    }
}
