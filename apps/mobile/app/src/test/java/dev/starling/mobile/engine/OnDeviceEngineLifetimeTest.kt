package dev.starling.mobile.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.RandomAccessFile
import java.nio.file.Files
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

/**
 * Engine lifetime hooks that need no native library: the memory gate runs
 * inside the engine lock, so a gate that blocks stands in for a long load.
 */
class OnDeviceEngineLifetimeTest {
    private fun modelDir(): File = Files.createTempDirectory("starling-lifetime-test").toFile().also { dir ->
        RandomAccessFile(File(dir, "parakeet.gguf"), "rw").use { it.setLength(ModelFiles.MIN_MODEL_BYTES) }
    }

    @Test
    fun startingALiveSessionNeverWaitsForALoadInProgress() {
        val loading = CountDownLatch(1)
        val release = CountDownLatch(1)
        val engine = OnDeviceEngine(
            modelDir(),
            memoryGate = {
                loading.countDown()
                release.await(10, TimeUnit.SECONDS)
                "refused for the test"
            },
            nativeSupport = { null },
        )
        val preload = thread { engine.prepare { false } }
        assertTrue(loading.await(5, TimeUnit.SECONDS))

        // The Record tap runs this on the main thread while the load holds the lock.
        val started = System.nanoTime()
        engine.liveSessionStarted()
        assertTrue(System.nanoTime() - started < TimeUnit.SECONDS.toNanos(1))

        release.countDown()
        preload.join(5_000)
        engine.liveSessionEnded(prepared = false)
    }

    @Test
    fun aFailedLoadIsReportedToTheObserver() {
        val failures = mutableListOf<Pair<String?, String>>()
        val engine = OnDeviceEngine(modelDir(), memoryGate = { "only 10 MB free" }, nativeSupport = { null })
        engine.observer = object : OnDeviceEngine.Observer {
            override fun loadFailed(model: String?, reason: String) {
                failures += model to reason
            }
        }
        val reason = engine.prepare { false }
        assertEquals(listOf("parakeet.gguf" to reason), failures)
    }

    @Test
    fun aDisallowedPreloadNeverTouchesTheEngine() {
        var gateCalls = 0
        val engine = OnDeviceEngine(modelDir(), memoryGate = { gateCalls++; "refused" }, nativeSupport = { null })
        assertEquals(null, engine.preload { false })
        assertEquals(0, gateCalls)
        engine.preload { true }
        assertEquals(1, gateCalls)
    }

    @Test
    fun aThrowingLoadIsReportedAsAFailedLoad() {
        val failures = mutableListOf<String>()
        val engine = OnDeviceEngine(modelDir(), memoryGate = { throw IllegalStateException("gate broke") }, nativeSupport = { null })
        engine.observer = object : OnDeviceEngine.Observer {
            override fun loadFailed(model: String?, reason: String) {
                failures += reason
            }
        }
        assertTrue(runCatching { engine.preload { true } }.isFailure)
        assertEquals(listOf("gate broke"), failures)
    }

    @Test
    fun aClosedSessionNeverLoads() {
        var gateCalls = 0
        val engine = OnDeviceEngine(modelDir(), memoryGate = { gateCalls++; null }, nativeSupport = { null })
        assertEquals("the live session was closed", engine.prepare { true })
        assertEquals(0, gateCalls)
    }

    @Test
    fun theIdleReleaseHasNothingToFreeWithoutAResidentModel() {
        val engine = OnDeviceEngine(modelDir(), memoryGate = { "refused" }, nativeSupport = { null })
        engine.prepare { false }
        assertFalse(engine.releaseIfIdle(0))
    }
}
