package dev.starling.mobile.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.RejectedExecutionException

/**
 * The lifetime policy against a fake engine, a manual executor and a
 * manual timer: preload dedup, failure backoff, #325 driver failures that
 * are never retried speculatively, and the generation-guarded idle release.
 */
class ModelLifetimeTest {
    private class FakeEngine : ModelLifetime.Engine {
        var model: String? = "parakeet.gguf"
        var prepares = 0
        val releases = mutableListOf<Long>()
        var onPrepare: () -> String? = { null }

        override fun preload(allowed: (model: String) -> Boolean): String? {
            val active = model ?: return null
            if (!allowed(active)) return null
            prepares++
            return onPrepare()
        }
        override fun releaseIfIdle(generation: Long): Boolean {
            releases += generation
            return true
        }
    }

    private class Timer(val delayMs: Long, val task: Runnable) {
        var cancelled = false
    }

    private val engine = FakeEngine()
    private val queued = ArrayDeque<Runnable>()
    private val timers = mutableListOf<Timer>()
    private var now = 0L
    private val states = mutableListOf<ModelLifetime.State>()

    private val lifetime = ModelLifetime(
        engine = engine,
        worker = { queued.addLast(it) },
        scheduler = { delayMs, task ->
            val timer = Timer(delayMs, task)
            timers += timer
            ModelLifetime.Cancellable { timer.cancelled = true }
        },
        deliver = { it.run() },
        clock = { now },
        idleReleaseMs = 60_000L,
        retryFailedAfterMs = 300_000L,
    )

    private fun runQueued() {
        while (queued.isNotEmpty()) queued.removeFirst().run()
    }

    private fun liveTimers() = timers.filterNot { it.cancelled }

    @Test
    fun concurrentPreloadsCollapseIntoOneLoad() {
        assertEquals(ModelLifetime.PreloadResult.QUEUED, lifetime.preload())
        assertEquals(ModelLifetime.PreloadResult.IN_FLIGHT, lifetime.preload())
        assertEquals(ModelLifetime.PreloadResult.IN_FLIGHT, lifetime.preload())
        runQueued()
        assertEquals(1, engine.prepares)

        // Once it finished, the next activation only touches the model again.
        assertEquals(ModelLifetime.PreloadResult.QUEUED, lifetime.preload())
        runQueued()
        assertEquals(2, engine.prepares)
    }

    @Test
    fun nothingIsPreloadedWithoutAnInstalledModel() {
        engine.model = null
        assertEquals(ModelLifetime.PreloadResult.QUEUED, lifetime.preload())
        runQueued()
        assertEquals(0, engine.prepares)
    }

    @Test
    fun aRejectedPreloadTaskIsLoggedNotThrownAndUnwedges() {
        val failures = mutableListOf<Throwable>()
        var workerRejects = true
        val rejectingLifetime = ModelLifetime(
            engine = engine,
            worker = { if (workerRejects) throw RejectedExecutionException("shut down") else queued.addLast(it) },
            scheduler = { _, _ -> ModelLifetime.Cancellable { } },
            deliver = { it.run() },
            logFailure = { failures += it },
        )
        assertEquals(ModelLifetime.PreloadResult.REJECTED, rejectingLifetime.preload())
        assertEquals("shut down", failures.single().message)

        // The rejection reset the in-flight flag: once the worker accepts
        // tasks again, a later activation queues instead of joining a ghost.
        workerRejects = false
        assertEquals(ModelLifetime.PreloadResult.QUEUED, rejectingLifetime.preload())
        runQueued()
        assertEquals(1, engine.prepares)
    }

    @Test
    fun theStateFollowsTheEngineAndReachesListeners() {
        lifetime.addListener { states += it }
        lifetime.loading("a.gguf")
        lifetime.loaded("a.gguf")
        lifetime.unloaded()
        assertEquals(
            listOf(
                ModelLifetime.State.Unloaded,
                ModelLifetime.State.Loading("a.gguf"),
                ModelLifetime.State.Ready("a.gguf"),
                ModelLifetime.State.Unloaded,
            ),
            states,
        )
    }

    @Test
    fun aFailedModelIsNotReloadedOnEveryActivation() {
        lifetime.loadFailed("parakeet.gguf", "not enough free memory")
        assertEquals(ModelLifetime.State.Failed("parakeet.gguf", "not enough free memory"), lifetime.state())
        // The backoff is decided on the worker, so the preload still queues.
        assertEquals(ModelLifetime.PreloadResult.QUEUED, lifetime.preload())
        runQueued()
        assertEquals(0, engine.prepares)

        // Another model is tried at once.
        engine.model = "other.gguf"
        assertEquals(ModelLifetime.PreloadResult.QUEUED, lifetime.preload())
        runQueued()
        assertEquals(1, engine.prepares)

        // The failed one again after the backoff.
        engine.model = "parakeet.gguf"
        now += 300_000L
        assertEquals(ModelLifetime.PreloadResult.QUEUED, lifetime.preload())
        runQueued()
        assertEquals(2, engine.prepares)
    }

    @Test
    fun aRecordingThatLoadsTheModelClearsTheFailure() {
        lifetime.loadFailed("parakeet.gguf", "transient")
        lifetime.loading("parakeet.gguf")
        lifetime.loaded("parakeet.gguf")
        assertEquals(ModelLifetime.State.Ready("parakeet.gguf"), lifetime.state())
        assertEquals(ModelLifetime.PreloadResult.QUEUED, lifetime.preload())
        runQueued()
        assertEquals(1, engine.prepares)
    }

    @Test
    fun aGpuDriverFailureIsPropagatedAndNeverRetriedSpeculatively() {
        val wedged = "vkWaitForFences failed (VkResult 2) (GPU work did not finish within 20000 ms; " +
            "the driver may be wedged)"
        lifetime.loaded("parakeet.gguf")
        lifetime.engineFailed(wedged)
        assertEquals(ModelLifetime.State.DriverFailed(wedged), lifetime.state())
        assertEquals(ModelLifetime.PreloadResult.DRIVER_FAILED, lifetime.preload())

        // Neither time, an unload, nor a later load clears it.
        now += 3_600_000L
        lifetime.unloaded()
        lifetime.loading("parakeet.gguf")
        lifetime.loaded("parakeet.gguf")
        assertEquals(ModelLifetime.State.DriverFailed(wedged), lifetime.state())
        assertEquals(ModelLifetime.PreloadResult.DRIVER_FAILED, lifetime.preload())
        assertTrue(queued.isEmpty())
    }

    @Test
    fun aGpuFailureDuringTheLoadIsFatalToo() {
        lifetime.loadFailed("parakeet.gguf", "The on-device engine failed its warmup: device lost: the GPU driver has failed")
        assertTrue(lifetime.state() is ModelLifetime.State.DriverFailed)
    }

    @Test
    fun anOrdinaryEngineErrorIsNotADriverFailure() {
        lifetime.loaded("parakeet.gguf")
        lifetime.engineFailed("audio is empty")
        assertEquals(ModelLifetime.State.Ready("parakeet.gguf"), lifetime.state())
    }

    @Test
    fun anIdleModelIsReleasedForTheGenerationThatWentIdle() {
        lifetime.idle(7)
        val timer = liveTimers().single()
        assertEquals(60_000L, timer.delayMs)
        timer.task.run()
        assertEquals(listOf(7L), engine.releases)
    }

    @Test
    fun aNewUseReArmsTheIdleTimer() {
        lifetime.idle(7)
        lifetime.idle(9)
        val timer = liveTimers().single()
        timer.task.run()
        assertEquals(listOf(9L), engine.releases)
    }

    @Test
    fun aStaleIdleTimerCannotEraseTheNewerOne() {
        lifetime.idle(7)
        lifetime.idle(9)
        // Timer 1 fires late, after the newer idle re-armed: its release is
        // refused by the generation, and it must not clear timer 2 either.
        timers.first().task.run()
        assertEquals(listOf(7L), engine.releases)
        lifetime.loading("parakeet.gguf")
        assertTrue(liveTimers().isEmpty())
    }

    @Test
    fun aLoadOrAnUnloadCancelsThePendingRelease() {
        lifetime.idle(7)
        lifetime.loading("parakeet.gguf")
        assertTrue(liveTimers().isEmpty())
        lifetime.idle(8)
        lifetime.unloaded()
        assertTrue(liveTimers().isEmpty())
    }

    @Test
    fun aThrowingLoadIsAFailureWithBackoffNotAnEndlessLoading() {
        // As OnDeviceEngine does: a throwing load is reported, then rethrown.
        engine.onPrepare = {
            lifetime.loading("parakeet.gguf")
            lifetime.loadFailed("parakeet.gguf", "no libstarling_jni")
            throw UnsatisfiedLinkError("no libstarling_jni")
        }
        lifetime.preload()
        runQueued()
        assertEquals(ModelLifetime.State.Failed("parakeet.gguf", "no libstarling_jni"), lifetime.state())
        // The failed model is inside its backoff: queued, but never loaded.
        lifetime.preload()
        runQueued()
        assertEquals(1, engine.prepares)
        now += 300_000L
        lifetime.preload()
        runQueued()
        assertEquals(2, engine.prepares)
    }

    @Test
    fun aPreloadQueuedBeforeALoadFailureNeverRetriesIt() {
        assertEquals(ModelLifetime.PreloadResult.QUEUED, lifetime.preload())
        // A recording fails this model while the preload waits for the engine;
        // the backoff check runs on the worker, after that failure.
        lifetime.loadFailed("parakeet.gguf", "no libstarling_jni")
        runQueued()
        assertEquals(0, engine.prepares)
    }

    @Test
    fun aPreloadQueuedBeforeADriverFailureNeverRuns() {
        assertEquals(ModelLifetime.PreloadResult.QUEUED, lifetime.preload())
        // A recording hits the wedge while the preload waits for the engine.
        lifetime.engineFailed("vkWaitForFences failed (VkResult -4) (device lost: the GPU driver has failed)")
        runQueued()
        assertEquals(0, engine.prepares)
    }

    @Test
    fun aModelThatDoesNotFitTheGpuIsNotADriverFailure() {
        val oom = "fast engine unavailable: vkAllocateMemory failed (VkResult -2) (900 MiB) — the model may not fit, " +
            "or the GPU driver is degraded; a device restart clears the latter (do not keep retrying)"
        lifetime.loadFailed("big.gguf", oom)
        assertEquals(ModelLifetime.State.Failed("big.gguf", oom), lifetime.state())
        engine.model = "small.gguf"
        assertEquals(ModelLifetime.PreloadResult.QUEUED, lifetime.preload())
        runQueued()
        assertEquals(1, engine.prepares)

        val lost = oom.replace("VkResult -2", "VkResult -4")
        assertTrue(ModelLifetime.isDriverFailure(lost))
        assertFalse(
            ModelLifetime.isDriverFailure(
                "fast engine unavailable: fast engine: GPU memory preflight failed: need 900 MiB, 400 MiB " +
                    "available - the GPU driver may be degraded; restarting the app/device clears it",
            ),
        )
    }

    @Test
    fun aRemovedListenerHearsNothing() {
        val listener: (ModelLifetime.State) -> Unit = { states += it }
        lifetime.addListener(listener)
        lifetime.removeListener(listener)
        lifetime.loaded("parakeet.gguf")
        assertEquals(listOf<ModelLifetime.State>(ModelLifetime.State.Unloaded), states)
        assertFalse(states.any { it is ModelLifetime.State.Ready })
        assertNull(states.firstOrNull { it is ModelLifetime.State.Failed })
    }
}
