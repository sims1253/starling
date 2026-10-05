package dev.starling.mobile.engine

import java.util.concurrent.CopyOnWriteArraySet
import java.util.concurrent.Executor

/**
 * Lifetime policy for the on-device model (E13, #229): when it loads ahead of
 * use, how long it stays warm, and what the voice surfaces show about it.
 *
 * - **Preload.** [preload] is the speculative trigger of a voice surface
 *   becoming active (the keyboard shown, the recognizer popup or recorder
 *   opened). It loads and warms up the *active* installed model on [worker],
 *   never on the calling (UI) thread, and never opens the microphone or
 *   downloads anything. Concurrent requests collapse into the one in flight,
 *   and a model already resident is only touched (its idle timer re-armed).
 *   A recording never waits for a preload: the live session captures from
 *   the first sample and feeds the backlog once the model is ready (see
 *   [OnDeviceStreamSession]).
 * - **Failures.** A load that failed for a model is not retried
 *   speculatively for [retryFailedAfterMs] (a corrupt or oversized model
 *   must not reload on every keyboard show); selecting another model, or a
 *   recording, still tries at once. A GPU driver failure (#325, see
 *   [GpuFailure]) is fatal to speculative work for the rest of the process:
 *   it is reported as [State.DriverFailed] and never retried, because every
 *   retry into a wedged driver prolongs the wedge. Recordings stay safe
 *   either way; their audio is saved before any transcription.
 * - **Idle release.** Once the model has been idle (no use, no live
 *   session) for [idleReleaseMs] it is freed through
 *   [OnDeviceEngine.releaseIfIdle], which refuses when anything used the
 *   model since. Short app or keyboard switches inside that window keep the
 *   model warm; memory pressure still releases it earlier (the
 *   application's onTrimMemory), but never under a live recording.
 *
 * Thread-safe. [deliver] runs listener notifications (the main thread in
 * the app). The engine reports to this class as its [OnDeviceEngine.Observer]
 * with the engine lock held, so nothing here calls into the engine while
 * holding this object's monitor.
 */
class ModelLifetime(
    private val engine: Engine,
    private val worker: Executor,
    private val scheduler: Scheduler,
    private val deliver: (Runnable) -> Unit,
    private val clock: () -> Long = System::currentTimeMillis,
    private val idleReleaseMs: Long = IDLE_RELEASE_MS,
    private val retryFailedAfterMs: Long = RETRY_FAILED_AFTER_MS,
) : OnDeviceEngine.Observer {
    /** The engine surface this policy drives; [OnDeviceEngine] in the app. */
    interface Engine {
        fun activeModelName(): String?

        /** [OnDeviceEngine.preload]: runs only if [allowed] still holds under the engine lock. */
        fun preload(allowed: () -> Boolean): String?
        fun releaseIfIdle(generation: Long): Boolean
    }

    /** Delayed tasks for the idle timer. */
    fun interface Scheduler {
        fun schedule(delayMs: Long, task: Runnable): Cancellable
    }

    fun interface Cancellable {
        fun cancel()
    }

    sealed interface State {
        /** No model is resident; the next use loads it. */
        data object Unloaded : State

        /** [model] is loading (a preload or a recording started it). */
        data class Loading(val model: String) : State

        /** [model] is resident and warmed up. */
        data class Ready(val model: String) : State

        /** The last load of [model] failed. A recording still tries again. */
        data class Failed(val model: String?, val reason: String) : State

        /** The GPU driver failed (#325); speculative loads stop for this process. */
        data class DriverFailed(val reason: String) : State
    }

    enum class PreloadResult {
        /** A load (or a touch of the resident model) was queued. */
        QUEUED,

        /** A preload is already queued or running; this request joined it. */
        IN_FLIGHT,

        /** No model is installed, so there is nothing to preload. */
        NO_MODEL,

        /** The last load of this model failed recently; not retried speculatively yet. */
        RECENTLY_FAILED,

        /** A GPU driver failure was seen; speculative work stays off. */
        DRIVER_FAILED,
    }

    private val lock = Any()
    private val listeners = CopyOnWriteArraySet<(State) -> Unit>()

    // Guarded by [lock].
    private var state: State = State.Unloaded
    private var preloadInFlight = false
    private var failedModel: String? = null
    private var failedAtMillis = 0L
    private var idleTimer: Cancellable? = null

    fun state(): State = synchronized(lock) { state }

    /**
     * Adds [listener] and immediately delivers the current state to it (on
     * [deliver]). Remove it with [removeListener] when the surface goes away.
     */
    fun addListener(listener: (State) -> Unit) {
        listeners += listener
        val current = state()
        deliver { if (listener in listeners) listener(current) }
    }

    fun removeListener(listener: (State) -> Unit) {
        listeners -= listener
    }

    /**
     * Speculatively loads the active installed model, or keeps the resident
     * one warm. Cheap and safe to call from every activation callback: it
     * returns at once and does nothing while a preload is in flight.
     */
    fun preload(): PreloadResult {
        // File-system work, outside the monitor (see the class doc).
        val model = engine.activeModelName() ?: return PreloadResult.NO_MODEL
        synchronized(lock) {
            if (state is State.DriverFailed) return PreloadResult.DRIVER_FAILED
            if (preloadInFlight) return PreloadResult.IN_FLIGHT
            if (failedModel == model && clock() - failedAtMillis < retryFailedAfterMs) {
                return PreloadResult.RECENTLY_FAILED
            }
            preloadInFlight = true
        }
        try {
            worker.execute(::runPreload)
        } catch (t: Throwable) {
            // A rejected task (executor shut down) must not wedge preloads.
            synchronized(lock) { preloadInFlight = false }
            throw t
        }
        return PreloadResult.QUEUED
    }

    private fun runPreload() {
        try {
            // Blocking: waits for the engine lock (an in-flight recording or
            // transcription goes first) and then loads, or merely touches
            // the resident model. The outcome reaches [state] through the
            // observer callbacks below. A driver failure seen while this
            // waited for the lock cancels it there.
            engine.preload { synchronized(lock) { state !is State.DriverFailed } }
        } catch (_: Throwable) {
            // A throwing load (a broken native library) was already reported
            // through loadFailed, with the model it actually tried, so the
            // state leaves Loading and the backoff applies to that model.
        } finally {
            synchronized(lock) { preloadInFlight = false }
        }
    }

    // OnDeviceEngine.Observer: called with the engine lock held. Only state
    // updates and posted notifications happen here.

    override fun loading(model: String) {
        update { current ->
            cancelIdleTimerLocked()
            if (current is State.DriverFailed) current else State.Loading(model)
        }
    }

    override fun loaded(model: String) {
        update { current ->
            if (failedModel == model) failedModel = null
            if (current is State.DriverFailed) current else State.Ready(model)
        }
    }

    override fun loadFailed(model: String?, reason: String) {
        update { current ->
            failedModel = model
            failedAtMillis = clock()
            when {
                current is State.DriverFailed -> current
                isDriverFailure(reason) -> State.DriverFailed(reason)
                else -> State.Failed(model, reason)
            }
        }
    }

    override fun unloaded() {
        update { current ->
            cancelIdleTimerLocked()
            // A failure stays visible after the release that followed it.
            if (current is State.DriverFailed || current is State.Failed) current else State.Unloaded
        }
    }

    override fun engineFailed(error: String) {
        if (!isDriverFailure(error)) return
        update { current -> if (current is State.DriverFailed) current else State.DriverFailed(error) }
    }

    override fun idle(generation: Long) {
        synchronized(lock) {
            cancelIdleTimerLocked()
            if (idleReleaseMs <= 0) return
            idleTimer = scheduler.schedule(idleReleaseMs) {
                synchronized(lock) { idleTimer = null }
                // The engine refuses when anything used the model since
                // [generation]; a refused release changes nothing.
                runCatching { engine.releaseIfIdle(generation) }
            }
        }
    }

    private fun cancelIdleTimerLocked() {
        idleTimer?.cancel()
        idleTimer = null
    }

    private inline fun update(transition: (State) -> State) {
        val changed = synchronized(lock) {
            val next = transition(state)
            if (next == state) return
            state = next
            next
        }
        for (listener in listeners) {
            deliver { if (listener in listeners) listener(changed) }
        }
    }

    companion object {
        /**
         * A GPU driver failure (#325) that must stop speculative work. The
         * native allocation error mentions a degraded driver too, but an
         * allocation that merely did not fit is recoverable (another model,
         * or more free memory), and the native engine does not mark it as a
         * wedge either; only a lost device there is fatal.
         */
        internal fun isDriverFailure(reason: String): Boolean {
            if (!GpuFailure.matches(reason)) return false
            // The GPU memory preflight refuses a load that does not fit the
            // current budget; nothing is wedged.
            if (reason.contains(PREFLIGHT_FAILURE)) return false
            if (!reason.contains(ALLOCATION_FAILURE)) return true
            return reason.contains("VkResult -4") || reason.contains("device lost", ignoreCase = true)
        }

        /** Wording of the native vkAllocateMemory failure (cpp/fast/vk_runtime.cpp). */
        private const val ALLOCATION_FAILURE = "the model may not fit"

        /** Wording of the native GPU memory-budget preflight refusal (cpp/fast/vk_runtime.cpp). */
        private const val PREFLIGHT_FAILURE = "GPU memory preflight failed"

        /**
         * How long an idle model stays resident. Long enough to keep it warm
         * across a quick app or keyboard switch, short enough that a phone
         * left alone does not hold hundreds of MB, and the GPU device that
         * comes with the fast engine (#325), for long. To be tuned against
         * the Pixel measurements (warm return, resident memory, idle energy).
         */
        const val IDLE_RELEASE_MS = 60_000L

        /** A failed speculative load is retried at most this often. */
        const val RETRY_FAILED_AFTER_MS = 5 * 60_000L
    }
}
