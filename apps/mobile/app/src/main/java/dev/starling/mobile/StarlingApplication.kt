package dev.starling.mobile

import android.app.ActivityManager
import android.app.Application
import android.content.ComponentCallbacks2
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.util.Log
import dev.starling.mobile.engine.ModelLifetime
import dev.starling.mobile.engine.OnDeviceBackend
import dev.starling.mobile.engine.OnDeviceEngine
import dev.starling.mobile.network.BackendConfig
import dev.starling.mobile.network.BackendSettings
import dev.starling.mobile.network.TranscriptionCoordinator
import dev.starling.mobile.network.TranscriptionEngine
import dev.starling.mobile.storage.RecordingStore
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit

class StarlingApplication : Application() {
    lateinit var recordings: RecordingStore
        private set
    lateinit var backendSettings: BackendSettings
        private set
    lateinit var transcription: TranscriptionCoordinator
        private set
    lateinit var onDeviceEngine: OnDeviceEngine
        private set
    lateinit var modelDownloads: ModelDownloadController
        private set
    lateinit var modelLifetime: ModelLifetime
        private set

    private val mainHandler = Handler(Looper.getMainLooper())

    // One worker for releases: repeated trims while a transcription holds the
    // engine queue behind each other instead of stacking waiting threads.
    private val releaseExecutor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "starling-model-release").apply { isDaemon = true }
    }

    // Speculative loads (ModelLifetime.preload). One thread at most, and none
    // while nothing is queued: it exits after a short idle period.
    private val preloadExecutor = ThreadPoolExecutor(1, 1, 5, TimeUnit.SECONDS, LinkedBlockingQueue()) { runnable ->
        Thread(runnable, "starling-model-preload").apply { isDaemon = true }
    }.apply { allowCoreThreadTimeOut(true) }

    override fun onCreate() {
        super.onCreate()
        recordings = RecordingStore(this)
        backendSettings = BackendSettings(this)
        onDeviceEngine = OnDeviceEngine(
            File(filesDir, "models"),
            memoryGate = ::memoryGate,
            keepAwake = partialWakeLock(),
        )
        modelDownloads = ModelDownloadController(onDeviceEngine)
        modelLifetime = ModelLifetime(
            engine = object : ModelLifetime.Engine {
                override fun preload(allowed: (model: String) -> Boolean) = onDeviceEngine.preload(allowed)
                override fun releaseIfIdle(generation: Long) = onDeviceEngine.releaseIfIdle(generation)
            },
            worker = preloadExecutor,
            // The timer lives on the main looper (no thread of its own); the
            // release itself waits for the engine lock, so it runs off it.
            scheduler = { delayMs, task ->
                val post = Runnable { releaseExecutor.execute(task) }
                mainHandler.postDelayed(post, delayMs)
                ModelLifetime.Cancellable { mainHandler.removeCallbacks(post) }
            },
            deliver = { mainHandler.post(it) },
            logFailure = { t -> runCatching { Log.w(TAG, "on-device model preload failed", t) } },
        )
        onDeviceEngine.observer = modelLifetime
        transcription = TranscriptionCoordinator(
            recordings,
            backendSettings,
            OnDeviceBackend(onDeviceEngine),
        )
    }

    /**
     * Resource budget for the on-device model (E13): the resident model is
     * hundreds of MB of native memory, so it is released under real pressure
     * (TRIM_MEMORY_RUNNING_CRITICAL) or once the process is in the background
     * LRU list (TRIM_MEMORY_BACKGROUND and above). Merely hiding the UI
     * (TRIM_MEMORY_UI_HIDDEN) keeps it, so switching apps mid-dictation does
     * not force a reload. The next transcription reloads it.
     */
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        @Suppress("DEPRECATION") // Still delivered; the only signal for a foreground process in trouble.
        val underPressure = level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL ||
            level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND
        if (underPressure) releaseOnDeviceModel()
    }

    @Deprecated("Deprecated in Java")
    override fun onLowMemory() {
        @Suppress("DEPRECATION")
        super.onLowMemory()
        releaseOnDeviceModel()
    }

    // releaseWhenIdle waits for an in-flight transcription; never on the main thread.
    private fun releaseOnDeviceModel() {
        releaseExecutor.execute { onDeviceEngine.releaseWhenIdle() }
    }

    /**
     * A voice surface became active (the keyboard shown, the recognizer popup
     * or the recorder opened): start loading the selected local model so the
     * first words of a recording do not wait for it. A no-op for the server
     * engine; never opens the microphone, never downloads (see ModelLifetime).
     */
    fun preloadOnDeviceModel() {
        if (backendSettings.load().engine != TranscriptionEngine.ON_DEVICE) return
        runCatching { modelLifetime.preload() }
            .onFailure { t ->
                // runCatching around Log: it is a stub in local unit tests,
                // and a failed diagnostic must not take down the surface.
                runCatching { Log.w(TAG, "on-device model preload failed", t) }
            }
    }

    /**
     * Whether [config] streams on-device while the model is still loading:
     * the one predicate behind the voice surfaces' "loading" status (the
     * recorder, the recognizer popup and the keyboard), so they cannot
     * drift apart. True while the model is Loading, and while it is
     * Unloaded — a take that starts cold loads in its own prepare(), which
     * reports its state asynchronously, so Unloaded at the Record tap means
     * the load is about to start. Failed and DriverFailed are not loading:
     * nothing retries them speculatively, so the surface must not promise
     * text that will not come. Audio is saved from the first sample either
     * way.
     */
    fun isOnDeviceModelLoading(config: BackendConfig): Boolean =
        config.engine == TranscriptionEngine.ON_DEVICE && isLoading(modelLifetime.state())

    /**
     * Refuses a model load that clearly cannot fit: the model's file size is
     * a lower bound on its resident size, so loading it with less memory
     * available would only end with the process killed mid-load. Advisory
     * only: free memory can still drop between this check and the load, and
     * the working-set allowance is an estimate for the Parakeet 0.6B class.
     */
    private fun memoryGate(modelBytes: Long): String? {
        val manager = getSystemService(ActivityManager::class.java) ?: return null
        val info = ActivityManager.MemoryInfo().also(manager::getMemoryInfo)
        if (info.availMem >= modelBytes + MODEL_WORKING_SET_BYTES) return null
        val mb = 1024 * 1024
        return getString(
            R.string.on_device_memory_error,
            (modelBytes + MODEL_WORKING_SET_BYTES) / mb,
            info.availMem / mb,
        )
    }

    /**
     * The engine's keepAwake: kernel suspend with GPU work outstanding wedges
     * the Pixel's GPU driver (benchmarks/fast_engine/RESEARCH_LOG.md, P3-4).
     * Reference-counted, so nested holds are fine; the timeout only bounds a
     * leaked hold.
     */
    private fun partialWakeLock(): () -> AutoCloseable {
        val lock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "starling:on-device-engine")
        return {
            lock.acquire(WAKE_LOCK_TIMEOUT_MS)
            AutoCloseable { if (lock.isHeld) lock.release() }
        }
    }

    internal companion object {
        private const val TAG = "StarlingApplication"

        /** Activations and graph buffers on top of the weights. */
        const val MODEL_WORKING_SET_BYTES = 128L * 1024 * 1024

        /** Upper bound on one wake-lock hold: a long recording's chunked pass, with margin. */
        private const val WAKE_LOCK_TIMEOUT_MS = 10L * 60 * 1000

        /**
         * The state half of [isOnDeviceModelLoading], split out for its
         * unit test (the application itself needs the main looper).
         */
        fun isLoading(state: ModelLifetime.State): Boolean = when (state) {
            is ModelLifetime.State.Loading -> true
            ModelLifetime.State.Unloaded -> true
            is ModelLifetime.State.Ready,
            is ModelLifetime.State.Failed,
            is ModelLifetime.State.DriverFailed,
            -> false
        }
    }
}

fun Context.starlingApplication(): StarlingApplication =
    applicationContext as StarlingApplication
