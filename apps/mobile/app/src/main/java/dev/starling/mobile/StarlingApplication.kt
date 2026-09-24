package dev.starling.mobile

import android.app.ActivityManager
import android.app.Application
import android.content.ComponentCallbacks2
import android.content.Context
import dev.starling.mobile.engine.OnDeviceBackend
import dev.starling.mobile.engine.OnDeviceEngine
import dev.starling.mobile.network.BackendSettings
import dev.starling.mobile.network.TranscriptionCoordinator
import dev.starling.mobile.storage.RecordingStore
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

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

    // One worker for releases: repeated trims while a transcription holds the
    // engine queue behind each other instead of stacking waiting threads.
    private val releaseExecutor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "starling-model-release").apply { isDaemon = true }
    }

    override fun onCreate() {
        super.onCreate()
        recordings = RecordingStore(this)
        backendSettings = BackendSettings(this)
        onDeviceEngine = OnDeviceEngine(File(filesDir, "models"), memoryGate = ::memoryGate)
        modelDownloads = ModelDownloadController(onDeviceEngine)
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

    private companion object {
        /** Activations and graph buffers on top of the weights. */
        const val MODEL_WORKING_SET_BYTES = 128L * 1024 * 1024
    }
}

fun Context.starlingApplication(): StarlingApplication =
    applicationContext as StarlingApplication
