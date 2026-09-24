package dev.starling.mobile

import android.os.Handler
import android.os.Looper
import android.util.Log
import dev.starling.mobile.engine.ModelDownload
import dev.starling.mobile.engine.ModelDownloader
import dev.starling.mobile.engine.OnDeviceEngine

/**
 * App-scoped owner of the one model download that may run at a time, so it
 * outlives the Activity (rotation, leaving the screen). Listeners are called
 * on the main thread; a newly added listener immediately gets the current
 * state. The download itself resumes across process death through
 * [OnDeviceEngine.downloadFile], so being killed in the background only
 * pauses it.
 */
class ModelDownloadController(private val engine: OnDeviceEngine) {
    sealed interface State {
        data object Idle : State
        data class Running(val bytes: Long, val total: Long, val verifying: Boolean) : State
        data class Finished(val result: OnDeviceEngine.ImportResult) : State
        data class Failed(val reason: String) : State
        data object Paused : State
    }

    private val main = Handler(Looper.getMainLooper())
    private val listeners = mutableListOf<(State) -> Unit>()
    private var state: State = State.Idle
    @Volatile private var downloader: ModelDownloader? = null

    val isRunning: Boolean get() = state is State.Running

    /** Bytes already on disk from an earlier, unfinished attempt. */
    fun resumableBytes(spec: ModelDownload): Long =
        engine.downloadFile(spec).length().coerceAtMost(spec.sizeBytes)

    fun addListener(listener: (State) -> Unit) {
        listeners += listener
        listener(state)
    }

    fun removeListener(listener: (State) -> Unit) {
        listeners -= listener
    }

    /**
     * Renames a pre-0.2.2 copy of [spec] to its catalog name (see
     * [OnDeviceEngine.recognizeLegacyDownload]) and re-renders listeners.
     * Blocking; any thread. Never throws: a failure only leaves the old name.
     */
    fun recognizeLegacyDownload(spec: ModelDownload) {
        val renamed = runCatching { engine.recognizeLegacyDownload(spec) }
            .onFailure { Log.w(TAG, "legacy model recognition failed", it) }
            .getOrDefault(false)
        if (renamed) main.post { publish(state) }
    }

    /** Starts (or resumes) downloading [spec]; a no-op while one is running. Main thread. */
    fun start(spec: ModelDownload) {
        if (isRunning) return
        val job = ModelDownloader()
        downloader = job
        publish(State.Running(resumableBytes(spec), spec.sizeBytes, verifying = false))
        Thread({ run(job, spec) }, "starling-model-download").apply { isDaemon = true }.start()
    }

    /** Stops the running download; the partial file is kept for a resume. Any thread. */
    fun cancel() {
        downloader?.cancel()
    }

    private fun run(job: ModelDownloader, spec: ModelDownload) {
        val partial = engine.downloadFile(spec)
        val result = runCatching {
            job.download(spec, partial) { bytes, total, verifying ->
                main.post { if (downloader === job) publish(State.Running(bytes, total, verifying)) }
            }
        }.getOrElse { error ->
            Log.e(TAG, "model download failed unexpectedly", error)
            ModelDownloader.Result.Failed(error.message ?: error::class.java.simpleName)
        }
        val next = when (result) {
            is ModelDownloader.Result.Done -> State.Finished(
                runCatching { engine.adoptDownloaded(result.file, spec.fileName) }.getOrElse { error ->
                    Log.e(TAG, "downloaded model import failed unexpectedly", error)
                    OnDeviceEngine.ImportResult.Rejected(
                        "The downloaded model could not be imported: ${error.message}",
                        OnDeviceEngine.ImportStage.PROMOTE,
                    )
                },
            )
            is ModelDownloader.Result.Failed -> State.Failed(result.reason)
            ModelDownloader.Result.Cancelled -> State.Paused
        }
        main.post {
            // Only the current job may end the download (a paused job's
            // late result must not override a newer run).
            if (downloader !== job) return@post
            downloader = null
            publish(next)
        }
    }

    // Progress posts are tagged with their job in run(), so a stale post
    // from a paused run can never overwrite a newer state.
    private fun publish(next: State) {
        state = next
        for (listener in listeners.toList()) listener(next)
    }

    private companion object {
        const val TAG = "StarlingModelDownload"
    }
}
