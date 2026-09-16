package dev.starling.mobile.network

import android.os.Handler
import android.os.Looper
import dev.starling.mobile.data.Recording
import dev.starling.mobile.engine.OnDeviceBackend
import dev.starling.mobile.storage.RecordingStore
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** Uploads only after local audio finalization and persists every outcome. */
class TranscriptionCoordinator(
    private val store: RecordingStore,
    private val settings: BackendSettings,
    private val onDevice: OnDeviceBackend,
) {
    private val executor: ExecutorService = Executors.newCachedThreadPool { runnable ->
        Thread(runnable, "starling-transcription").apply { isDaemon = true }
    }
    private val mainHandler = Handler(Looper.getMainLooper())
    private val client = InferenceClient()
    private val activeIds = ConcurrentHashMap.newKeySet<String>()

    fun transcribe(
        id: String,
        config: BackendConfig = settings.load(),
        callback: (Recording) -> Unit = {},
    ) {
        // A double tap must not race two responses that overwrite the same
        // durable raw-transcript field.
        if (!activeIds.add(id)) return
        val queued = runCatching { store.markTranscribing(id) }.getOrElse { exception ->
            activeIds.remove(id)
            return callbackFailure(id, exception.message ?: "Unable to queue the recording", callback)
        }
        executor.execute {
            try {
                val audioFile = store.audioFile(queued)
                val result = if (config.engine == TranscriptionEngine.ON_DEVICE) {
                    onDevice.transcribe(audioFile, config)
                } else {
                    client.transcribe(audioFile, config)
                }
                val completed = when (result) {
                    is InferenceResult.Success -> runCatching {
                        store.markTranscribed(id, result.rawTranscript)
                    }.getOrElse {
                        store.markFailed(id, "Transcript was received but could not be saved")
                    }
                    is InferenceResult.Failure -> runCatching {
                        store.markFailed(id, result.message)
                    }.getOrElse {
                        queued.copy(errorMessage = result.message)
                    }
                }
                mainHandler.post { callback(completed) }
            } finally {
                activeIds.remove(id)
            }
        }
    }

    fun shutdown() {
        executor.shutdownNow()
    }

    private fun callbackFailure(id: String, message: String, callback: (Recording) -> Unit) {
        val failed = runCatching { store.markFailed(id, message) }.getOrNull()
        if (failed != null) {
            mainHandler.post { callback(failed) }
        }
    }
}
