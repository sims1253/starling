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
    private val loadConfig: () -> BackendConfig,
    private val onDevice: OnDeviceBackend,
    private val postToMain: (Runnable) -> Unit = { runnable ->
        Handler(Looper.getMainLooper()).post(runnable)
    },
) {
    private val executor: ExecutorService = Executors.newCachedThreadPool { runnable ->
        Thread(runnable, "starling-transcription").apply { isDaemon = true }
    }
    private val client = InferenceClient()
    private val activeIds = ConcurrentHashMap.newKeySet<String>()

    fun transcribe(
        id: String,
        config: BackendConfig = loadConfig(),
        callback: (Recording) -> Unit = {},
    ) {
        // A double tap must not race two responses that overwrite the same
        // durable raw-transcript field. The rejected call still reports the
        // current store state so its caller is never left waiting: the first
        // request keeps running and delivers the final outcome itself.
        if (!activeIds.add(id)) {
            runCatching { store.get(id) }.getOrNull()?.let { current ->
                postToMain(Runnable { callback(current) })
            }
            return
        }
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
                postToMain(Runnable { callback(completed) })
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
            postToMain(Runnable { callback(failed) })
        }
    }
}
