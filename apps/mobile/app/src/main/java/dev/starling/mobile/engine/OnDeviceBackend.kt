package dev.starling.mobile.engine

import dev.starling.mobile.network.BackendConfig
import dev.starling.mobile.network.InferenceResult
import dev.starling.mobile.network.StreamEvent
import dev.starling.mobile.network.StreamSession
import java.io.File

/**
 * Runs transcription through the on-device engine, ignoring connection
 * settings. Never throws: the engine's first use loads the JNI library, and a
 * missing or broken native build must fail the recording the way the remote
 * client does, not crash the caller.
 */
class OnDeviceBackend(private val engine: OnDeviceEngine) {
    fun transcribe(audioFile: File, @Suppress("UNUSED_PARAMETER") config: BackendConfig): InferenceResult =
        runCatching { engine.transcribe(audioFile) }.getOrElse {
            InferenceResult.Failure(
                "The on-device engine could not run: ${it.message ?: it::class.java.simpleName}",
                false,
            )
        }

    /**
     * Starts live on-device transcription for a capture that is about to
     * begin, or null when no model is imported (the recording then goes
     * through [transcribe] after Stop, which reports the missing model).
     * [events] is invoked on the session's worker thread.
     */
    fun beginStreaming(events: (StreamEvent) -> Unit): StreamSession? =
        if (engine.hasModel()) OnDeviceStreamSession(engine, events).start() else null
}
