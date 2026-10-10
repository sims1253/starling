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
 *
 * Every transcription is bounded ([BoundedTranscription], #356): a model
 * call that hangs fails the attempt, retryable and with the audio kept,
 * and a result it returns after that is dropped.
 */
class OnDeviceBackend(
    private val engine: OnDeviceEngine,
    private val bound: BoundedTranscription = BoundedTranscription(engine::nativeCallAgeMillis),
) {
    fun transcribe(audioFile: File, config: BackendConfig): InferenceResult = runCatching {
        bound.run(BoundedTranscription.budgetFor(audioFile.length())) { attempt ->
            runCatching { engine.transcribe(audioFile, config.onDeviceModel, attempt) }.getOrElse {
                InferenceResult.Failure(
                    "The on-device engine could not run: ${it.message ?: it::class.java.simpleName}",
                    false,
                )
            }
        }
    }.getOrElse {
        // The bound itself failed (no thread for the call, an interrupt):
        // nothing is wrong with the recording, so it stays retryable.
        if (it is InterruptedException) Thread.currentThread().interrupt()
        InferenceResult.Failure(
            "The on-device transcription could not run (${it.message ?: it::class.java.simpleName}). " +
                "The recording is kept; retry to transcribe it again.",
            true,
        )
    }

    /** The model a transcription without an explicit one uses; null when none is installed. */
    fun activeModelName(): String? = runCatching { engine.activeModelName() }.getOrNull()

    /**
     * Starts live on-device transcription for a capture that is about to
     * begin, or null when no model is imported (the recording then goes
     * through [transcribe] after Stop, which reports the missing model).
     * [events] is invoked on the session's worker thread; callers that need
     * main-thread delivery repost it (TranscriptionCoordinator does).
     * [savedAudio] is the file the capture writes (its partial WAV): audio the
     * session cannot hold in memory while the model loads is read back from
     * it instead of interrupting the stream.
     */
    fun beginStreaming(savedAudio: File? = null, events: (StreamEvent) -> Unit): StreamSession? =
        if (engine.hasModel()) {
            // StreamDebug's cadence override and trace are debug-build only.
            val streamer = StreamDebug.streamer()
            OnDeviceStreamSession(
                engine,
                events,
                streamer = streamer,
                backlog = savedAudio?.let { file -> { SavedAudioBacklog(file) } },
                trace = StreamDebug.trace(streamer),
            ).start()
        } else {
            null
        }
}
