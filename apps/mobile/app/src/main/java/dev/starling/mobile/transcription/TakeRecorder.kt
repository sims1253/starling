package dev.starling.mobile.transcription

import android.content.Context
import dev.starling.mobile.audio.AudioCapture
import dev.starling.mobile.audio.AudioChunkListener
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording
import dev.starling.mobile.storage.RecordingStore
import java.io.File

/**
 * Owns capture, durable audio and live transcription together. Each caller
 * owns one recorder and keeps the returned [Take] for display and delivery.
 * All methods and callbacks run on the main thread, except PCM forwarding.
 * A stopped take can settle after another starts without changing its state.
 */
class TakeRecorder internal constructor(
    private val store: AudioStore,
    private val transcription: Transcriber,
    private val capture: Capture,
) {
    constructor(context: Context, store: RecordingStore, transcription: TranscriptionCoordinator) : this(
        object : AudioStore {
            override fun create(ephemeral: Boolean) = store.create(ephemeral = ephemeral)
            override fun partial(recording: Recording) = store.partialFile(recording)
            override fun commit(recording: Recording, duration: Double) = store.commitAudio(recording, duration)
            override fun salvage(recording: Recording, message: String) { store.salvageCapture(recording.id, message) }
            override fun delete(recording: Recording) { store.delete(recording.id) }
        },
        object : Transcriber {
            override fun begin(audio: File, events: (StreamEvent) -> Unit) = transcription.beginStreaming(audio, events)
            override fun finish(session: StreamSession, recording: Recording, done: (Recording) -> Unit) =
                transcription.finishStreaming(session, recording.id, done)
            override fun transcribe(recording: Recording, done: (Recording) -> Unit) =
                transcription.transcribe(recording.id, callback = done)
        },
        object : Capture {
            private val audio = AudioCapture()
            override fun start(
                attributionContext: Context?, output: File, chunks: AudioChunkListener?, ended: () -> Unit,
                diskWarned: Boolean, diskLow: (Long) -> Unit,
            ) = audio.start(attributionContext ?: context, output, chunks, ended, diskWarned, diskLow)
            override fun stop(done: (CaptureResult) -> Unit) = audio.stop(done)
        },
    )

    sealed interface Start {
        data class Started(val take: Take) : Start
        data class Failed(val message: String, val storage: Boolean) : Start
    }

    sealed interface Stopped {
        data class Saved(val recording: Recording, val capture: CaptureResult.Completed) : Stopped
        /** A cancelled private take whose audio has been deleted. */
        data object Discarded : Stopped
        data class Failed(val kind: Failure, val message: String? = null) : Stopped
    }

    enum class Failure { CAPTURE, FINALIZATION, TRANSCRIPTION, ALREADY_STOPPED }

    private var active: Take? = null

    fun start(
        ephemeral: Boolean = false,
        attributionContext: Context? = null,
        diskWarned: Boolean = false,
        onEvent: (StreamEvent) -> Unit = {},
        onEnded: () -> Unit = {},
        onDiskLow: (Long) -> Unit = {},
    ): Start {
        val recording = runCatching { store.create(ephemeral) }.getOrElse {
            return Start.Failed(it.message ?: "Unable to create the recording", storage = true)
        }
        val take = Take(recording, ephemeral)
        val session = transcription.begin(store.partial(recording)) { event ->
            if (active === take && take.capturing) onEvent(event)
        }
        take.session = session
        val error = capture.start(
            attributionContext, store.partial(recording),
            session?.let { AudioChunkListener(it::onAudio) },
            { if (active === take && take.capturing) onEnded() },
            diskWarned,
            { minutes -> if (active === take && take.capturing) onDiskLow(minutes) },
        )
        if (error != null) {
            session?.close()
            take.keepFailure(error)
            return Start.Failed(error, storage = false)
        }
        active = take
        return Start.Started(take)
    }

    inner class Take internal constructor(val recording: Recording, private val ephemeral: Boolean) {
        internal var session: StreamSession? = null
        val live: Boolean get() = session != null
        var capturing: Boolean = true
            private set
        private var captureSettled = false

        /**
         * Stops once. Ordinary cancelled takes retain their audio without
         * transcription; private takes are deleted when they settle.
         * [onStopped] runs after commit/salvage and transcription admission,
         * so history can show the attempt's current status immediately.
         * [onCaptured] releases platform holds as soon as the microphone and
         * WAV have settled, including a failed capture.
         */
        fun stop(
            transcribe: Boolean = true,
            onCaptured: () -> Unit = {},
            onStopped: (Stopped) -> Unit = {},
            onTranscribed: (Recording) -> Unit = {},
        ) {
            if (!capturing) return
            capturing = false
            if (active === this) active = null
            val stream = session
            if (!transcribe) stream?.close()
            capture.stop settle@{ result ->
                if (captureSettled) return@settle
                captureSettled = true
                onCaptured()
                when (result) {
                    is CaptureResult.Completed -> {
                        val saved = runCatching { store.commit(recording, result.durationSeconds) }.getOrElse {
                            stream?.close()
                            keepFailure(FINALIZATION_ERROR)
                            onStopped(Stopped.Failed(Failure.FINALIZATION))
                            return@settle
                        }
                        if (!transcribe && ephemeral) {
                            delete()
                            onStopped(Stopped.Discarded)
                            return@settle
                        }
                        if (transcribe) {
                            var rejected = false
                            val done: (Recording) -> Unit = { completed ->
                                if (!rejected) {
                                    if (ephemeral) delete()
                                    onTranscribed(completed)
                                }
                            }
                            val queued = if (stream != null) transcription.finish(stream, saved, done)
                            else transcription.transcribe(saved, done)
                            // A rejected stream was not transferred to a worker.
                            if (!queued) {
                                rejected = true
                                stream?.close()
                                if (ephemeral) delete()
                                onStopped(Stopped.Failed(Failure.TRANSCRIPTION))
                                return@settle
                            }
                        }
                        onStopped(Stopped.Saved(saved, result))
                    }
                    is CaptureResult.Failed -> {
                        stream?.close()
                        keepFailure(result.message)
                        onStopped(Stopped.Failed(Failure.CAPTURE, result.message))
                    }
                    CaptureResult.AlreadyStopped -> {
                        stream?.close()
                        if (ephemeral) delete()
                        onStopped(Stopped.Failed(Failure.ALREADY_STOPPED))
                    }
                }
            }
        }

        internal fun keepFailure(message: String) {
            runCatching { if (ephemeral) store.delete(recording) else store.salvage(recording, message) }
        }

        private fun delete() { runCatching { store.delete(recording) } }
    }

    internal interface AudioStore {
        fun create(ephemeral: Boolean): Recording
        fun partial(recording: Recording): File
        fun commit(recording: Recording, duration: Double): Recording
        fun salvage(recording: Recording, message: String)
        fun delete(recording: Recording)
    }

    /** Terminal callbacks are posted to main after admission returns; rejection may also post one. */
    internal interface Transcriber {
        fun begin(audio: File, events: (StreamEvent) -> Unit): StreamSession?
        fun finish(session: StreamSession, recording: Recording, done: (Recording) -> Unit): Boolean
        fun transcribe(recording: Recording, done: (Recording) -> Unit): Boolean
    }

    internal interface Capture {
        fun start(
            attributionContext: Context?, output: File, chunks: AudioChunkListener?, ended: () -> Unit,
            diskWarned: Boolean, diskLow: (Long) -> Unit,
        ): String?
        fun stop(done: (CaptureResult) -> Unit)
    }

    companion object {
        private const val FINALIZATION_ERROR = "Unable to finalize the private WAV recording"
    }
}
