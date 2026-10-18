package dev.starling.mobile.transcription

import android.content.Context
import dev.starling.mobile.audio.AudioChunkListener
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.data.Recording
import dev.starling.mobile.data.RecordingStatus
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.io.IOException

class TakeRecorderTest {
    private class Store : TakeRecorder.AudioStore {
        val rows = mutableMapOf<String, Recording>()
        val committed = mutableListOf<String>()
        val salvaged = mutableListOf<String>()
        var failCreate = false
        var failCommit = false
        private var next = 0
        override fun create(ephemeral: Boolean): Recording {
            if (failCreate) throw IOException("disk full")
            return Recording("take-${++next}", 0, "take-$next.wav", RecordingStatus.RECORDING, ephemeral = ephemeral)
                .also { rows[it.id] = it }
        }
        override fun partial(recording: Recording) = File("${recording.id}.partial.wav")
        override fun commit(recording: Recording, duration: Double): Recording {
            if (failCommit) throw IOException("commit failed")
            committed += recording.id
            return recording.copy(status = RecordingStatus.PENDING, durationSeconds = duration).also { rows[it.id] = it }
        }
        override fun salvage(recording: Recording, message: String) {
            salvaged += recording.id
            rows[recording.id] = recording.copy(status = RecordingStatus.FAILED, errorMessage = message)
        }
        override fun delete(recording: Recording) { rows.remove(recording.id) }
    }

    private class Session : StreamSession {
        var closed = false
        val chunks = mutableListOf<Byte>()
        override fun acceptsAudio() = !closed
        override fun onAudio(bytes: ByteArray, count: Int) { chunks += bytes.take(count) }
        override fun finish(): CommitOutcome = error("the transcription coordinator owns finalization")
        override fun close() { closed = true }
    }

    private class Transcriber(private val store: Store) : TakeRecorder.Transcriber {
        var live = true
        var queued = true
        val sessions = mutableListOf<Session>()
        val events = mutableListOf<(StreamEvent) -> Unit>()
        val requests = mutableListOf<Pair<String, Boolean>>()
        val completions = mutableListOf<() -> Unit>()
        override fun begin(audio: File, events: (StreamEvent) -> Unit): StreamSession? {
            this.events += events
            return if (live) Session().also { sessions += it } else null
        }
        override fun finish(session: StreamSession, recording: Recording, done: (Recording) -> Unit) =
            enqueue(recording, true, done)
        override fun transcribe(recording: Recording, done: (Recording) -> Unit) = enqueue(recording, false, done)
        private fun enqueue(recording: Recording, live: Boolean, done: (Recording) -> Unit): Boolean {
            assertTrue("inference must wait for durable audio", recording.id in store.committed)
            requests += recording.id to live
            if (queued) completions += {
                val result = recording.copy(status = RecordingStatus.TRANSCRIBED, rawTranscript = "verbatim ${recording.id}")
                store.rows[recording.id] = result
                done(result)
            }
            return queued
        }
    }

    private class Capture : TakeRecorder.Capture {
        var startError: String? = null
        var chunks: AudioChunkListener? = null
        val ended = mutableListOf<() -> Unit>()
        val lowDisk = mutableListOf<(Long) -> Unit>()
        val stops = mutableListOf<(CaptureResult) -> Unit>()
        override fun start(
            attributionContext: Context?, output: File, chunks: AudioChunkListener?, ended: () -> Unit,
            diskWarned: Boolean, diskLow: (Long) -> Unit,
        ): String? {
            this.chunks = chunks
            this.ended += ended
            lowDisk += diskLow
            return startError
        }
        override fun stop(done: (CaptureResult) -> Unit) { stops += done }
    }

    private class Fixture {
        val store = Store()
        val audio = Capture()
        val transcriber = Transcriber(store)
        val recorder = TakeRecorder(store, transcriber, audio)
        fun start(ephemeral: Boolean = false) =
            (recorder.start(ephemeral = ephemeral) as TakeRecorder.Start.Started).take
    }

    @Test
    fun stopWaitsForCaptureCommitsThenHandsOffTheStreamOnce() {
        val f = Fixture()
        val take = f.start()
        val notices = mutableListOf<String>()
        f.audio.chunks!!.onChunk(byteArrayOf(1, 2, 3), 2)
        assertEquals(listOf<Byte>(1, 2), f.transcriber.sessions.single().chunks)
        take.stop(
            onCaptured = { notices += "captured"; assertTrue(f.store.committed.isEmpty()) },
            onStopped = { notices += "saved"; assertTrue(it is TakeRecorder.Stopped.Saved) },
            onTranscribed = { notices += it.rawTranscript!! },
        )
        take.stop()
        assertFalse(take.capturing)
        assertTrue(f.transcriber.requests.isEmpty())
        assertEquals(1, f.audio.stops.size)
        f.audio.stops.single()(CaptureResult.Completed(2.0))
        f.audio.stops.single()(CaptureResult.Completed(2.0))
        assertEquals(listOf(take.recording.id to true), f.transcriber.requests)
        assertEquals(listOf("captured", "saved"), notices)
        f.transcriber.completions.single()()
        assertEquals(listOf("captured", "saved", "verbatim ${take.recording.id}"), notices)
    }

    @Test
    fun missingLiveModelStillCapturesAndTranscribesTheSavedAudio() {
        val f = Fixture()
        f.transcriber.live = false
        val take = f.start()
        assertFalse(take.live)
        assertNull(f.audio.chunks)
        take.stop()
        f.audio.stops.single()(CaptureResult.Completed(1.0, stoppedForLowDisk = true))
        assertEquals(listOf(take.recording.id to false), f.transcriber.requests)
    }

    @Test
    fun commitFailureSalvagesOrdinaryAudioAndNeverStartsInference() {
        val f = Fixture()
        val take = f.start()
        f.store.failCommit = true
        var result: TakeRecorder.Stopped? = null
        take.stop(onStopped = { result = it })
        f.audio.stops.single()(CaptureResult.Completed(1.0))
        assertEquals(TakeRecorder.Stopped.Failed(TakeRecorder.Failure.FINALIZATION), result)
        assertEquals(listOf(take.recording.id), f.store.salvaged)
        assertTrue(f.transcriber.sessions.single().closed)
        assertTrue(f.transcriber.requests.isEmpty())
    }

    @Test
    fun cancellationKeepsOrdinaryAudioWithoutTranscribing() {
        val f = Fixture()
        val take = f.start()
        take.stop(transcribe = false)
        assertTrue(f.transcriber.sessions.single().closed)
        f.audio.stops.single()(CaptureResult.Completed(3.0))
        assertEquals(3.0, f.store.rows.getValue(take.recording.id).durationSeconds, 0.0)
        assertTrue(f.transcriber.requests.isEmpty())
    }

    @Test
    fun privateTakesAreDeletedAfterCancelCaptureFailureOrCommitFailure() {
        for (result in listOf(CaptureResult.Completed(1.0), CaptureResult.Failed("mic failed"), CaptureResult.AlreadyStopped)) {
            val f = Fixture()
            val take = f.start(ephemeral = true)
            take.stop(transcribe = false)
            f.audio.stops.single()(result)
            assertTrue(f.store.rows.isEmpty())
        }
        val f = Fixture()
        val take = f.start(ephemeral = true)
        f.store.failCommit = true
        take.stop()
        f.audio.stops.single()(CaptureResult.Completed(1.0))
        assertTrue(f.store.rows.isEmpty())
        assertTrue(f.transcriber.requests.isEmpty())
    }

    @Test
    fun privateTranscriptionIsDeletedEvenAfterAnotherTakeStarts() {
        val f = Fixture()
        val privateTake = f.start(ephemeral = true)
        var delivered: String? = null
        privateTake.stop(onTranscribed = { delivered = it.rawTranscript })
        f.audio.stops.single()(CaptureResult.Completed(1.0))
        val next = f.start()
        f.transcriber.completions.single()()
        assertEquals("verbatim ${privateTake.recording.id}", delivered)
        assertNull(f.store.rows[privateTake.recording.id])
        assertTrue(next.capturing)
        assertTrue(f.store.rows.containsKey(next.recording.id))
    }

    @Test
    fun callbacksFromStoppedTakesCannotAffectTheNextTake() {
        val f = Fixture()
        val seen = mutableListOf<StreamEvent>()
        var ended = 0
        var warned = 0
        val first = (f.recorder.start(onEvent = { seen += it }, onEnded = { ended++ }, onDiskLow = { warned++ })
            as TakeRecorder.Start.Started).take
        first.stop()
        f.audio.stops.single()(CaptureResult.Completed(1.0))
        val next = f.start()
        f.transcriber.events.first()(StreamEvent.Partial("late text"))
        f.audio.ended.first()()
        f.audio.lowDisk.first()(3)
        assertTrue(seen.isEmpty())
        assertEquals(0, ended)
        assertEquals(0, warned)
        assertTrue(next.capturing)
        f.transcriber.completions.single()()
        assertTrue(next.capturing)
        assertEquals(listOf(first.recording.id), f.store.committed)
    }

    @Test
    fun startupFailureClosesTheStreamAndHonorsPrivacy() {
        for (ephemeral in listOf(false, true)) {
            val f = Fixture()
            f.audio.startError = "microphone unavailable"
            assertEquals(TakeRecorder.Start.Failed("microphone unavailable", false), f.recorder.start(ephemeral = ephemeral))
            assertTrue(f.transcriber.sessions.single().closed)
            assertEquals(ephemeral, f.store.rows.isEmpty())
            assertEquals(if (ephemeral) 0 else 1, f.store.salvaged.size)
        }
        val f = Fixture()
        f.store.failCreate = true
        assertEquals(TakeRecorder.Start.Failed("disk full", true), f.recorder.start())
        assertTrue(f.transcriber.sessions.isEmpty())
        assertTrue(f.audio.ended.isEmpty())
    }

    @Test
    fun lowDiskAndCapRemainSuccessfulSavedTakes() {
        val f = Fixture()
        val take = f.start()
        val capture = CaptureResult.Completed(4.0, cappedAtLimit = true, stoppedForLowDisk = true)
        var stopped: TakeRecorder.Stopped? = null
        take.stop(onStopped = { stopped = it })
        f.audio.stops.single()(capture)
        assertEquals(TakeRecorder.Stopped.Saved(f.store.rows.getValue(take.recording.id), capture), stopped)
        assertEquals(1, f.transcriber.requests.size)
    }

    @Test
    fun captureFailureSalvagesButNeverTranscribesAndRejectedStreamsClose() {
        val f = Fixture()
        val take = f.start()
        take.stop()
        f.audio.stops.single()(CaptureResult.Failed("storage write failed"))
        assertEquals(RecordingStatus.FAILED, f.store.rows.getValue(take.recording.id).status)
        assertTrue(f.transcriber.requests.isEmpty())
        assertTrue(f.transcriber.sessions.single().closed)
        val next = f.start()
        f.transcriber.queued = false
        next.stop()
        f.audio.stops.last()(CaptureResult.Completed(1.0))
        assertTrue(f.transcriber.sessions.last().closed)
    }
}
