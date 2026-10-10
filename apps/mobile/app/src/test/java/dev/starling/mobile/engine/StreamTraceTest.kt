package dev.starling.mobile.engine

import org.junit.Assert.assertEquals
import org.junit.Test

/** The on-device call ledger's JSON, in the replay harness's event-log shape. */
class StreamTraceTest {
    private var now = 0.0
    private val trace = StreamTrace(clock = { now })

    private fun call(kind: ChunkStreamer.CallKind, start: Long, samples: Int, result: String) =
        trace.call(StreamTrace.Call(kind, start, samples, now, now + 0.5, result))

    @Test
    fun captureStartIsTheEarliestOriginTheChunksImply() {
        // The first 0.25 s chunk is delivered 0.5 s late; the later ones on time.
        now = 10.75
        trace.audio(4_000)
        now = 11.0
        trace.audio(8_000)
        now = 11.25
        trace.audio(12_000)

        assertEquals(10.5, trace.toJson().getDouble("t_start"), 1e-9)
    }

    @Test
    fun stopWorkCountsOnlyCallsFromTheFlushOnAndNamesItsPath() {
        now = 1.0
        trace.audio(32_000)
        call(ChunkStreamer.CallKind.PREVIEW, 0, 16_000, StreamTrace.RESULT_OK)
        now = 2.0
        trace.partial("a")
        trace.stopRequested()
        // A preview still running at Stop, cancelled: not stop work.
        call(ChunkStreamer.CallKind.PREVIEW, 0, 24_000, StreamTrace.RESULT_PREEMPTED)
        trace.flushing(32_000)
        call(ChunkStreamer.CallKind.FLUSH_TAIL, 0, 32_000, StreamTrace.RESULT_OK)
        now = 3.0
        trace.settled("a b", null)

        val json = trace.toJson()
        val events = json.getJSONArray("events")
        assertEquals(1.0, events.getJSONArray(0).getJSONObject(1).getJSONObject("trace").getDouble("covered_s"), 1e-9)
        val final = events.getJSONArray(1).getJSONObject(1)
        assertEquals("final", final.getString("type"))
        assertEquals(2.0, final.getDouble("duration_s"), 1e-9)
        val stop = final.getJSONObject("trace").getJSONObject("stop")
        assertEquals(StreamTrace.PATH_TAIL, stop.getString("path"))
        assertEquals(2.0, stop.getJSONObject("totals").getDouble("engine_audio_s"), 1e-9)
        val totals = final.getJSONObject("trace").getJSONObject("totals")
        assertEquals(1, totals.getInt("preempted"))
        assertEquals(2, totals.getInt("engine_calls"))
        assertEquals(listOf(2.0), (0 until json.getJSONArray("commits").length()).map { json.getJSONArray("commits").getDouble(it) })
    }
}
