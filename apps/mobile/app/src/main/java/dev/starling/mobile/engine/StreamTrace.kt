package dev.starling.mobile.engine

import org.json.JSONArray
import org.json.JSONObject

/**
 * Call ledger of one on-device live session (#226/#357), the Android
 * counterpart of the native server's `/stream?trace=1` instrumentation:
 * every engine call with its kind, audio span, wall time and result; every
 * capture chunk; every partial with the audio it reflected; and how Stop
 * produced the final. Debug-only ([StreamDebug]); production sessions run
 * without one.
 *
 * [toJson] writes the replay harness's event-log shape
 * (`benchmarks/experiments/stream_replay.py`: `t_start`, `sends`, `events`,
 * `commits`, `samples`, with the server's `trace` blocks on partials and the
 * final), so the host computes first text, partial age, backlog, work per
 * recorded second and stop-to-final with the server's own metric code.
 *
 * Times are seconds on the session clock. Thread-safe: the capture thread
 * reports audio, the worker everything else.
 */
class StreamTrace(
    private val clock: () -> Double = { System.nanoTime() / 1e9 },
    private val onComplete: (StreamTrace) -> Unit = {},
) {
    /** One engine call. [result]: ok, failed, preempted (cancelled or discarded) or reused. */
    data class Call(
        val kind: ChunkStreamer.CallKind,
        val startSample: Long,
        val samples: Int,
        val t0: Double,
        val t1: Double,
        val result: String,
    )

    private val lock = Any()
    private var tStart = Double.NaN
    private val sends = ArrayList<Pair<Double, Long>>()
    private val partials = ArrayList<Triple<Double, String, Long>>()
    private val calls = ArrayList<Call>()
    private var coveredEnd = 0L
    private var preparedAt = Double.NaN
    private var prepareMs = Double.NaN
    private var stopAt = Double.NaN
    private var settledAt = Double.NaN
    private var finalText: String? = null
    private var fallback: String? = null
    private var stopUnfinalized = 0L
    private var stopCallsFrom = Int.MAX_VALUE
    private val marks = JSONObject()

    /** Capture thread: [total] samples delivered so far, the latest just now. */
    fun audio(total: Long) = synchronized(lock) {
        val now = clock()
        // Each chunk arrives no earlier than its last sample was captured,
        // so the earliest implied origin over all chunks is the capture
        // start (to within the least delivery delay); the first chunk alone
        // would shift every age by its own delay.
        val origin = now - total.toDouble() / ChunkStreamer.SAMPLE_RATE
        if (tStart.isNaN() || origin < tStart) tStart = origin
        sends += now to total
    }

    fun prepared(startedAt: Double) = synchronized(lock) {
        preparedAt = clock()
        prepareMs = (preparedAt - startedAt) * 1000.0
    }

    fun call(call: Call) = synchronized(lock) {
        calls += call
        if (call.result == RESULT_OK || call.result == RESULT_REUSED) {
            coveredEnd = maxOf(coveredEnd, call.startSample + call.samples)
        }
    }

    fun partial(text: String) = synchronized(lock) { partials += Triple(clock(), text, coveredEnd) }

    /** finish() was called (the user's Stop): stop-to-final is measured from here. */
    fun stopRequested() = synchronized(lock) {
        if (stopAt.isNaN()) stopAt = clock()
    }

    /**
     * The worker started finalizing with [unfinalized] samples past the last
     * committed window boundary: every call recorded from here on is stop
     * work (a preview still running at Stop is not, though Stop waited for it).
     */
    fun flushing(unfinalized: Long) = synchronized(lock) {
        stopUnfinalized = unfinalized
        stopCallsFrom = calls.size
    }

    /** The session settled with a final [text], or a batch fallback for [fallbackReason]. */
    fun settled(text: String?, fallbackReason: String?) = synchronized(lock) {
        if (!settledAt.isNaN()) return@synchronized
        settledAt = clock()
        finalText = text
        fallback = fallbackReason
    }

    /** A named value the caller adds to the JSON (cadence, model, warm/cold, thermal state). */
    fun mark(name: String, value: Any?) = synchronized(lock) { marks.put(name, value ?: JSONObject.NULL) }

    fun calls(): List<Call> = synchronized(lock) { calls.toList() }

    /** The session's worker ended: hands the finished trace to its owner. */
    fun complete() = onComplete(this)

    fun toJson(): JSONObject = synchronized(lock) {
        val samples = sends.lastOrNull()?.second ?: 0L
        val events = JSONArray()
        for ((t, text, covered) in partials) {
            events.put(
                JSONArray().put(t).put(
                    JSONObject().put("type", "partial").put("text", text)
                        .put("trace", JSONObject().put("covered_s", seconds(covered))),
                ),
            )
        }
        if (!settledAt.isNaN()) {
            val event = if (finalText != null) {
                JSONObject().put("type", "final").put("text", finalText).put("duration_s", seconds(samples))
            } else {
                JSONObject().put("type", "error").put("message", "fallback: $fallback")
            }
            event.put("trace", finalTrace())
            events.put(JSONArray().put(settledAt).put(event))
        }
        JSONObject()
            .put("version", 1)
            .put("t_start", if (tStart.isNaN()) JSONObject.NULL else tStart)
            .put("sends", JSONArray().apply { sends.forEach { (t, n) -> put(JSONArray().put(t).put(n)) } })
            .put("events", events)
            .put("commits", JSONArray().apply { if (!stopAt.isNaN()) put(stopAt) })
            .put("busy_commit_retries", 0)
            .put("samples", samples)
            .put("prepare_ms", if (prepareMs.isNaN()) JSONObject.NULL else prepareMs)
            .put("live_at", if (preparedAt.isNaN()) JSONObject.NULL else preparedAt)
            .put("fallback", fallback ?: JSONObject.NULL)
            .put("marks", JSONObject(marks.toString()))
    }

    private fun finalTrace(): JSONObject {
        val byKind = JSONObject()
        ChunkStreamer.CallKind.entries.forEach { kind -> byKind.put(kind.label, totals(calls.filter { it.kind == kind })) }
        val stopCalls = if (stopCallsFrom <= calls.size) calls.subList(stopCallsFrom, calls.size) else emptyList()
        // Named like the server's stop paths (stream_session.cpp `final_path_`).
        val stopPath = when {
            finalText == null -> null
            stopCalls.any { it.result == RESULT_OK } -> PATH_TAIL
            stopCalls.any { it.result == RESULT_REUSED } -> PATH_REUSED
            else -> PATH_COMMITTED
        }
        return JSONObject()
            .put("covered_s", seconds(coveredEnd))
            .put("totals", totals(calls))
            .put("by_kind", byKind)
            .put(
                "stop",
                JSONObject()
                    .put("path", stopPath ?: JSONObject.NULL)
                    .put("unfinalized_s", seconds(stopUnfinalized))
                    .put("totals", totals(stopCalls)),
            )
            .put(
                "calls",
                JSONArray().apply {
                    calls.forEach { c ->
                        put(
                            JSONObject().put("kind", c.kind.label)
                                .put("start_s", seconds(c.startSample))
                                .put("end_s", seconds(c.startSample + c.samples))
                                .put("t0", c.t0).put("t1", c.t1).put("result", c.result),
                        )
                    }
                },
            )
    }

    /** The server's per-kind totals (cpp/serve/stream_session.cpp `totals_json`). */
    private fun totals(of: List<Call>): JSONObject {
        val ok = of.filter { it.result == RESULT_OK }
        val preempted = of.filter { it.result == RESULT_PREEMPTED }
        return JSONObject()
            .put("calls", of.size)
            .put("engine_calls", ok.size)
            .put("engine_audio_s", seconds(ok.sumOf { it.samples.toLong() }))
            .put("engine_ms", ok.sumOf { (it.t1 - it.t0) * 1000.0 })
            .put("reused", of.count { it.result == RESULT_REUSED })
            .put("busy", 0)
            .put("failed", of.count { it.result == RESULT_FAILED })
            .put("preempted", preempted.size)
            .put("preempted_ms", preempted.sumOf { (it.t1 - it.t0) * 1000.0 })
    }

    private fun seconds(samples: Long): Double = samples.toDouble() / ChunkStreamer.SAMPLE_RATE

    companion object {
        const val RESULT_OK = "ok"
        const val RESULT_FAILED = "failed"
        const val RESULT_PREEMPTED = "preempted"
        const val RESULT_REUSED = "reused"

        // Stop paths, as the server names them: the tail was transcribed;
        // an exact preview of the tail was reused; nothing was left to do.
        const val PATH_TAIL = "tail"
        const val PATH_REUSED = "reused"
        const val PATH_COMMITTED = "committed"
    }
}
