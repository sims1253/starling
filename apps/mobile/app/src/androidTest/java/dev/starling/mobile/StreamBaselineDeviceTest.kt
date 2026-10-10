package dev.starling.mobile

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import android.system.Os
import android.system.OsConstants
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.starling.mobile.audio.AudioCapture
import dev.starling.mobile.audio.CaptureResult
import dev.starling.mobile.audio.PcmSource
import dev.starling.mobile.audio.WavWriter
import dev.starling.mobile.data.Recording
import dev.starling.mobile.engine.StreamDebug
import dev.starling.mobile.engine.StreamTrace
import dev.starling.mobile.engine.WavPcm
import dev.starling.mobile.network.InferenceResult
import dev.starling.mobile.network.TranscriptionEngine
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Paced same-audio streaming workload for #226/#357 on a real phone, in the
 * debug app's own process: each workload take (stream_workload.py, pushed to
 * files/debug/stream-workload/) is played once at real-time pace through the
 * production path — AudioCapture writing the take's WAV, the on-device live
 * session fed from its chunks, Stop, commit, finishStreaming — and the
 * session's [StreamTrace] is written to
 * files/debug/stream-runs/<label>/r<repeat>-<take>.json together with
 * thermal state, app CPU time and a batch transcription of the same audio.
 * `benchmarks/experiments/android_stream_replay.py` pushes the workload,
 * runs this per repeat and reduces the traces with the server harness's
 * metric code.
 *
 * One instrumentation is one fresh process: without `warmup` its first take
 * starts with the model unloaded (cold); with it, the model is loaded and
 * warmed before the first take.
 *
 * Opt-in: `am instrument -w -e stream run -e takes short,medium,long
 * -e label L -e repeat 0 -e min 1.0 -e interval 1.0 [-e model <installed file>]
 * [-e warmup false] [-e batch false] -e class dev.starling.mobile.StreamBaselineDeviceTest ...`
 */
@RunWith(AndroidJUnit4::class)
class StreamBaselineDeviceTest {
    private val app = ApplicationProvider.getApplicationContext<StarlingApplication>()
    private val arguments = InstrumentationRegistry.getArguments()
    private val debugDir = File(app.filesDir, "debug")

    @Test
    fun pacedWorkload() {
        assumeTrue("set -e stream run to run", arguments.getString("stream") == "run")
        val takes = (arguments.getString("takes") ?: "short").split(",")
        val label = arguments.getString("label") ?: "adhoc"
        val repeat = arguments.getString("repeat")?.toInt() ?: 0
        val min = arguments.getString("min")?.toDouble() ?: 1.0
        val interval = arguments.getString("interval")?.toDouble() ?: 1.0
        val warmup = arguments.getString("warmup") != "false"
        val batch = arguments.getString("batch") != "false"
        val outDir = File(debugDir, "stream-runs/$label").apply { mkdirs() }

        val engine = app.onDeviceEngine
        arguments.getString("model")?.let { assertTrue("$it is not installed", engine.selectModel(it)) }
        val model = engine.activeModelName()
        assertNotNull("install an on-device model first", model)
        val config = app.backendSettings.load().copy(engine = TranscriptionEngine.ON_DEVICE)

        val traced = AtomicReference<StreamTrace?>()
        val traceDone = AtomicReference(CountDownLatch(1))
        StreamDebug.cadence = { min to interval }
        StreamDebug.traceSink = {
            { trace ->
                traced.set(trace)
                traceDone.get().countDown()
            }
        }
        val previousSource = PcmSource.debugSource
        // Recording keeps the CPU awake in real use (AudioRecord is active);
        // the paced stand-in sleeps between chunks, so hold it awake here.
        val wake = (app.getSystemService(Context.POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "starling:stream-baseline")
        wake.acquire(TimeUnit.HOURS.toMillis(2))
        try {
            var served = 0
            if (warmup) {
                val t0 = System.nanoTime()
                assertEquals(null, engine.preload { true })
                Log.i(TAG, "warmup: model loaded in ${(System.nanoTime() - t0) / 1_000_000} ms")
                served++
            } else {
                // A fresh process has nothing loaded; make sure of it.
                engine.releaseWhenIdle()
            }
            for (take in takes) {
                val wav = File(debugDir, "stream-workload/$take.wav")
                assertTrue("push $wav first", wav.isFile)
                val pcm = WavPcm.decodePcm16(wav)!!.pcm
                traced.set(null)
                traceDone.set(CountDownLatch(1))
                val state = if (served == 0) "cold" else "warm"
                Log.i(TAG, "[$label] repeat $repeat $take (${pcm.size / 32_000.0} s, $state, $min s / $interval s)")

                val thermalBefore = thermal()
                val cpuBefore = cpuSeconds()
                val wallBefore = System.nanoTime() / 1e9
                val run = runTake(pcm, config)
                val cpu = cpuSeconds() - cpuBefore
                val wall = System.nanoTime() / 1e9 - wallBefore
                assertTrue("no trace for $take", traceDone.get().await(30, TimeUnit.SECONDS))
                val trace = traced.get()!!
                served++

                trace.mark("take", take)
                trace.mark("repeat", repeat)
                trace.mark("label", label)
                trace.mark("state", state)
                trace.mark("model", model)
                trace.mark("app_git_sha", BuildConfig.GIT_SHA)
                trace.mark("stop_pressed", run.stopPressed)
                trace.mark("delivered", run.delivered)
                trace.mark("final_status", run.recording.status.name)
                trace.mark("final_provenance", run.recording.provenance?.name)
                trace.mark("thermal_before", thermalBefore)
                trace.mark("thermal_after", thermal())
                trace.mark("app_cpu_s", cpu)
                trace.mark("take_wall_s", wall)
                if (batch) {
                    val t0 = System.nanoTime()
                    val result = engine.transcribe(wav)
                    trace.mark("batch_ms", (System.nanoTime() - t0) / 1e6)
                    trace.mark("batch_text", (result as? InferenceResult.Success)?.rawTranscript)
                }
                val json = trace.toJson()
                File(outDir, "r$repeat-$take.json").writeText(json.toString())
                Log.i(TAG, "[$label] $take: ${run.recording.status} cpu=${"%.1f".format(cpu)} s; trace written")
                app.recordings.delete(run.recording.id)
            }
        } finally {
            PcmSource.debugSource = previousSource
            StreamDebug.cadence = null
            StreamDebug.traceSink = null
            wake.release()
        }
    }

    private class TakeRun(val recording: Recording, val stopPressed: Double, val delivered: Double)

    /** Records [pcm] once at real-time pace through the production capture and streaming path. */
    private fun runTake(pcm: ByteArray, config: dev.starling.mobile.network.BackendConfig): TakeRun {
        val recording = app.recordings.create()
        val partial = app.recordings.partialFile(recording)
        val session = app.transcription.beginStreaming(config, partial)
        assertNotNull("no on-device live session", session)
        session!!
        val source = PacedOnceSource(pcm)
        PcmSource.debugSource = { source }
        val capture = AudioCapture()
        // As the app's entry points do: the WAV first, then the live session.
        assertEquals(null, capture.start(app, partial, onChunk = { bytes, count -> session.onAudio(bytes, count) }))
        assertTrue("the take did not play out", source.exhausted.await(30, TimeUnit.MINUTES))
        val stopPressed = System.nanoTime() / 1e9

        val stopped = CountDownLatch(1)
        val captured = AtomicReference<CaptureResult>()
        capture.stop { captured.set(it); stopped.countDown() }
        assertTrue(stopped.await(30, TimeUnit.SECONDS))
        val completed = captured.get() as? CaptureResult.Completed ?: throw AssertionError("capture: ${captured.get()}")
        assertEquals(pcm.size / 32_000.0, completed.durationSeconds, 1e-6)
        val committed = app.recordings.commitAudio(recording, completed.durationSeconds)

        val settled = CountDownLatch(1)
        val result = AtomicReference<Recording>()
        var delivered = 0.0
        assertTrue(
            app.transcription.finishStreaming(session, committed.id, config) {
                delivered = System.nanoTime() / 1e9
                result.set(it)
                settled.countDown()
            },
        )
        assertTrue(settled.await(30, TimeUnit.MINUTES))
        return TakeRun(result.get(), stopPressed, delivered)
    }

    private fun thermal(): JSONObject {
        val power = app.getSystemService(Context.POWER_SERVICE) as PowerManager
        val battery = app.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return JSONObject()
            .put("status", power.currentThermalStatus)
            .put("headroom_10s", power.getThermalHeadroom(10).toDouble().takeIf { it.isFinite() } ?: JSONObject.NULL)
            .put("battery_temp_c", (battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1) / 10.0)
            .put("plugged", battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) ?: -1)
    }

    /** User + system CPU time of this process, every thread (capture, engine, GEMV workers). */
    private fun cpuSeconds(): Double {
        val fields = File("/proc/self/stat").readText().substringAfterLast(") ").split(" ")
        // utime and stime are fields 14 and 15; the split starts at field 3.
        val ticks = fields[11].toLong() + fields[12].toLong()
        return ticks.toDouble() / Os.sysconf(OsConstants._SC_CLK_TCK)
    }

    /**
     * Plays [pcm] once at real-time pace like a microphone, then blocks until
     * stopped: [exhausted] opens once the last chunk was delivered, the
     * moment a user would press Stop.
     */
    private class PacedOnceSource(private val pcm: ByteArray) : PcmSource {
        val exhausted = CountDownLatch(1)

        @Volatile
        private var stopped = false
        private var startNanos = 0L
        private var emitted = 0

        override fun start() {
            startNanos = System.nanoTime()
        }

        override fun read(buffer: ByteArray, count: Int): Int {
            val wanted = minOf(count - count % WavWriter.BYTES_PER_SAMPLE, pcm.size - emitted)
            if (wanted <= 0) {
                exhausted.countDown()
                while (!stopped) Thread.sleep(5)
                return 0
            }
            val due = startNanos + (emitted + wanted).toLong() * 1_000_000_000L / BYTES_PER_SECOND
            while (!stopped) {
                val wait = due - System.nanoTime()
                if (wait <= 0) break
                Thread.sleep(minOf(wait / 1_000_000 + 1, 50L))
            }
            if (stopped) return 0
            System.arraycopy(pcm, emitted, buffer, 0, wanted)
            emitted += wanted
            if (emitted == pcm.size) exhausted.countDown()
            return wanted
        }

        override fun stop() {
            stopped = true
        }

        override fun release() {
            stopped = true
        }
    }

    companion object {
        private const val TAG = "StarlingStream"
        private const val BYTES_PER_SECOND = WavWriter.SAMPLE_RATE * WavWriter.BYTES_PER_SAMPLE
    }
}
