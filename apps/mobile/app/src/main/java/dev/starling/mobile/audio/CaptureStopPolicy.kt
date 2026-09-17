package dev.starling.mobile.audio

import java.util.concurrent.TimeUnit

/**
 * Pure decision core for stopping a capture: the timing budget of each
 * bounded wait and the [CaptureResult] a settled capture produces. Keeps
 * the numbers and the result mapping in one stateless, Android-free place
 * so they are unit-testable on the JVM without a microphone, in the shape
 * of the other guard classes (InputTargetGuard,
 * RecognitionSessionGuard). The thread choreography itself stays in
 * AudioCapture, inseparable from the real worker joins it performs.
 */
object CaptureStopPolicy {
    /**
     * How long stop() blocks its caller while the worker exits gracefully.
     * After AudioRecord.stop() a blocking read returns promptly and
     * finalizing the WAV is one small header write, so healthy devices
     * settle in milliseconds; 400 ms still covers a full capture-buffer
     * interval (the worker reads at most max(2 x minBuffer, 250 ms of
     * audio) per blocking call) plus finalization and scheduling jitter,
     * stays under the roughly half-second perceptibility bar, and is an
     * order of magnitude below the 5 s input-dispatch ANR threshold.
     */
    const val FAST_STOP_WAIT_MILLIS = 400L

    /**
     * Total grace across the caller-thread fast join and the background
     * join that follows escalation. The same 3 s the synchronous stop used
     * to spend inline before force-releasing.
     */
    val GRACEFUL_STOP_MILLIS = TimeUnit.SECONDS.toMillis(3)

    /** The remaining grace handed to the background forced-release phase. */
    val ESCALATED_GRACE_MILLIS = GRACEFUL_STOP_MILLIS - FAST_STOP_WAIT_MILLIS

    /**
     * Wait after force-releasing the recorder and interrupting the worker.
     * The same 1 s the synchronous stop used to spend inline.
     */
    val FORCED_STOP_WAIT_MILLIS = TimeUnit.SECONDS.toMillis(1)

    /**
     * Delivered when the worker survives even the forced release: the
     * capture stays STOPPING so a later start() cannot race the zombie or
     * its writer, and the partial WAV is kept for explicit cleanup.
     */
    val ZOMBIE_RESULT = CaptureResult.Failed(
        "The microphone did not stop cleanly; the partial recording was kept",
    )

    /**
     * The result of a capture whose worker has exited: a worker error wins
     * so the failure is surfaced; otherwise the finalized WAV is reported
     * with its exact duration, including the at-cap case that must be
     * committed like a normal completion.
     */
    fun settle(error: String?, bytes: Long, cappedAtLimit: Boolean): CaptureResult =
        if (error != null) {
            CaptureResult.Failed(error)
        } else {
            CaptureResult.Completed(
                durationSeconds = bytes.toDouble() / (WavWriter.SAMPLE_RATE * WavWriter.BYTES_PER_SAMPLE),
                cappedAtLimit = cappedAtLimit,
            )
        }
}
