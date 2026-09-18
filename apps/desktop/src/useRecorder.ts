import { useCallback, useEffect, useRef, useState } from "react";
import type { PcmAudio } from "@starling/dictation";
import { discardRecorderHandles, RecorderSession, type RecorderHandles } from "./recorderSession";

const BAR_COUNT = 52;

export interface RecorderStartOptions {
  /**
   * Preferred capture rate. Requesting 16 kHz lets chunks stream to the
   * server without resampling; the actual rate is reported per chunk because
   * the browser may not honor the request.
   */
  readonly sampleRate?: number;
  /** Receives each captured chunk (mono, at the actual context sample rate). */
  readonly onChunk?: (chunk: Float32Array, sampleRate: number) => void;
}

export function useRecorder() {
  const [recording, setRecording] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [levels, setLevels] = useState<number[]>(() => Array(BAR_COUNT).fill(0.06));
  const animationRef = useRef<number | undefined>(undefined);
  const timerRef = useRef<number | undefined>(undefined);
  const startingRef = useRef(false);
  const recordingRef = useRef(false);
  // True while stop() is still releasing the audio context (#143).
  const stoppingRef = useRef(false);
  const aliveRef = useRef(true);
  const startedAtRef = useRef(0);
  const chunksRef = useRef<Float32Array[]>([]);
  const [session] = useState(() => new RecorderSession());

  const stopVisualization = useCallback(() => {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);

    animationRef.current = undefined;

    if (timerRef.current) window.clearInterval(timerRef.current);

    timerRef.current = undefined;
  }, []);

  useEffect(() => {
    aliveRef.current = true;

    return () => {
      aliveRef.current = false;
      stopVisualization();
      void session.release();
    };
  }, [session, stopVisualization]);

  const drawLevels = useCallback(
    function drawFrame() {
      const analyser = session.current()?.analyser;

      if (!analyser) return;
      const bins = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(bins);
      const stride = Math.max(1, Math.floor(bins.length / BAR_COUNT));
      setLevels(
        Array.from({ length: BAR_COUNT }, (_, index) => {
          const value = bins[index * stride] ?? 0;

          return Math.max(0.045, Math.pow(value / 255, 1.45));
        }),
      );
      animationRef.current = requestAnimationFrame(drawFrame);
    },
    [session],
  );

  /**
   * Start a capture. Resolves false when the start was rejected — another
   * start is in flight, or a preceding stop is still releasing its audio
   * context — so the caller knows this call does not own the microphone;
   * resolves true once it does. Setup failures still reject.
   */
  const start = useCallback(
    async (options?: RecorderStartOptions): Promise<boolean> => {
      // Refs, not the `recording` snapshot: guards must hold between renders.
      if (recordingRef.current || startingRef.current || stoppingRef.current) {
        return false;
      }

      startingRef.current = true;
      let stream: MediaStream | undefined;
      let context: AudioContext | undefined;

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          },
        });
        context = new AudioContext(
          options?.sampleRate !== undefined ? { sampleRate: options.sampleRate } : undefined,
        );
        const source = context.createMediaStreamSource(stream);
        const analyser = context.createAnalyser();
        const processor = context.createScriptProcessor(4096, 1, 1);
        const silent = context.createGain();
        silent.gain.value = 0;
        const handles: RecorderHandles = { stream, context, source, analyser, processor };
        processor.onaudioprocess = (event) => {
          const chunk = new Float32Array(event.inputBuffer.getChannelData(0));
          chunksRef.current.push(chunk);
          options?.onChunk?.(chunk, handles.context.sampleRate);
        };

        source.connect(analyser);
        source.connect(processor);
        processor.connect(silent);
        silent.connect(context.destination);

        if (!aliveRef.current) {
          // Unmounted while permission was pending; nobody will stop this
          // capture later, so release the hardware immediately.
          await discardRecorderHandles(handles);

          return false;
        }

        session.install(handles);
        chunksRef.current = [];
        recordingRef.current = true;
        startedAtRef.current = performance.now();
        setElapsedMs(0);
        setRecording(true);
        timerRef.current = window.setInterval(
          () => setElapsedMs(performance.now() - startedAtRef.current),
          100,
        );
        drawLevels();

        return true;
      } catch (error) {
        stream?.getTracks().forEach((track) => track.stop());

        try {
          await context?.close();
        } catch {
          /* context was not fully initialized */
        }

        throw error;
      } finally {
        startingRef.current = false;
      }
    },
    [drawLevels, session],
  );

  const stop = useCallback(async (): Promise<
    { audio: PcmAudio; durationMs: number } | undefined
  > => {
    const handles = session.current();

    if (!recordingRef.current || !handles) return;

    recordingRef.current = false;
    stoppingRef.current = true;

    try {
      handles.processor.onaudioprocess = null;
      const durationMs = performance.now() - startedAtRef.current;
      const chunks = chunksRef.current.splice(0);
      const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const samples = new Float32Array(length);
      let offset = 0;

      for (const chunk of chunks) {
        samples.set(chunk, offset);
        offset += chunk.length;
      }

      const sampleRate = handles.context.sampleRate;
      setLevels(Array(BAR_COUNT).fill(0.06));
      stopVisualization();

      // release() detaches synchronously, so a start() racing this await keeps
      // its own handles and remains stoppable (#119). `recording` is published
      // and starts are re-accepted only after the release completes, so no new
      // capture is taken against a context that is still closing (#143).
      await session.release();

      setRecording(false);

      return { audio: { samples, sampleRate, channels: 1 }, durationMs };
    } finally {
      stoppingRef.current = false;
    }
  }, [session, stopVisualization]);

  return { recording, elapsedMs, levels, start, stop };
}
