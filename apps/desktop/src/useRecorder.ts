import { useCallback, useEffect, useRef, useState } from "react";
import type { PcmAudio } from "@starling/dictation";

const BAR_COUNT = 52;

export function useRecorder() {
  const [recording, setRecording] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [levels, setLevels] = useState<number[]>(() => Array(BAR_COUNT).fill(0.06));
  const streamRef = useRef<MediaStream | undefined>(undefined);
  const contextRef = useRef<AudioContext | undefined>(undefined);
  const processorRef = useRef<ScriptProcessorNode | undefined>(undefined);
  const sourceRef = useRef<MediaStreamAudioSourceNode | undefined>(undefined);
  const analyserRef = useRef<AnalyserNode | undefined>(undefined);
  const animationRef = useRef<number | undefined>(undefined);
  const timerRef = useRef<number | undefined>(undefined);
  const startingRef = useRef(false);
  const startedAtRef = useRef(0);
  const chunksRef = useRef<Float32Array[]>([]);

  const release = useCallback(async () => {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);

    if (timerRef.current) window.clearInterval(timerRef.current);

    if (processorRef.current) processorRef.current.onaudioprocess = null;

    try {
      processorRef.current?.disconnect();
    } catch {
      /* already disconnected */
    }

    try {
      sourceRef.current?.disconnect();
    } catch {
      /* already disconnected */
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());

    try {
      await contextRef.current?.close();
    } catch {
      /* already closed */
    }

    streamRef.current = undefined;
    contextRef.current = undefined;
    processorRef.current = undefined;
    sourceRef.current = undefined;
    analyserRef.current = undefined;
  }, []);

  useEffect(() => () => void release(), [release]);

  const drawLevels = useCallback(function drawFrame() {
    const analyser = analyserRef.current;

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
  }, []);

  const start = useCallback(async () => {
    if (recording || startingRef.current) return;
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
      context = new AudioContext();
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      const processor = context.createScriptProcessor(4096, 1, 1);
      const silent = context.createGain();
      silent.gain.value = 0;
      chunksRef.current = [];
      processor.onaudioprocess = (event) => {
        chunksRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };

      source.connect(analyser);
      source.connect(processor);
      processor.connect(silent);
      silent.connect(context.destination);
      streamRef.current = stream;
      contextRef.current = context;
      sourceRef.current = source;
      analyserRef.current = analyser;
      processorRef.current = processor;
      startedAtRef.current = performance.now();
      setElapsedMs(0);
      setRecording(true);
      timerRef.current = window.setInterval(
        () => setElapsedMs(performance.now() - startedAtRef.current),
        100,
      );
      drawLevels();
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
  }, [drawLevels, recording]);

  const stop = useCallback(async (): Promise<
    { audio: PcmAudio; durationMs: number } | undefined
  > => {
    const context = contextRef.current;

    if (!recording || !context) return;
    setRecording(false);

    if (processorRef.current) processorRef.current.onaudioprocess = null;
    const durationMs = performance.now() - startedAtRef.current;
    const chunks = chunksRef.current.splice(0);
    const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const samples = new Float32Array(length);
    let offset = 0;

    for (const chunk of chunks) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }

    const sampleRate = context.sampleRate;
    setLevels(Array(BAR_COUNT).fill(0.06));
    await release();

    return { audio: { samples, sampleRate, channels: 1 }, durationMs };
  }, [recording, release]);

  return { recording, elapsedMs, levels, start, stop };
}
