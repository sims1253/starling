# gpui vs Electron — Starling desktop

Measured 2026-09-16 on the dev machine (CachyOS, Wayland, 12 cores, 14 GB RAM,
Rust 1.98.1, gpui 0.2.2, Electron 44.3.0). Both apps print the same
`STARLING_DIAGNOSTICS` hook: process start → window ready-to-show, plus memory.

## Numbers

| Metric | gpui (release) | gpui (debug) | Electron 44 (dev main, blank renderer) |
|---|---|---|---|
| Ready-to-show, warm | **70 ms** | 238 ms | 313 ms |
| Ready-to-show, cold-ish | 705 ms | 835 ms | — |
| Memory at first frame | **51 MB** | 80 MB | — |
| Memory steady-state (idle) | **58 MB** (single process) | ~80 MB | **~460 MB working set across 4 processes** (Browser 186, GPU 101, Utility 77, Tab 86) |

The Electron figure is a *lower bound*: its renderer had no page loaded (the
`vp`/vite-plus toolchain wasn't installed in that checkout), so the real app —
React 19 + Vite dev server — would land noticeably higher, especially in the
Tab process. Its ready-to-show also excludes the vite dev server startup that
`pnpm run desktop` pays before the window opens.

Raw lines:

```
gpui release (warm):  STARLING_DIAGNOSTICS {"readyToShowMs":70,"rssBytes":51179520}
gpui release steady:  VmRSS 58340 kB
gpui debug (warm):    STARLING_DIAGNOSTICS {"readyToShowMs":238,"rssBytes":80244736}
electron:             {"mainProcess":{"rssBytes":193945600,...},"processes":[Browser 190136 KiB, GPU 103712 KiB, Utility 79196 KiB, Tab 88452 KiB],"readyToShowMs":313}
```

## Feel

- Startup: the release binary goes from click to interactive window in well
  under a second; there is no multi-process coordinator spin-up, so the window
  appears "all at once" rather than blank-then-fill.
- Scrolling/waveform: the 52-bar meter and history list render on the GPU via
  gpui's blade/Vulkan backend; at idle the app sits at ~58 MB with a flat CPU
  profile, versus Chromium's baseline of four processes before any app code
  runs.
- Recording + upload flows feel identical to the Electron app because the
  logic is a direct port (same endpoints, same WAV bytes on the wire, same
  manifests on disk).
- Typography matches (serif display face, mono eyebrows — resolved through
  gpui's font-fallback chain since fontconfig substitution is not applied to
  missing family names), but a few CSS niceties don't exist in gpui: no
  letter-spacing, no per-side border colors, no backdrop blur, no transform
  animations (hover rotate/scale, drawer slide-up are approximated or omitted).

## Functional parity

Same: recording flow (mono capture → PCM16 16 kHz WAV → local history →
upload), starling + OpenAI protocols, retry/failure states with retained audio,
transcript fidelity warnings, settings persistence, in-app + global
Cmd/Ctrl+Shift+Space, import of `.wav` files, clipboard copy, exports.

Different by design:

- IndexedDB → on-disk store (`~/.local/share/starling-gpui/sessions`), same
  manifest schema.
- `<audio controls>` → Play/Stop toggle (rodio).
- Browser downloads → files written straight into the downloads directory.
- `window.confirm` → inline two-step confirm for discarding unsaved audio.
- AnalyserNode → local FFT for the meter (visual mapping kept, not
  bit-identical).
- Single process, so `STARLING_DIAGNOSTICS` reports one RSS instead of
  per-process Chromium metrics.

## First real finding from the comparison

Recording the same phrase in both apps produced a clean transcript in Electron
and `牵牵牵…` garbage in the gpui port. Root cause, with receipts:

- The saved WAV from the gpui take had **RMS 0.56 with 11% of samples pinned
  at full scale** — destroyed audio; the model never stood a chance.
- WirePlumber stores this laptop's internal-mic route at **100% / 0 dB**
  (`~/.local/state/wireplumber/default-routes`), which saturates the codec's
  ADC on close speech.
- The port captures raw (cpal → downmix → encode is transparent, proven with
  an instrumented probe), while Chromium's getUserMedia pipeline attenuates /
  limits even with the app's constraints disabling AGC — so Electron "just
  works" on the same hot source.

Fixes in the port: an **attenuation-only auto gain** in the capture callback
(never amplifies; quiet input passes bit-exact; pulls hot-but-unclipped peaks
toward −2.9 dBFS, live-probed: peak 0.99 → 0.87 under the same conditions),
and a **"Recording clipped" banner** when a take still arrives with >2%
full-scale samples, telling the user to lower their mic level. Raw sources at
100% that clip at the ADC remain unfixable in software — lower the input gain
if you see the banner.

## Caveats

- gpui 0.2.2 is the crates.io release (Oct 2025), not Zed `main`.
- Global hotkey on Wayland is best-effort (portal-dependent); the in-app
  binding always works.
- IME support in the settings text fields is minimal (single-line, no marked
  text composition).
- The Electron numbers should be re-measured with the real renderer loaded
  (`pnpm install && pnpm run desktop` with `STARLING_DIAGNOSTICS=1`) for a
  fully apples-to-apples chart; the gap shown here only grows with the
  renderer's real content.
