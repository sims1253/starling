# Superwhisper teardown — what we can borrow

Findings from downloading and inspecting Superwhisper's Linux and Windows
builds (their own public distribution channels), plus their public docs. The
goal: concrete, borrowable ideas for Starling's engine, apps, and product
shape. Evidence was extracted 2026-11-16; quotes below are from the binaries,
so this note stands alone if the scratch copies disappear.

Provenance and scope: we downloaded what Superwhisper publishes to everyone —
the Linux release archive from their public GitHub releases
(`superultrainc/superwhisper-linux-releases` v1.3.1, sha256 verified against
their manifest) and the Windows installer from their CDN (v1.6.11, NSIS).
This is behavior analysis for design reference, the same rule the monorepo
already applies to FluidVoice, Handy and Wispr Flow: their code and prompt
text is not copied into Starling. Where their exact wording is quoted here it
is evidence; we write Starling's own text from the observed behavior.

The Android app is Google-Play-only (package `com.superultra.superwhisper`);
no public APK exists — APKPure does not mirror it and APKMirror is
Cloudflare-gated. Its behavior is documented from Superwhisper's own docs
below. If we want dex-level analysis, the clean path is `adb` from a device
with the app installed, then jadx; offer noted at the end.

## 1. What Superwhisper is (verified shape)

| Platform | Build | Offline ASR | Cleanup LLM | Insertion |
| --- | --- | --- | --- | --- |
| Linux 1.3.1 | Native Rust daemon (`superwhisper`) + GPUI settings window + TUI fallback, systemd user service | ggml-native: Parakeet TDT 0.6b v3 GGUF ("Offline Lite") or Cohere Transcribe GGUF ("Offline") | llama.cpp subprocess, S1-mini GGUF (open weights) | Paste routes + AT-SPI verification; fcitx5 direct insertion on KDE |
| Windows 1.6.11 | Tauri app | Same `transcribe` engine GGUFs plus ONNX Runtime: Parakeet v3 ONNX, Nemotron Speech streaming ONNX, custom Whisper beam-search contrib op | llama.cpp GGUFs (user-selectable, incl. German Leo Mistral), Ollama, BYOK cloud | TTS/clipboard (not inspected in depth) |
| Android 0.6.3 | Play-only | none — cloud transcription only | cloud | Floating-bubble overlay + AccessibilityService insertion; history with playback/share |
| macOS 3.3.1 | (not inspected) | Whisper models + Parakeet/Cohere | S1 family + BYOK | Accessibility |

Both desktop platforms run the *same* core: capture → local ASR worker →
optional LLM cleanup worker → vocabulary correction → paste/insert. The
pipeline is process-separated: the daemon spawns `asr-worker` and
`cleanup-worker` as independent executables (`usage: asr-worker MODEL
cpu|cuda|vulkan`, `usage: cleanup-worker MODEL GPU_LAYERS`), speaking JSON on
stdio — the same shape as their `desktop_portal.py` worker protocol ("one
JSON request per stdin line, one reply per stdout line").

## 2. Engine findings (borrowable for `cpp/` and `docs/models.md`)

### 2.1 One ggml engine, many ASR architectures, capability metadata

The Linux `asr-worker` is a single ggml-based binary implementing many
architecture families natively (all `stt.*` GGUF keys we extracted):
`whisper`, `parakeet`, `canary`, `canary_qwen`, `qwen3_asr`, `voxtral`,
`voxtral_realtime`, `moonshine`, `moonshine_streaming`, `sensevoice`,
`funasr_nano`, `gigaam`, `granite`, `granite_nar`, `medasr`, `moss`,
`cohere` (Cohere Transcribe, encoder/decoder with cross-KV cache, language
probe, batched decode) plus VAD-gated longform chunking ("Whisper longform
chunker: speech detection failed").

The borrowable part is not the list — it is that every model carries a
machine-readable capability record. Their error messages point users at
`transcribe_capabilities.max_audio_ms` when a decode hits a model's context
cap: "decode reached the position cap (%d) before end-of-stream … See
transcribe_capabilities.max_audio_ms."

**Recommendation (bold):** put a `capabilities` block in Starling's GGUF
model catalog now — at minimum `max_audio_ms`, `streaming` (none/partial/final),
`languages`, `itl` (intra-text language control), `pnc` (punctuation-and-casing
control; their strings show `runtime ITN/PNC control` as reserved flags), and
`vocabulary_biasing: bool`. Starling's fidelity rule "do not claim vocabulary
biasing unless the chosen engine implements it" then becomes a data check,
not a review check. This is directly compatible with `quants/` artifact
records.

### 2.2 Baseline binaries shipped in production releases

Their release ships `asr-worker-baseline` and `cleanup-worker-baseline`
beside the optimized workers. That is Starling's AUTORESEARCH alternating-A/B
discipline built into distribution: every installed user has the exact
baseline for the shipped build. When Starling bundles a server with the
desktop app (the first "Next product milestone"), shipping the previous
engine build alongside as a named fallback gives us instant A/B on user
machines and an automatic rollback path.

### 2.3 Dual-backend strategy

Linux = ggml everywhere; Windows/iOS = ONNX Runtime (Parakeet v3 ONNX,
Nemotron Speech streaming ONNX, and a custom `WhisperBeamSearch_Microsoft_ver1`
contrib op registered into ONNX Runtime for Whisper beam search). We should
not copy the dual strategy — Starling's bet is one native ggml engine — but
their Nemotron streaming choice validates "streaming ASR as a separate model
family with its own capabilities", which maps onto Starling's `WS /stream`
work.

### 2.4 Model download hygiene (borrowable verbatim as requirements)

From the updater strings: per-model download states (`installed`, `missing`,
`downloading`, `paused`), `verify`/`repair` actions ("Optimized model is
damaged; download to repair"), size-capped resumable curl
(`--continue-at -`, `--max-filesize`, `--speed-time 30 --speed-limit`,
`--proto=https --proto-redir`), checksum verification with discard-on-failure
("Checksum mismatch … the unverified file was discarded"), and disk-space
pre-check. Starling's "bundle and supervise a local server, with model
installation and verified artifacts" milestone should adopt this exact
requirement set; their failure messages are a good UX bar.

## 3. The cleanup pipeline (borrowable; prompt wording is theirs)

S1-mini (0.6B, 484 MB, open weights on HF — Starling already converts and
quantizes it, see `docs/quantization-s1-mini.md`) runs in a llama.cpp
subprocess with per-model-family jinja chat templates (they ship qwen3,
llama3, llama-3.2, phi-2, ministral, mistral, deepseek-r1, gpt-oss). The
qwen3 template pre-fills an empty `<think>\n\n</think>` block to skip CoT
tokens — a neat latency trick for reasoning-family models.

Their system prompt (extracted; quoted here as evidence, not for copying)
is worth studying as a behavior spec:

- Task framed as minimal edit: "Clean up the user's dictated message using
  the smallest possible changes."
- An explicit *may* list (punctuation, caps, spelling, homophones; number and
  date formatting; removing "um/uh", accidental repeats, explicit
  self-corrections; paragraph breaks; spoken emoji names → emoji).
- An explicit *never* list: never rewrite phrasing; keep tone, uncertainty,
  slang, profanity ("Never soften or remove profanity"); keep every clause
  including title-like openings; never delete to tidy.
- Disambiguation rules for the hard case: "smiley face" as emoji vs "the
  emoji picker" as ordinary words, with a fixed emoji name table.
- ~20 few-shot pairs covering self-correction ("Tuesday I mean Thursday"),
  spoken numbers/dates, homophones ("your welcome"), preserved rambling,
  preserved profanity, code text ("python script parse URL"), emoji vs noun.
- A 5-level tone system (casual → formal) defined *only* over casing,
  punctuation and word-forms (contractions, "gonna"→"going to"), each with
  one fixed example pair, applied by a separate instruction: tone "controls
  casing, punctuation, and word forms only, and never changes meaning,
  structure, or which words appear."
- A language guard: non-Latin-script messages keep their language and its
  own conventions, register preserved as spoken.
- Output contract: "Return only the cleaned message."

Their tone parameter reaches the worker as a `tone_control` field (strings:
`tone_control`, `invalid tone`).

**Recommendation:** Starling's milestone "reviewable cleanup proposals" gets
a concrete spec from this: separate the *formatting* contract (tone:
casing/punctuation/word-forms only) from the *content* contract (fillers,
self-corrections), emit cleanup as a diff proposal against the raw transcript,
and define the proposal's scope with the same three-way split. We keep
Starling's rule that nothing is silently inserted; their default
(auto-paste the LLM output) is exactly what Starling's fidelity rules
forbid. Their own worker embodies our fallback: on any failure — input too
long, model load failure, abnormal stop — it reports and the raw
transcription is retained verbatim.

## 4. Vocabulary: two layers (the second is the interesting one)

Layer 1 — text replacement (Starling has hints already): terms,
replacements ("super whisper" → "Superwhisper"), misheard-forms mapping with
pronunciations (`vocab pronounce "Tauri" "Towery, Tory app"`), a 1,000-term
limit, CSV import/export, preview, per-rule learning of user corrections
(`vocab auto on`).

Layer 2 — **acoustic vocabulary biasing** ("supervocab"), implemented, not
claimed: a distilled wav2vec2-XLSR-espeak acoustic model (shipped as both
ONNX and GGUF), a G2P (grapheme-to-phoneme) encoder/decoder pair, a
`phoneme_cost.u8` table, a common-words list, and a compressed `vocab.bin`
that is also uploaded to their cloud path (`enable_audio_vocab` on the S1
request). It runs on the live stream too ("[Supervocab][Audio][stream] mic
broadcast lagged by …"). Misheard forms expand phonetically at runtime
(the binary carries the generated variant set for "Omarchy": Amachi, Homachi,
Imochi, Omochi, O Machi, Oh Marchie…).

**Recommendation:** this is the concrete design behind Starling milestone
"capability-backed vocabulary biasing". A phoneme-cost scorer over
hypothesis alternatives (or constrained decoding where the engine supports
it), driven by the user's term list + G2P expansions, is implementable in
Starling's ggml engine without a second ASR pass: the wav2vec2 model is tiny
and its job is rescoring, not transcribing. The "vocab suggest from local
sources" feature (agent sessions, notes, git repos, apps) is a product idea
worth stealing later.

## 5. Insertion machinery (Linux) — the deepest craft in the build

Their insertion stack, recovered from scripts and strings:

- A route ladder with per-app knowledge: `portal`, `wtype`, `ydotool`, and
  combinations (`portal+wtype`, `wtype+ydotool`, …), overridable via
  `SW_PASTE_FORCE_ROUTE`; per-toolkit chords (Ctrl+V, Shift+Insert,
  Ctrl+Shift+V), per-browser handling (chromium/gecko/firefox family strings),
  editors, terminals; GNOME Shell needs the portal because background clients
  get no clipboard access.
- Paste *verification*: capture the focused editable field before (bounded
  AT-SPI read: only the focused ENTRY/TEXT node, 4096 chars before caret,
  200 ms deadline, one-shot bounded process), then after the paste measure
  `chars_before/chars_after`, `field_growth`, `paste_verdict` — with
  messages like "You switched windows, so it was not pasted." and "Clipboard
  changed since the paste; not restoring it". A clipboard-diff check decides
  whether restoring the old clipboard is safe.
- Context awareness for cleanup: `insertion-context.py` returns the focused
  field's character before/after the caret, whether the caret follows a
  newline — feeding the LLM whitespace-correct insertions.
- KDE's AT-SPI quirk is handled explicitly: menu nodes are skipped because
  "KWrite's menus are 560 of its 668 nodes", and stale FOCUSED states on Qt
  cells are ignored.
- GNOME: a Shell extension (overlay pill + push-to-talk pre-48), GlobalShortcuts
  portal with activations delivered to the daemon's control socket, custom
  keybindings fallback; Hyprland: direct `hyprctl` checks (session locked →
  "Desktop locked; text retained"); KDE: fcitx5 direct insertion so terminals
  work without touching the clipboard.
- Lock-screen and focus-loss handling: text is retained, never pasted blind.

**Recommendation:** Starling's milestone "platform-specific insertion,
overlays" should adopt this architecture wholesale as requirements: (1) a
route ladder with capability probing and per-app overrides, (2) a pre/post
field-capture verification step with typed verdicts, (3) bounded one-shot
helper processes for every accessibility call so a stuck AT-SPI peer cannot
hang the daemon, (4) explicit user-facing verdict messages. The one-shot
bounded-process pattern is the single most transferable implementation idea
here — it converts "accessibility APIs hang" from a daemon bug into a
contained timeout.

## 6. Audio capture craft

- Microphone wake: `prepareOnRecordingStart`, `keepMicrophoneReady` (keeps
  the input awake; replaces WirePlumber's suspend hook with a config drop-in
  so *other* devices still sleep), a prewarm capture node
  (`node.name=superwhisper-prewarm`) so first words aren't clipped.
- Playback courtesy during dictation: `defaultPlaybackBehavior = keepPlaying |
  pause | duck | mute | muffle`, implemented with a PipeWire link-group
  (`node.link-group=superwhisper-muffle`) + a filter, with a volume journal
  and `playback-restore` on service stop so volumes always return.
- Silence removal (always on in their Linux 1.3.1), sound effects through
  their own link-group, `pw-record` with explicit 16 kHz mono properties.
- Non-WAV file transcription decoded via ffmpeg (`-ac 1 -ar 16000 pcm_s16le`);
  cloud uploads are Opus-in-OGG (16 kHz mono) — for Starling's mobile/remote
  path, uploading Opus instead of WAV is an easy bandwidth win.

The muffle/duck set is a small feature with outsized UX value for Starling's
desktop app; the link-group implementation is clean PipeWire usage, and
`muffle` is even a separate `libsuperwhisper_muffle.so`.

## 7. Distribution, update, and integration surface

- Install: staged install → atomic `current` symlink swap → previous version
  kept for automatic rollback if startup fails; minisign signatures plus
  sha256 manifests; beta channel; systemd user units (`superwhisper.service`,
  an update timer with `RandomizedDelaySec`, `Conflicts=superwhisper-dev.service`
  so dev installs can't collide); per-distro dependency installation with
  Vulkan ICD selection by PCI vendor; PATH integration across bash/zsh/fish +
  environment.d with marker comments so uninstall removes exactly its lines.
- They publish an NixOS flake with a Home Manager module.
- **Agent-facing surface** (directly relevant to `docs/mcp-dictation.md`):
  a full CLI (`status`, `models select/download/verify`, `vocab …`,
  `history …`, `record/start/stop/cancel`, `transcribe file.wav --insert`,
  `doctor`, `update`) that prints JSON, plus a history-search/export/stats
  CLI, an MCP server, per-agent plugins (Claude Code, OpenCode, Pi, Codex,
  Grok, DeepSeek) that notify on agent finish and let you speak the next
  instruction, and — the clever bit — the installer symlinks a curated
  `SKILL.md` into `~/.agents/skills`, `~/.claude/skills`, `~/.codex/skills`,
  and `~/.pi/agent/skills` so every coding agent can configure the app
  safely ("Change settings through the CLI, never by editing files").
  **Recommendation:** when Starling's desktop app gains a daemon + CLI, ship
  the same four-piece surface: JSON CLI, MCP server, agent skill file,
  finish-notification plugin. It is cheap and it is exactly Starling's
  audience.

## 8. Modes (product shape)

Modes = named prompt presets with per-mode model choice (voice model +
language model chosen independently), per-mode language, optional
"activate when using app X" bindings, and Custom modes with three context
checkboxes (Application, Copied text, Selected text) plus an examples editor
(spoken input → expected output pairs). Their docs push XML-tagged prompts
and 2–3 few-shot examples for custom modes.

Starling's glossary already has the right primitive (staged draft, explicit
delivery). **Recommendation:** adopt the *editor* shape — mode = {name,
voice model, cleanup instructions, examples, context toggles, app bindings} —
as the UI for Starling's cleanup-proposal feature, with Starling's review
step kept. Modes/vocabulary stored per-device with planned sync matches
Starling's local-first posture.

## 9. Android (from their docs; no APK available)

Floating-bubble overlay (not an IME keyboard) + AccessibilityService
insertion; cloud transcription only, no on-device models yet; history with
audio playback and share; a settings screen that lists every permission with
a jump-to-system-setting fix action; phones only (tablets unsupported);
modes/vocabulary per-device, sync planned.

Starling's Android app is an IME + on-device engine. The borrowable bits:
the permissions-screen pattern (each permission's purpose + direct fix
link), history playback UI, and — for the keyboard-vs-bubble debate — their
existence proves the bubble+accessibility route ships fine on Android 13+;
Starling's explicit-Insert IME remains the more conservative choice, but a
bubble overlay is a plausible second surface for Starling without any IME
constraints.

## 10. What Starling should not borrow

- Silent auto-paste of LLM-rewritten text as the default path — violates
  Starling's fidelity rules (raw recognition is kept; cleanup is a reviewed
  proposal).
- Their cloud licensing/device-binding scheme (Starling is self-hosted;
  its "remote access belongs behind an authenticated HTTPS proxy" rule
  already covers the need).
- Screen-content capture ("Super Mode" reads the active app) without an
  explicit per-dictation opt-in — Starling's private-field concept should
  extend to *all* captured context.
- APKPure/mirror APK distribution for testing.

## 11. Suggested follow-ups

1. **Engine:** add the capability record (`max_audio_ms`, streaming class,
   language list, ITN/PNC/vocab-biasing flags) to `docs/models.md` +
   `quants/` artifacts; make fidelity rule 5 a check against it.
2. **Cleanup:** spec Starling's cleanup proposal as {formatting contract,
   content contract, scope rules} modeled on §3, with diff-based review in
   the desktop app; keep raw text authoritative.
3. **Insertion:** turn §5 into the requirements list for the desktop
   insertion milestone (route ladder, field-capture verdicts, bounded
   one-shot helpers).
4. **Audio:** adopt playback duck/mute/muffle + mic prewarm/keep-ready in
   `apps/desktop-gpui`; evaluate Opus-in-OGG for the Android/remote upload
   path.
5. **Distribution:** when bundling the server, copy their model-download
   requirement set (§2.4) and their staged-install/rollback pattern (§7).
6. **Android dex teardown (optional):** if wanted, install the app from Play
   on a device and `adb pull` the APK (or provide it), then jadx; expected
   value is moderate (cloud-only app), so schedule it behind the desktop
   follow-ups.
