# TypeScript workspace

The shared dictation package and the `starling-serve` npm launcher use
TypeScript 7.0.2. Effect is pinned to `4.0.0-rc.115`: this is the v4 release
candidate, not the v3 stable release. The desktop app is native Rust
(`apps/desktop-gpui`); TypeScript owns the reference client library, the
fidelity rules, and the prebuilt-binary launcher.

```text
packages/dictation
  → Effect request + Schema decoding
    → Starling or OpenAI-compatible server

packages/serve
  → npm launcher for prebuilt starling-serve binaries
```

Decode external data with Effect Schema before using it as a domain value.
Keep asynchronous failures in the Effect error channel, pass interruption to
network requests, and release resources when an operation ends. Promise methods
remain available at Node and browser API boundaries. Pure audio and
text transformations can stay ordinary TypeScript functions.

The library returns raw text without cleanup. The optional transcript
refinement layer keeps that stance: it runs only on an explicit per-take action,
stores its result in a separate labeled `refined` field beside the raw transcript,
and never rewrites the raw transcript or its history. Changing the compiler,
error handling, or schemas must not change transcripts or discard saved audio.

Multi-turn threads (#117) keep the same stance. A take joins a thread only
through an explicit "Refine in thread" action, which stores an optional
`threadId` label on the session — additive like `refined`, so no session-schema
or database version bump. A threaded refine sends the thread's current text as
an assistant turn before the new dictated turn; each turn keeps its own
immutable raw transcript, and each turn's `refined` copy is the thread's state
at that turn. Threading is escapable at any time: "start new thread" only
clears a UI hint and never mutates sessions.

These stances were introduced by the Electron desktop app that this workspace
used to build; they remain the contract every client of `packages/dictation`
inherits, and the Rust desktop re-implements them at
`apps/desktop-gpui/crates/dictation`.

## Checks

Use Node.js 24.13.1 or later and pnpm. Run commands from the repository root:

```bash
pnpm install
pnpm run check
```

`check` runs Oxfmt, Oxlint, TypeScript, Vitest unit tests, and the production
build. Vite+ (`vp`) provides Oxfmt, Oxlint, and Vitest; their settings live in
the root `vite.config.ts`.
`pnpm run lint:fix` applies supported lint fixes. `pnpm run fmt` formats the
shared TypeScript source.

Keep `@oxlint/plugins` at the Oxlint version that Vite+ bundles (`vp --version`
lists it; currently 1.82.0). The complete generic
and Effect [anti-slop rules](https://github.com/dmmulroy/anti-slop) are vendored in
`tools/oxlint/anti-slop`. That directory includes the upstream revision and
licenses. Oxlint executes the rules; ESLint is not the lint runner. Review the
upstream diff before updating the vendored source.

Fix reported patterns at their source. A narrow host-API exception needs a
comment explaining the invariant; do not disable a rule group to pass a build.
Generated bundles and unchanged vendored code are excluded from lint and format
checks.

For changes to the client contract, build the native contract fixture and run
the native contract tests:

```bash
cmake --preset native-cpu
cmake --build build/native-cpu --target starling-serve-contract-fixture
pnpm run test:api
```

The Rust mirror of the client (`apps/desktop-gpui/crates/dictation`) runs the
identical PCM fixtures; keep both sides passing when touching audio handling.
