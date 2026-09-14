# TypeScript workspace

The desktop renderer, Electron main process, preload, development launcher, and
shared dictation package use TypeScript 7.0.2. Effect is pinned to
`4.0.0-rc.115`: this is the v4 release candidate, not the v3 stable release.
Vite+ builds the React renderer; esbuild bundles the Electron entry points.

```text
React UI
  → typed preload bridge
    → Effect request + Schema decoding
      → Starling or OpenAI-compatible server

Browser preview
  → shared Effect client
    → development proxy → server
```

Decode external data with Effect Schema before using it as a domain value.
Keep asynchronous failures in the Effect error channel, pass interruption to
network requests, and release resources when an operation ends. Promise methods
remain available at React, Electron, and browser API boundaries. Pure audio and
text transformations can stay ordinary TypeScript functions.

The renderer displays the returned text without cleanup. Changing the compiler,
error handling, or schemas must not change transcripts or discard saved audio.

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
application and shared TypeScript source.

Keep `@oxlint/plugins` at the Oxlint version that Vite+ bundles (`vp --version`
lists it; currently 1.81.0). The complete generic
and Effect [anti-slop rules](https://github.com/dmmulroy/anti-slop) are vendored in
`tools/oxlint/anti-slop`. That directory includes the upstream revision and
licenses. Oxlint executes the rules; ESLint is not the lint runner. Review the
upstream diff before updating the vendored source.

Fix reported patterns at their source. A narrow host-API exception needs a
comment explaining the invariant; do not disable a rule group to pass a build.
Generated bundles and unchanged vendored code are excluded from lint and format
checks.

For changes to recording, persistence, HTTP, or IPC, build the native contract
fixture and run the application workflows:

```bash
cmake --preset native-cpu
cmake --build build/native-cpu --target starling-serve-contract-fixture
pnpm run test:apps
pnpm run build
xvfb-run -a pnpm run test:electron
```

The last command uses Xvfb on Linux without a display. These tests need Playwright
and a Chromium installation; install the matching browser with
`uv run --no-project --with playwright python -m playwright install chromium`.

## Electron build outputs

```text
apps/desktop/electron/*.ts
  → dist-electron/main.mjs
  → dist-electron/preload.cjs
```

The CommonJS preload is generated for Electron's sandbox. Maintain its TypeScript
source. Packaging includes the compiled renderer and shell, without TypeScript,
Effect source, lint tooling, or workspace dependencies as separate runtime files.
