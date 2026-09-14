# TypeScript workspace

The desktop renderer, Electron main process, preload, development launcher, and
shared dictation package use TypeScript 7.0.2. Effect is pinned to
`4.0.0-rc.115`: this is the v4 release candidate, not the v3 stable release.
Vite builds the React renderer; esbuild bundles the Electron entry points.

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

Use Node.js 22.12 or later. Run commands from the repository root:

```bash
npm ci
npm run check
```

`check` runs Oxlint, Oxfmt, TypeScript, unit tests, and the production build.
`npm run lint:fix` applies supported lint fixes. `npm run format` formats the
application and shared TypeScript source.

Oxlint and `@oxlint/plugins` are pinned together at 1.83.0. The complete generic
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
npm run test:apps
npm run build
xvfb-run -a npm run test:electron
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
