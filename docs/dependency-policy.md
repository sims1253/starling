# Dependency policy

Starling builds from five ecosystems (uv, CMake with vendored sources and a
submodule, Gradle, Swift, Cargo) plus GitHub Actions tooling. This document
defines what counts as a dependency change, what evidence a change needs, how
lockfiles are regenerated, which CI job validates each surface, and how
socket-security warnings are triaged. It was written from an audit of branch
`program/wave-a` on 20 September 2026 (issue E16); every state claim below was
read or executed on that tree, not assumed.

## Inventory

| Surface | Manifest(s) | Lockfile / pin | State (audited 2026-09-20) | Drift gate |
| --- | --- | --- | --- | --- |
| Python research backend (deprecated) | `pyproject.toml` | `uv.lock` (149 packages) | In sync; `uv lock --check --offline` exits 0. Ranges intentionally floating (research only, off the production path) | `uv lock --check` in `apps.yml` `api-contracts` |
| Quant catalog | `quants/pyproject.toml` | `quants/uv.lock` | Stdlib only; `uv lock --check --offline --project quants` exits 0 | existing `uv run --project quants --locked` in `apps.yml` |
| SONAR harness | `benchmarks/sonar/pyproject.toml` | `benchmarks/sonar/uv.lock` | `psdn-sonar[ml]==0.1.2` exact with in-file rationale; `uv lock --check --offline --project benchmarks/sonar` exits 0 | existing `uv sync --locked --project benchmarks/sonar` in `test.yml` |
| Native build | root `CMakeLists.txt`, `backends/native`, `cpp/`, `apps/mobile/src/main/cpp` | none needed | No `FetchContent` / `ExternalProject` / `file(DOWNLOAD)` in any Starling-owned CMake file; configure-time downloads do not exist | builds compile only from the submodule and vendored files below |
| ggml engine | `.gitmodules` | submodule commit `e91ded11` (tag `v0.23.0`) + `third_party/ggml-patches` | Pinned by commit; patches applied by `scripts/apply_ggml_patches.sh` | checkout with `submodules: recursive`; patch script fails on mismatch |
| `third_party/dr_wav.h` | vendored single file | version stamp in header | `dr_wav - v0.14.6`, public domain / MIT-0 | full-file diff review on update |
| `third_party/httplib.h` | vendored single file | `CPPHTTPLIB_VERSION` | `0.53.0`, MIT (Yuji Hirose) | full-file diff review on update |
| GitHub Actions | `.github/workflows/*.yml` | commit-SHA pins | Every `uses:` is a 40-char SHA except `pullfrog/pullfrog@v0` (documented exception, below) | Dependabot, weekly, all workflows |
| Android | `apps/mobile/build.gradle.kts`, `apps/mobile/app/build.gradle.kts` | none — dependency locking not enabled | Direct deps exact (`androidx.core 1.16.0`, `okhttp 4.12.0`, `mockwebserver 4.12.0`, `junit 4.13.2`, `org.json 20260814`); AGP `9.4.0`; wrapper `gradle-9.7.1`; NDK `28.2.13676358`; CMake `3.31.1`. Transitive deps float at build time (follow-up) | Dependabot gradle weekly; CI build in `apps.yml` `android` |
| iOS | `apps/ios/Package.swift` | — | No external package declarations; local targets only | `swift test` in `apps.yml` `ios` |
| Rust / GPUI desktop | `apps/desktop-gpui/Cargo.toml` (workspace, 4 crates) | `apps/desktop-gpui/Cargo.lock` (committed) | `gpui 0.2.2` plus direct cargo deps, exact via the lockfile | `desktop-gpui-rust.yml` cargo lanes |

Notes verified during the audit:

- uv spot-checks: `torch 2.13.0+cu130` (pytorch-cu130 explicit index),
  `transformers 5.15.0`, `fastapi 0.141.1`, `pytest 9.1.1`, `accelerate 1.14.0`.

## What counts as a dependency change

A PR changes dependencies if it touches any of:

1. A manifest: `pyproject.toml` (root, `quants/`, `benchmarks/sonar/`),
   `apps/mobile/**/build.gradle.kts`, `apps/ios/Package.swift`, or — when the
   GPUI port lands — `Cargo.toml`.
2. A lockfile: `uv.lock` (any of the three), `Cargo.lock`.
3. The `third_party/ggml` submodule pointer or `third_party/ggml-patches/**`.
4. A vendored header in `third_party/` (`dr_wav.h`, `httplib.h`).
5. Any GitHub Actions `uses:` pin, or any CI step that installs tools at run
   time (`brew install`, `pip install`, `uv ... --with ...`).
6. Build-tool versions that gate the build: the Gradle wrapper distribution,
   AGP/NDK/CMake versions in `apps/mobile`, XcodeGen usage, `setup-uv` action
   pins.

These changes must be called out in the PR description (what and why), not
buried in an unrelated refactor.

## Evidence a change needs

- What: package name, old and new version, and whether it is direct or
  transitive. Transitive-only lockfile churn from a routine regeneration needs
  no per-package notes; a deliberate transitive pin (override or catalog
  entry) does.
- Why: link to release notes; for security-driven bumps, the advisory
  (GHSA/CVE) or the socket-security finding that motivates it.
- License: confirm the license is unchanged and compatible. A checksum or
  lockfile entry is provenance, never a redistribution grant; release
  artifacts additionally need the packaging checklist (issue #21 owns the
  vendor-code-license questions).
- Prerelease or build-tool exceptions: a one-line rationale plus a removal
  condition (see the exceptions register).
- Regeneration: the lockfile is regenerated in the same PR by the commands
  below — never hand-edited.
- Validation: the CI jobs from the inventory table must pass. Desktop-facing
  changes must exercise the `desktop-gpui-rust.yml` lanes (ubuntu tests plus
  the macOS/Windows package builds), not just unit tests.

## Lockfile regeneration per ecosystem

Run from the repository root; commit the lockfile together with the manifest
change:

```bash
# uv — three independent projects
uv lock                            # root research backend
uv lock --project quants
uv lock --project benchmarks/sonar
uv lock --check                    # and the --project variants must pass

# Vendored headers: replace the file from the upstream release, keep the
# version stamp / CPPHTTPLIB_VERSION macro accurate, review the full diff.
# ggml: move the submodule to the chosen commit, re-run
# scripts/apply_ggml_patches.sh, commit pointer + any patch edits together.
```

## Who validates: CI coverage

| Gate | Where | Notes |
| --- | --- | --- |
| Root uv drift | `apps.yml` `api-contracts` | `uv lock --check` after `setup-uv`; previously only covered by the workflow_dispatch-only `test.yml` `cpu-tests` job |
| quants lock | `apps.yml` `api-contracts` | pre-existing `uv run --project quants --locked` |
| sonar lock | `test.yml` `sonar-tests` | pre-existing `uv sync --locked --project benchmarks/sonar` |
| Actions pins | Dependabot | weekly, `github-actions` ecosystem, all workflows |
| Gradle bumps | Dependabot | weekly, `/apps/mobile` |
| Rust desktop (gpui) | `desktop-gpui-rust.yml` | cargo test on ubuntu, windows host check, packaged macOS DMG + Windows zip builds |

## Exceptions register

Each entry has a rationale and a removal condition; new exceptions must be
added here in the same PR that introduces them.

| Exception | Rationale | Removal condition |
| --- | --- | --- |
| `pullfrog/pullfrog@v0` moving tag (`pullfrog.yml`) | Vendor documents that a SHA pin *without* an updater freezes the action's `post:` cleanup step and later breaks every run; the tag is vendor-supported and the action runs with `id-token: write` | When pullfrog ships a pinned-SHA-plus-updater combination that its docs endorse |
| Floating `>=` ranges in the root `pyproject.toml` | Deprecated research backend, kept independent of the production native path; `uv.lock` still pins exact versions for any given checkout | When the research backend is retired or frozen |
| CI-resolved tools (`brew install xcodegen`, `uv --with openapi-spec-validator`) | Test-only tooling; failures surface immediately in the same job | Follow-up: pin versions once a lockfile-equivalent mechanism exists (see gaps) |

## socket-security triage

The socket-security bot comments on PRs that change direct dependencies
(verified on PR #193, where it tabulated the `gpui-port` cargo additions such
as `gpui@0.2.2` with per-dimension scores). Triage flow:

1. Open the diff-scan link in the bot comment. Work the **added and changed
   direct dependencies**; score changes on untouched transitive packages are
   informational.
2. For each package, check the dimensions the bot flags: supply chain
   (install scripts, typosquat heuristics), vulnerability, maintenance,
   license. A low score alone is not a blocker; an unexplained one is.
3. Blocking conditions: new postinstall/preinstall scripts on a package that
   did not have them, a license incompatible with the package's use, or a
   known-vulnerable version with a fixed release available.
4. Record the outcome in the PR: which packages were reviewed, what was
   checked, and any accepted risk with its condition. If the bot comment and
   the lockfile disagree about what changed, trust the lockfile diff and say
   so.
5. Security-driven bumps follow the evidence rules above (advisory link,
  regeneration, CI).

## Known gaps (reported, not fixed here)

1. Gradle dependency locking is not enabled in `apps/mobile` (no
   `dependencyLocking` block, no `gradle.lockfile`, no version catalog), so
   transitive Android dependencies float between builds. Enabling it requires
   a Gradle run to write the lockfiles — follow-up.
2. `apps/mobile/gradle/wrapper/gradle-wrapper.properties` pins the
   distribution URL (9.7.1) but has no `distributionSha256Sum`.
3. CI-resolved tooling is unpinned: `brew install xcodegen` (`apps.yml`
   `ios`), `uv --with openapi-spec-validator` (`apps.yml` `api-contracts`),
   and — in the gated-off
   `e2e-real-model` job — floating `pip install torch transformers`
   (`ci-starling-serve.yml`; its other pip installs are exact-pinned).
4. Several action SHAs carry no version comment (`setup-uv` in `apps.yml`,
   `setup-java`, `cache`, `upload/download-artifact`, `cuda-toolkit`,
   `action-gh-release`). Dependabot still updates them; comments only aid
   human review. `test.yml` shows the house style (`# astral-sh/setup-uv
   v10.0.1`) if someone adds them opportunistically.
5. When PR #193 lands, `Cargo.lock` must be committed and the Rust toolchain
   channel recorded, per the E16 consolidation note; this document's rules
   then apply to it unchanged.
