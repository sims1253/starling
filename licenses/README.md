# Third-party notices

The desktop and Android apps each carry a Licenses page that lists every
open-source component they ship with its license text. The lists are
generated from the builds' own dependency graphs, so they follow dependency
updates without hand edits:

- **Desktop:** `scripts/licenses/notices.py` reads `cargo metadata` for the
  targets being packaged and writes
  `apps/desktop-gpui/crates/app/notices/third-party-notices.json`, which
  `crates/app/build.rs` embeds. Packaging (`package-desktop.yml`,
  `package-macos.sh`) and the desktop CI lane run it; run it yourself to see
  the list in a development build.
- **Android:** `apps/mobile/app/build.gradle.kts` resolves each variant's
  runtime classpath at build time and writes
  `assets/licenses/third-party-notices.json`.

Both use each component's own license files where it ships them (crates), or
the canonical text from `spdx/` for the license it declares. What this
directory holds:

| File | Purpose |
| --- | --- |
| `components.json` | Components no package manager reports: the engine's C/C++ sources (ggml, dr_wav, cpp-httplib, Vulkan headers, LiteRT-derived code), fonts, icons, the Rust standard library, and which app ships each. Update it when vendoring something new. |
| `android-dependencies.json` | License rules for the Android app's Maven artifacts, by group prefix. The build fails on an artifact no rule matches: check its license, then add a rule. |
| `texts/` | License texts of components in `components.json` that have no file in the tree. |
| `spdx/` | Canonical license texts from [spdx/license-list-data](https://github.com/spdx/license-list-data) v3.29.0. The desktop generator fails on a crate whose license has no text here; add the missing `<ID>.txt` from that repository. |
