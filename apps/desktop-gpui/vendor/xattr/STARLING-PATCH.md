# xattr 0.2.3 compatibility patch

Source: https://crates.io/crates/xattr/0.2.3 (upstream commit
`e42c1da499b72103a9c9ef5693ff31f998ad31e6`). MIT OR Apache-2.0; both license
files are retained. Sources, tests, and manifests come from the published crate.

GPUI 0.2.2 uses zed-async-tar 0.5.0-zed, which requires xattr ^0.2;
upgrading directly to xattr 1.x does not satisfy that requirement.
The only source change is in `src/sys/mod.rs`: Linux/Android use ENODATA
for a missing attribute, replacing libc's removed ENOATTR alias. macOS/BSD
keep their existing ENOATTR behavior. The upstream tests exercise missing
attributes and set/get/list/remove through both paths and file descriptors.

Remove this Cargo patch once GPUI's archive dependency accepts a maintained
xattr release, or an upstream 0.2.x compatibility release is available.

From apps/desktop-gpui, test with:

```sh
cargo test --manifest-path vendor/xattr/Cargo.toml
```
