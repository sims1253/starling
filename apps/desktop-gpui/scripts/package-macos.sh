#!/usr/bin/env bash
# Assemble the macOS distribution of the gpui desktop app: a universal
# (aarch64 + x86_64) Starling.app and a compressed disk image. Run on macOS
# with a stable Rust toolchain; output lands in target/package/.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

version="$(sed -n 's/^version = "\(.*\)"$/\1/p' crates/app/Cargo.toml)"
if [[ -z "$version" ]]; then
  echo "error: no version found in crates/app/Cargo.toml" >&2
  exit 1
fi
# Distinguishes CI builds of the same version; local builds report 0.
build_number="${GITHUB_RUN_NUMBER:-0}"

rustup target add aarch64-apple-darwin x86_64-apple-darwin
cargo build --release -p starling-gpui \
  --target aarch64-apple-darwin --target x86_64-apple-darwin

staging="target/package/Starling.app"
rm -rf "$staging"
mkdir -p target/package "$staging/Contents/MacOS" "$staging/Contents/Resources"

# lipo drops the linker's per-arch ad-hoc signatures; a fresh one is re-applied
# below because arm64 macOS refuses to launch unsigned binaries.
lipo -create \
  target/aarch64-apple-darwin/release/starling-gpui \
  target/x86_64-apple-darwin/release/starling-gpui \
  -output "$staging/Contents/MacOS/starling-gpui"
strip "$staging/Contents/MacOS/starling-gpui"

cp assets/icon.icns "$staging/Contents/Resources/AppIcon.icns"

cat > "$staging/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>starling-gpui</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleIdentifier</key><string>dev.starling.dictation</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>Starling</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${version}</string>
  <key>CFBundleVersion</key><string>${build_number}</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key>
  <string>Starling records audio when you press Record so your selected transcription server can turn it into text.</string>
</dict>
</plist>
EOF
plutil -lint "$staging/Contents/Info.plist"

codesign --force --sign - "$staging"

hdiutil create -volname Starling -srcfolder "$staging" -format UDZO -ov \
  "target/package/Starling-macOS-universal.dmg"
