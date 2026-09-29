#!/usr/bin/env python3
"""Require every experimental package before preparing a public release."""

import argparse
import hashlib
import json
from pathlib import Path


def expected_assets(version: str) -> set[str]:
    return {
        *(f"starling-serve-linux-{backend}.tar.gz" for backend in ("cpu", "vulkan", "cuda")),
        *(f"starling-serve-windows-{backend}.zip" for backend in ("cpu", "vulkan", "cuda")),
        "starling-gpui-linux-x64.tar.gz",
        "starling-gpui-windows-x64.zip",
        f"starling-mobile-{version}-i8mm.apk",
        "SHA256SUMS-android.txt",
    }


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> None:
    root = args.artifacts
    expected = expected_assets(args.version)
    actual = {path.name for path in root.iterdir()}
    if actual != expected:
        raise ValueError(f"Asset mismatch: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}")
    for name in sorted(expected):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Empty or invalid asset: {name}")
    apk = f"starling-mobile-{args.version}-i8mm.apk"
    android_sum = (root / "SHA256SUMS-android.txt").read_text().split()
    if len(android_sum) != 2 or android_sum[1] != apk:
        raise ValueError(f"Malformed SHA256SUMS-android.txt: expected one digest for {apk}")
    if android_sum[0] != checksum(root / apk):
        raise ValueError("Android checksum does not match the APK")

    info = {
        "channel": "experimental",
        "commit": args.sha,
        "version": args.version,
        "tag": args.tag,
        "run_number": int(args.run_number),
        "build_url": f"https://github.com/{args.repository}/actions/runs/{args.run_id}",
        "cuda_architectures": ["120"],
        "android_application_id": "dev.starling.mobile.experimental",
        "desktop_bundled_engines": ["cpu", "vulkan"],
    }
    (root / "build-info.json").write_text(json.dumps(info, indent=2) + "\n")
    files = sorted(expected | {"build-info.json"})
    (root / "SHA256SUMS.txt").write_text("".join(f"{checksum(root / name)}  {name}\n" for name in files))
    source = f"https://github.com/{args.repository}/blob/{args.sha}"
    args.notes.write_text(f"""Experimental build of master at [{args.sha[:12]}](https://github.com/{args.repository}/commit/{args.sha}).

[Build log]({info['build_url']}) · [Installation guide]({source}/docs/experimental-releases.md)

| Device | Downloads |
| --- | --- |
| Linux desktop (CPU or Vulkan GPU) | `starling-gpui-linux-x64.tar.gz` |
| Windows desktop (CPU or Vulkan GPU) | `starling-gpui-windows-x64.zip` |
| NVIDIA RTX on Linux or Windows (CUDA) | Desktop archive + `starling-serve-<platform>-cuda` archive (Manual server mode) |
| Headless or advanced server use | `starling-serve-<platform>-<backend>` archives (CPU, Vulkan, CUDA) |
| Pixel 10 Pro | `{apk}` |

Desktop: download the one desktop archive for your platform and open the app — pick a model in the app and it downloads and verifies it in-app. The CPU and Vulkan engines ship inside the desktop archive; the app picks Vulkan when a Vulkan driver is present and otherwise CPU, and shows which it uses. Standalone server archives remain the headless/advanced option: run one and point the app at it with Manual server mode in settings. CUDA stays a separate download and is not bundled. Linux desktop requires Ubuntu 24.04 or a compatible newer system; the server alone supports Ubuntu 22.04.

Android: install **Starling Experimental**, select **This device**, and download the recommended model. It installs beside Starling Mobile and retains its own data between experimental updates. Its voice keyboard is **Starling Experimental Voice Input**.

CUDA packages target RTX 5090-class SM 120 GPUs only. Use Vulkan or CPU on other GPUs. Desktop packages are unsigned. These are development builds: GPU inference and phone behavior still need testing on real devices. The desktop currently delivers transcripts through Copy; see the guide for hotkey limitations.

`SHA256SUMS.txt` covers all packages and `build-info.json`. Include the commit above when reporting a bug. Older experimental releases remain available for comparison; Android refuses an older version code unless you uninstall first, which deletes its app data.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    for name in ("version", "tag", "sha", "repository", "run-id", "run-number"):
        parser.add_argument(f"--{name}", required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
