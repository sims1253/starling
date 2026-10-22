#!/usr/bin/env python3
"""Generate the desktop app's third-party notices (licenses of what it ships).

    python3 scripts/licenses/notices.py [--target TRIPLE ...] [--out PATH]

Lists every crate the starling-gpui binary links for each --target (default:
the host), read from `cargo metadata --locked`: the normal dependencies
reachable from starling-gpui, so dev- and build-only crates are left out.
Each crate's own license files (LICENSE*, COPYING*, NOTICE*, ...) are
included verbatim; a crate without one gets the canonical text of every
license its SPDX expression names (licenses/spdx/), and an unknown identifier
is an error, so a new dependency under an unreviewed license fails here
rather than shipping unlisted. The components no package manager reports
(the engine's C/C++ sources, fonts, icons) come from licenses/components.json.

The output (default apps/desktop-gpui/crates/app/notices/third-party-notices.json,
gitignored) is
embedded by crates/app/build.rs; packaging runs this before `cargo build`.
The Android app builds the same schema in Gradle (apps/mobile/app/build.gradle.kts).
Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "apps" / "desktop-gpui"
ROOT_CRATE = "starling-gpui"
LICENSE_FILE = re.compile(r"^(licen[cs]e|copying|notice|unlicense|copyright)([-_.].*)?$", re.I)


def host_target() -> str:
    out = subprocess.run(["rustc", "-vV"], cwd=WORKSPACE, check=True, capture_output=True, text=True).stdout
    return next(line.split(": ", 1)[1] for line in out.splitlines() if line.startswith("host: "))


def normalize_expression(expr: str) -> str:
    # Old crates write "MIT/Apache-2.0" for "MIT OR Apache-2.0".
    return re.sub(r"\s*/\s*", " OR ", expr.strip())


def spdx_ids(expr: str) -> list[str]:
    ids = []
    for tok in re.findall(r"[A-Za-z0-9.+-]+", expr):
        if tok in ("OR", "AND", "WITH"):
            continue
        ids.append(tok.rstrip("+"))
    return ids


def read_text(path: Path) -> str:
    text = path.read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip() + "\n"


class Notices:
    def __init__(self) -> None:
        self.texts: dict[str, str] = {}
        self.missing: list[str] = []

    def add_text(self, text: str) -> str:
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        self.texts.setdefault(key, text)
        return key

    def spdx(self, ident: str, who: str) -> str | None:
        path = ROOT / "licenses" / "spdx" / f"{ident}.txt"
        if not path.is_file():
            self.missing.append(f"{who}: no canonical text for {ident} (add licenses/spdx/{ident}.txt)")
            return None
        return self.add_text(read_text(path))

    def from_spec(self, spec: str, who: str) -> str | None:
        if spec.startswith("spdx:"):
            return self.spdx(spec[5:], who)
        path = ROOT / spec
        if not path.is_file():
            self.missing.append(f"{who}: {spec} does not exist")
            return None
        return self.add_text(read_text(path))


def shipped_crates(target: str) -> list[dict]:
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1", "--filter-platform", target],
        cwd=WORKSPACE, check=True, capture_output=True, text=True).stdout)
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    members = set(meta["workspace_members"])
    root = next(p["id"] for p in meta["packages"] if p["name"] == ROOT_CRATE and p["id"] in members)
    seen, stack = set(), [root]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for dep in nodes[pid]["deps"]:
            if any(kind["kind"] is None for kind in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    return [packages[pid] for pid in seen if pid not in members]


def crate_texts(pkg: dict, notices: Notices) -> list[str]:
    who = f"{pkg['name']} {pkg['version']}"
    crate_dir = Path(pkg["manifest_path"]).parent
    files = []
    if pkg.get("license_file"):
        files.append(crate_dir / pkg["license_file"])
    for entry in sorted(crate_dir.iterdir()):
        if entry.is_file() and LICENSE_FILE.match(entry.name):
            files.append(entry)
        elif entry.is_dir() and entry.name.upper() in ("LICENSES", "LICENSE"):
            files.extend(sorted(f for f in entry.iterdir() if f.is_file()))
    keys = []
    for f in dict.fromkeys(f.resolve() for f in files):
        if f.is_file():
            keys.append(notices.add_text(read_text(f)))
    if not keys:
        expr = normalize_expression(pkg.get("license") or "")
        if not expr:
            notices.missing.append(f"{who}: declares no license and ships no license file")
        for ident in spdx_ids(expr):
            key = notices.spdx(ident, who)
            if key:
                keys.append(key)
    return list(dict.fromkeys(keys))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", help="Rust target triple (repeatable; default: host)")
    ap.add_argument("--out", type=Path, default=WORKSPACE / "crates" / "app" / "notices" / "third-party-notices.json")
    args = ap.parse_args()
    targets = args.target or [host_target()]

    notices = Notices()
    crates: dict[tuple[str, str], dict] = {}
    for target in targets:
        for pkg in shipped_crates(target):
            crates.setdefault((pkg["name"], pkg["version"]), pkg)

    components = []
    for pkg in crates.values():
        components.append({
            "name": pkg["name"],
            "version": pkg["version"],
            "license": normalize_expression(pkg.get("license") or "see license text"),
            "url": pkg.get("repository") or pkg.get("homepage") or f"https://crates.io/crates/{pkg['name']}",
            "texts": crate_texts(pkg, notices),
        })
    for comp in json.loads((ROOT / "licenses" / "components.json").read_text())["components"]:
        if "desktop" not in comp["apps"]:
            continue
        keys = [k for k in (notices.from_spec(s, comp["name"]) for s in comp["texts"]) if k]
        components.append({k: comp[k] for k in ("name", "version", "license", "url")} | {"texts": keys})

    if notices.missing:
        print("third-party notices are incomplete:", *notices.missing, sep="\n  ", file=sys.stderr)
        return 1
    components.sort(key=lambda c: (c["name"].lower(), c["version"]))
    out = {
        "schema": 1,
        "app": "desktop",
        "targets": targets,
        "starling": {"license": "MIT", "text": notices.add_text(read_text(ROOT / "LICENSE"))},
        "components": components,
        "texts": dict(sorted(notices.texts.items())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=None, separators=(",", ":"), ensure_ascii=False) + "\n")
    size = args.out.stat().st_size
    print(f"wrote {args.out}: {len(components)} components, {len(notices.texts)} distinct texts, "
          f"{size / 1e6:.2f} MB ({', '.join(targets)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
