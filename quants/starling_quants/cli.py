"""Plan/build catalog recipes and hash artifacts without loading any models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PACKAGE = Path(__file__).resolve().parent
RECIPE_DIR = PACKAGE / "recipes" if (PACKAGE / "recipes").is_dir() else PACKAGE.parent / "recipes"


def catalog() -> list[dict]:
    document = json.loads((PACKAGE / "catalog.json").read_text(encoding="utf-8"))
    if document["schema_version"] != 1:
        raise ValueError("Unsupported catalog schema")
    profiles = document["profiles"]
    seen = set()
    for profile in profiles:
        if profile["id"] in seen:
            raise ValueError(f"Duplicate profile: {profile['id']}")
        seen.add(profile["id"])
        if "recipe" in profile:
            recipe = RECIPE_DIR / profile["recipe"]
            if recipe.parent != RECIPE_DIR or not recipe.is_file():
                raise ValueError(f"Missing recipe: {profile['recipe']}")
    return profiles


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def plan(profile_id: str, source: Path, output: Path, binary: Path, imatrix: Path | None) -> dict:
    profile = next((item for item in catalog() if item["id"] == profile_id), None)
    if profile is None:
        raise ValueError(f"Unknown profile: {profile_id}")
    source, output, binary = source.resolve(), output.resolve(), binary.resolve()
    if source == output:
        raise ValueError("Input and output must differ")
    if profile["requires_imatrix"] and imatrix is None:
        raise ValueError(f"{profile_id} requires a calibration importance matrix (--imatrix)")
    command = [str(binary), "--input", str(source), "--output", str(output)]
    if "recipe" in profile:
        command += ["--recipe", str(RECIPE_DIR / profile["recipe"])]
    else:
        command += ["--quant", profile["quant"]]
    if imatrix is not None:
        command += ["--imatrix", str(imatrix.resolve())]
    command += profile["flags"]
    return {"schema_version": 1, "profile": profile, "argv": command,
            "input": str(source), "output": str(output),
            "imatrix": str(imatrix.resolve()) if imatrix is not None else None}


def build(spec: dict) -> dict:
    """Publish a completed artifact, then its provenance, without replacing files.

    The output's parent must exist. Refuse existing files; never overwrite source
    models. A failed quantizer leaves no published artifact or metadata.
    """
    source, output = Path(spec["input"]), Path(spec["output"])
    record_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or record_path.exists():
        raise ValueError("Output or artifact record already exists; choose a new output")
    if not source.is_file():
        raise ValueError(f"Input does not exist: {source}")
    with source.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("Input is not a GGUF file")
    if spec["imatrix"] and not Path(spec["imatrix"]).is_file():
        raise ValueError("Calibration importance matrix does not exist")
    if not output.parent.is_dir():
        raise ValueError("Output directory must exist")
    # Hash dependencies before the process: the record identifies what we asked
    # the quantizer to consume, rather than a mutable model name or URL.
    inputs = {"source_sha256": digest(source), "quantizer_sha256": digest(Path(spec["argv"][0]))}
    if spec["imatrix"]:
        inputs["imatrix_sha256"] = digest(Path(spec["imatrix"]))
    if "recipe" in spec["profile"]:
        inputs["recipe_sha256"] = digest(RECIPE_DIR / spec["profile"]["recipe"])
    with tempfile.TemporaryDirectory(prefix=".starling-quant-", dir=output.parent) as directory:
        temporary = Path(directory) / "model.gguf"
        command = list(spec["argv"])
        command[command.index("--output") + 1] = str(temporary)
        subprocess.run(command, check=True)
        with temporary.open("rb") as stream:
            if stream.read(4) != b"GGUF" or temporary.stat().st_size < 24:
                raise ValueError("Quantizer did not produce a GGUF artifact")
        record = {**spec, **inputs, "created_at": datetime.now(timezone.utc).isoformat(),
                  "size_bytes": temporary.stat().st_size, "sha256": digest(temporary),
                  "evaluation": "not_evaluated"}
        # Hard links publish already-written files atomically and fail if a
        # destination exists. Temporary files live on the same filesystem.
        temporary_record = Path(directory) / "record.json"
        temporary_record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        created_output = False
        created_record = False
        try:
            os.link(temporary, output)
            created_output = True
            os.link(temporary_record, record_path)
            created_record = True
        except BaseException:
            if created_output:
                output.unlink(missing_ok=True)
            if created_record:
                record_path.unlink(missing_ok=True)
            raise
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List checked-in profiles as JSON")
    commands.add_parser("check", help="Validate the catalog and recipe paths")
    for name in ("plan", "build"):
        command = commands.add_parser(name)
        command.add_argument("profile")
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--binary", type=Path, default=Path("build/native-cpu/starling-quantize"))
        command.add_argument("--imatrix", type=Path)
    verify = commands.add_parser("verify", help="Verify an artifact against its recorded hash and size")
    verify.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ("list", "check"):
            profiles = catalog()
            print(json.dumps(profiles if args.command == "list" else {"profiles": len(profiles), "valid": True}, indent=2))
        elif args.command == "verify":
            artifact = args.artifact
            record = json.loads(artifact.with_suffix(artifact.suffix + ".json").read_text(encoding="utf-8"))
            if record["sha256"] != digest(artifact) or record["size_bytes"] != artifact.stat().st_size:
                raise ValueError("Artifact hash or size does not match its record")
            print(json.dumps({"verified": True, "sha256": record["sha256"]}))
        else:
            spec = plan(args.profile, args.input, args.output, args.binary, args.imatrix)
            print(json.dumps(build(spec) if args.command == "build" else spec, indent=2))
        return 0
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f"starling-quants: {error}", file=sys.stderr)
        return 1
