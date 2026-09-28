"""Validate and summarize STARLING_GRAPH_EXPORT_DIR topology snapshots.

The snapshots omit tensor values and operation semantics. This tool reports
their structure and always leaves equivalence as unknown.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


HEX_PARAMS = re.compile(r"[0-9a-f]{128}\Z")


def snapshot_paths(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        found.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    if not found:
        raise ValueError("no graph snapshots found")
    return found


def inspect(path: Path) -> tuple[int, int, Counter[str]]:
    graph = json.loads(path.read_text())
    if (graph.get("schema") != 1 or graph.get("semantics") != "not_encoded"
            or graph.get("leaf_values") != "not_exported"):
        raise ValueError(f"{path}: unsupported snapshot semantics")
    tensors = graph["tensors"]
    if not tensors or [tensor["id"] for tensor in tensors] != list(range(len(tensors))):
        raise ValueError(f"{path}: tensor IDs are missing or repeated")
    def valid_id(value: int) -> bool:
        return type(value) is int and 0 <= value < len(tensors)
    for key in ("graph_nodes", "captures", "side_effect_roots"):
        if any(not valid_id(value) for value in graph[key]):
            raise ValueError(f"{path}: invalid {key} reference")
    if not valid_id(graph["output"]):
        raise ValueError(f"{path}: invalid output reference")
    ops: Counter[str] = Counter()
    for tensor in tensors:
        if (len(tensor["ne"]) != 4 or len(tensor["nb"]) != 4
                or len(tensor["src"]) != 10
                or not HEX_PARAMS.fullmatch(tensor["op_params_hex"])):
            raise ValueError(f"{path}: incomplete metadata for tensor {tensor['id']}")
        if any(ref is not None and not valid_id(ref) for ref in tensor["src"]):
            raise ValueError(f"{path}: invalid source reference for tensor {tensor['id']}")
        if tensor["view_src"] is not None and not valid_id(tensor["view_src"]):
            raise ValueError(f"{path}: invalid view source for tensor {tensor['id']}")
        ops[tensor["op"]] += 1
    return len(graph["graph_nodes"]), len(tensors), ops


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+", help="snapshot files or directories")
    args = parser.parse_args()
    try:
        files = snapshot_paths(args.paths)
        summaries = [inspect(path) for path in files]
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")
    ops = sum((summary[2] for summary in summaries), Counter())
    print(f"snapshots={len(files)} graph_nodes_min={min(x[0] for x in summaries)} "
          f"graph_nodes_max={max(x[0] for x in summaries)} "
          f"tensors_max={max(x[1] for x in summaries)}")
    print("ops=" + ",".join(f"{op}:{count}" for op, count in ops.most_common()))
    print("equivalence=unknown (leaf values and ggml op semantics are not encoded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
