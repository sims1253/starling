"""Minimal self-contained JSON Schema structural validator.

Fallback for validating the contract schemas in packages/contracts/ when the
third-party ``jsonschema`` package is not installed: the repo does not declare
it as a dependency, so tests prefer it when importable and fall back here.

Deliberately supports only the keyword subset the contract schemas use:
``$ref`` (local ``#/$defs/...`` pointers), ``type`` (incl. type lists),
``const``, ``enum``, ``properties``, ``required``,
``additionalProperties: false``, ``pattern``, ``minLength``, ``minimum``,
``items``, ``minItems``, ``maxItems``, ``uniqueItems``, ``oneOf``, ``anyOf``,
``allOf``, ``not`` and ``if``/``then``/``else``. Keep the schemas within this
subset or extend this module; anything else silently passes.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _type_ok(value: Any, name: str) -> bool:
    # JSON Schema treats booleans as distinct from numbers; bool is an int
    # subclass in Python, so exclude it explicitly.
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    raise ValueError(f"minischema does not know type {name!r}")


def _resolve_ref(ref: str, root: dict[str, Any]) -> tuple[Any, str | None]:
    """Resolve a local JSON Pointer; returns (node, error) so a malformed
    pointer becomes a validation error instead of a crash."""
    if not ref.startswith("#/"):
        return None, f"minischema only supports local $ref, got {ref!r}"
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")  # JSON Pointer escapes
        if not isinstance(node, dict) or part not in node:
            return None, f"unresolvable $ref pointer {ref!r}"
        node = node[part]
    return node, None


def _numbers_by_value(value: Any) -> Any:
    """Integral floats as ints, recursively: JSON Schema compares numbers
    by value (1 == 1.0), and bools stay bools."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_numbers_by_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _numbers_by_value(item) for key, item in value.items()}
    return value


def errors(
    instance: Any,
    schema: dict[str, Any],
    root: dict[str, Any] | None = None,
    path: str = "$",
) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    root = schema if root is None else root
    found: list[str] = []

    if schema is True:
        return found
    if schema is False:
        return [f"{path}: schema forbids any value"]

    if "$ref" in schema:
        # 2020-12 behavior: the ref target applies AND any sibling keywords
        # in this same schema object apply too (the code below evaluates them).
        target, ref_error = _resolve_ref(schema["$ref"], root)
        if ref_error is None:
            found.extend(errors(instance, target, root, path))
        else:
            found.append(f"{path}: {ref_error}")

    if "type" in schema:
        names = schema["type"]
        names = [names] if isinstance(names, str) else list(names)
        if not any(_type_ok(instance, n) for n in names):
            return [f"{path}: expected type {'/'.join(names)}, got {type(instance).__name__}"]

    # JSON Schema equality is type-aware: true != 1 even though bool is an int
    # subclass in Python.
    def _json_equal(a: Any, b: Any) -> bool:
        if isinstance(a, bool) != isinstance(b, bool):
            return False
        return a == b

    if "const" in schema and not _json_equal(instance, schema["const"]):
        found.append(f"{path}: expected const {schema['const']!r}, got {instance!r}")
    if "enum" in schema and not any(_json_equal(instance, v) for v in schema["enum"]):
        found.append(f"{path}: {instance!r} not in enum {schema['enum']!r}")

    if isinstance(instance, str):
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            found.append(f"{path}: {instance!r} does not match pattern {schema['pattern']!r}")
        if "minLength" in schema and len(instance) < schema["minLength"]:
            found.append(f"{path}: shorter than minLength {schema['minLength']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            found.append(f"{path}: {instance} below minimum {schema['minimum']}")

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                found.append(f"{path}: missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    found.append(f"{path}: unexpected property {key!r}")
        for key, sub in properties.items():
            if key in instance:
                found.extend(errors(instance[key], sub, root, f"{path}.{key}"))

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            found.append(f"{path}: fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            found.append(f"{path}: more than maxItems {schema['maxItems']}")
        if schema.get("uniqueItems"):
            # Canonical JSON keeps equality type-aware at every depth
            # (true vs 1, {"a": true} vs {"a": 1}).
            seen: set[str] = set()
            for i, item in enumerate(instance):
                key = json.dumps(_numbers_by_value(item), sort_keys=True, separators=(",", ":"))
                if key in seen:
                    found.append(f"{path}[{i}]: duplicate of an earlier item")
                seen.add(key)
        if "items" in schema:
            for i, item in enumerate(instance):
                found.extend(errors(item, schema["items"], root, f"{path}[{i}]"))

    for branch in schema.get("allOf", []):
        found.extend(errors(instance, branch, root, path))
    if "not" in schema and not errors(instance, schema["not"], root, path):
        found.append(f"{path}: matches a schema it must not match")
    if "if" in schema:
        branch = "then" if not errors(instance, schema["if"], root, path) else "else"
        if branch in schema:
            found.extend(errors(instance, schema[branch], root, path))

    for combiner in ("oneOf", "anyOf"):
        if combiner not in schema:
            continue
        branch_errors = [errors(instance, branch, root, path) for branch in schema[combiner]]
        valid = sum(1 for errs in branch_errors if not errs)
        first_errors = branch_errors[0][:3] if branch_errors else "(no branches)"
        if combiner == "oneOf" and valid != 1:
            found.append(
                f"{path}: expected exactly 1 matching branch, matched {valid}; "
                f"first branch errors: {first_errors}"
            )
        if combiner == "anyOf" and valid < 1:
            found.append(f"{path}: no matching branch; first errors: {first_errors}")

    return found


def is_valid(instance: Any, schema: dict[str, Any]) -> bool:
    return not errors(instance, schema)
