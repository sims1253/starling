"""Minimal self-contained JSON Schema structural validator.

Fallback for validating the contract schemas in packages/contracts/ when the
third-party ``jsonschema`` package is not installed: the repo does not declare
it as a dependency, so tests prefer it when importable and fall back here.

Deliberately supports only the keyword subset the contract schemas use:
``$ref`` (local ``#/$defs/...`` pointers), ``type`` (incl. type lists),
``const``, ``enum``, ``properties``, ``required``,
``additionalProperties: false``, ``pattern``, ``minLength``, ``minimum``,
``items``, ``minItems``, ``oneOf``, ``anyOf``. Keep the schemas within this
subset or extend this module; anything else silently passes.
"""

from __future__ import annotations

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


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"minischema only supports local $ref, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


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
        target = _resolve_ref(schema["$ref"], root)
        return errors(instance, target, root, path)

    if "type" in schema:
        names = schema["type"]
        names = [names] if isinstance(names, str) else list(names)
        if not any(_type_ok(instance, n) for n in names):
            return [f"{path}: expected type {'/'.join(names)}, got {type(instance).__name__}"]

    if "const" in schema and instance != schema["const"]:
        found.append(f"{path}: expected const {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
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
        if "items" in schema:
            for i, item in enumerate(instance):
                found.extend(errors(item, schema["items"], root, f"{path}[{i}]"))

    for combiner in ("oneOf", "anyOf"):
        if combiner not in schema:
            continue
        branch_errors = [errors(instance, branch, root, path) for branch in schema[combiner]]
        valid = sum(1 for errs in branch_errors if not errs)
        if combiner == "oneOf" and valid != 1:
            found.append(
                f"{path}: expected exactly 1 matching branch, matched {valid}; "
                f"first branch errors: {branch_errors[0][:3]}"
            )
        if combiner == "anyOf" and valid < 1:
            found.append(f"{path}: no matching branch; first errors: {branch_errors[0][:3]}")

    return found


def is_valid(instance: Any, schema: dict[str, Any]) -> bool:
    return not errors(instance, schema)
