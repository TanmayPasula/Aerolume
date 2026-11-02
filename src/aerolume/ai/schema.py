"""Validation of tool inputs against their JSON schemas (the subset the tools use).

Why the assistant validates every tool input itself, whatever the model:

* **Claude** streams tool inputs with ``eager_input_streaming``: the API then no
  longer buffers and validates each parameter, and the SDK accumulates the JSON
  with a *tolerant* parser that can return a silently truncated object for
  malformed input (an unescaped quote ends a string early, trailing garbage is
  dropped) instead of raising (claude-api skill, tool-use concepts).
* **Local models** (Ollama) have no schema enforcement at all: an 8-billion
  parameter model regularly sends ``"20"`` for a number, a single value where a
  list belongs, or invents a parameter.

So a tool never runs on input that does not match its schema. :func:`coerce`
first applies the few *lossless* repairs a small model needs (numeric strings
to numbers, ``"true"`` to ``True``, one value to a one-element list), then
:func:`validate` checks the result and returns readable problems, which go back
to the model as an error tool result so it can retry.

The supported subset is what Anthropic's strict tool schemas allow and what the
tools use: ``type`` (one or a list), ``properties``, ``required``,
``additionalProperties: false``, ``enum``, ``items``, ``anyOf`` and
``description``. Numeric and length limits are deliberately absent from the
schemas (strict mode rejects them); the tools check ranges themselves and say
why a value is out of range.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_TYPES = ("object", "array", "string", "number", "integer", "boolean", "null")


def _types(schema: Mapping[str, Any]) -> tuple[str, ...]:
    t = schema.get("type")
    if t is None:
        return ()
    return tuple(t) if isinstance(t, (list, tuple)) else (str(t),)


def _is(value: Any, t: str) -> bool:
    if t == "object":
        return isinstance(value, Mapping)
    if t == "array":
        return isinstance(value, (list, tuple))
    if t == "string":
        return isinstance(value, str)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "null":
        return value is None
    if isinstance(value, bool):          # bool is an int subclass; JSON keeps them apart
        return False
    if t == "integer":
        return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value)
                                          and value.is_integer())
    if t == "number":
        return isinstance(value, (int, float)) and math.isfinite(float(value))
    return False


def _number_from_text(text: str) -> float | int | None:
    s = text.strip().replace("_", "")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return int(v) if v.is_integer() and "." not in s and "e" not in s.lower() else v


def coerce(schema: Mapping[str, Any], value: Any) -> Any:
    """``value`` with lossless type repairs applied where the schema asks for another type.

    Only repairs that cannot change the meaning are made: a numeric string
    becomes a number, "true"/"false" a boolean, an integral float an int, a
    scalar a one-element list, and a JSON-ish "null" string None. Anything else
    is returned unchanged for :func:`validate` to report.
    """
    types = _types(schema)
    if "anyOf" in schema and not types:
        for sub in schema["anyOf"]:
            if not validate(sub, value):
                return value
        for sub in schema["anyOf"]:
            fixed = coerce(sub, value)
            if not validate(sub, fixed):
                return fixed
        return value
    if types and any(_is(value, t) for t in types):
        if "object" in types and isinstance(value, Mapping):
            props = schema.get("properties", {})
            return {k: (coerce(props[k], v) if k in props else v) for k, v in value.items()}
        if "array" in types and isinstance(value, (list, tuple)) and "items" in schema:
            return [coerce(schema["items"], v) for v in value]
        if "integer" in types and isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if ("number" in types or "integer" in types):
            v = _number_from_text(value)
            if v is not None and ("number" in types or float(v).is_integer()):
                return int(v) if "integer" in types and "number" not in types else v
        if "boolean" in types and low in ("true", "false"):
            return low == "true"
        if "null" in types and low in ("null", "none", ""):
            return None
    if "array" in types and value is not None and not isinstance(value, (list, tuple, Mapping)):
        item_schema = schema.get("items", {})
        fixed = coerce(item_schema, value)
        if not validate(item_schema, fixed):
            return [fixed]
    return value


def validate(schema: Mapping[str, Any], value: Any, path: str = "") -> list[str]:
    """Problems with ``value`` against ``schema`` (empty when it is valid).

    Messages name the parameter path ("alpha_end: expected a number, got
    'twelve'"), so a model can fix the call from the message alone.
    """
    where = path or "input"
    if "anyOf" in schema and not _types(schema):
        options = [validate(sub, value, path) for sub in schema["anyOf"]]
        if any(not o for o in options):
            return []
        return [f"{where}: does not match any allowed form ({'; '.join(o[0] for o in options if o)})"]
    types = _types(schema)
    if types and not any(_is(value, t) for t in types):
        want = " or ".join("an integer" if t == "integer" else f"a {t}" if t != "array" else "a list"
                           for t in types)
        shown = repr(value)
        if len(shown) > 60:
            shown = shown[:57] + "..."
        return [f"{where}: expected {want}, got {shown}"]
    problems: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(v) for v in schema["enum"])
        problems.append(f"{where}: {value!r} is not one of {allowed}")
    if isinstance(value, Mapping) and ("object" in types or "properties" in schema):
        props: Mapping[str, Any] = schema.get("properties", {})
        for key in schema.get("required", ()):
            if key not in value:
                problems.append(f"{path + '.' if path else ''}{key}: missing (required)")
        if schema.get("additionalProperties", True) is False:
            extra = [k for k in value if k not in props]
            if extra:
                problems.append(f"{where}: unknown parameter{'s' if len(extra) > 1 else ''} "
                                f"{', '.join(map(str, extra))} (allowed: {', '.join(props) or 'none'})")
        for key, sub in props.items():
            if key in value:
                problems.extend(validate(sub, value[key], f"{path + '.' if path else ''}{key}"))
    if isinstance(value, (list, tuple)) and "items" in schema:
        for i, item in enumerate(value):
            problems.extend(validate(schema["items"], item, f"{where}[{i}]"))
    return problems


def check_schema(schema: Mapping[str, Any], path: str = "schema") -> list[str]:
    """Problems that would make Anthropic's strict tool mode, or this validator, reject a schema.

    Every object needs ``additionalProperties: false`` and only known keys; types must be
    JSON types. Used by the tests so a new tool cannot ship an unsupported schema.
    """
    problems: list[str] = []
    allowed_keys = {"type", "properties", "required", "additionalProperties", "enum", "items", "anyOf",
                    "description", "default"}
    unknown = set(schema) - allowed_keys
    if unknown:
        problems.append(f"{path}: unsupported keys {sorted(unknown)}")
    for t in _types(schema):
        if t not in _TYPES:
            problems.append(f"{path}: unknown type {t!r}")
    if "object" in _types(schema):
        if schema.get("additionalProperties") is not False:
            problems.append(f"{path}: objects must set additionalProperties: false")
        props = schema.get("properties", {})
        for key in schema.get("required", ()):
            if key not in props:
                problems.append(f"{path}: required {key!r} is not a property")
        for key, sub in props.items():
            if not isinstance(sub.get("description", ""), str):
                problems.append(f"{path}.{key}: description must be text")
            problems.extend(check_schema(sub, f"{path}.{key}"))
    if "items" in schema:
        problems.extend(check_schema(schema["items"], f"{path}[]"))
    for i, sub in enumerate(schema.get("anyOf", ())):
        problems.extend(check_schema(sub, f"{path}.anyOf[{i}]"))
    return problems


__all__ = ["check_schema", "coerce", "validate"]
