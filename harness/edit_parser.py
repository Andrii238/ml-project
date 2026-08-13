"""Parse an LLM response into a list of typed edits.

Tolerates common noise:
- prose before/after the JSON array
- markdown code fences around the JSON
- trailing text after a valid JSON array

Fails cleanly on:
- no `[` found (returns [], parse_error explaining)
- JSON syntax error inside the array
- truncated JSON (unclosed bracket)

Per-edit validation errors are collected and returned alongside the successful
edits — a partial parse still returns whichever edits validated.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .edit_schema import Edit, parse_edit

MAX_EDITS = 100


@dataclass
class ParseResult:
    edits: list[Edit] = field(default_factory=list)
    edit_errors: list[str] = field(default_factory=list)  # per-item validation errors
    parse_error: str | None = None                        # top-level JSON error, if any

    @property
    def ok(self) -> bool:
        return self.parse_error is None and not self.edit_errors


# --------------------------------------------------------------- extraction

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_array(text: str) -> tuple[str | None, str | None]:
    """Return (array_text, error). Strips markdown fences and any prose
    around the array. If generation stops mid-list, keep all complete objects
    before the truncation and close the array."""
    # First strip markdown fences if present. Take the first fenced block.
    fenced = _FENCE_RE.findall(text)
    candidate = fenced[0] if fenced else text

    start = candidate.find("[")
    if start < 0:
        return None, "no '[' found in output"

    # Walk forward tracking bracket depth. Ignore brackets inside strings.
    depth = 0
    in_str = False
    escape = False
    end = -1
    for i, ch in enumerate(candidate[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        repaired = _repair_truncated_array(candidate[start:])
        if repaired is None:
            return None, "unterminated JSON array (truncated?)"
        return repaired, None

    return candidate[start:end + 1], None


def _repair_truncated_array(text: str) -> str | None:
    """Build a valid JSON array from complete top-level objects in a truncated
    array. This converts endless edit streams into a bounded candidate layout."""
    objects: list[str] = []
    depth = 0
    in_str = False
    escape = False
    obj_start = -1
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                objects.append(text[obj_start:i + 1])
                if len(objects) >= MAX_EDITS:
                    break
                obj_start = -1
    if not objects:
        return None
    return "[" + ",".join(objects[:MAX_EDITS]) + "]"


# --------------------------------------------------------------- main

def parse_edits(text: str) -> ParseResult:
    result = ParseResult()
    array_text, err = _extract_json_array(text)
    if err is not None:
        result.parse_error = err
        return result

    try:
        raw = json.loads(array_text)
    except json.JSONDecodeError as e:
        result.parse_error = f"JSON decode error: {e.msg} at pos {e.pos}"
        return result

    if not isinstance(raw, list):
        result.parse_error = f"top-level JSON is not a list, got {type(raw).__name__}"
        return result

    for i, item in enumerate(raw[:MAX_EDITS]):
        edit, err = parse_edit(item)
        if edit is not None:
            result.edits.append(edit)
        else:
            result.edit_errors.append(f"edit[{i}]: {err}")

    return result
