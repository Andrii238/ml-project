"""Parse an LLM completion string into an EditList.

The model is expected to emit a JSON object like:
    {"edits": [{"op": "add_entity", "id": "m1", ...}, ...]}

but real completions often wrap this in ```json ... ``` fences or add explanatory
prose. This parser is deliberately lenient:
1. Strip Markdown code fences if present.
2. Try to parse as JSON.
3. If parsing fails or the top-level object doesn't validate, look for the first
   `{ ... }` substring that does validate.
4. If nothing works, return `(EditList(edits=[]), parse_ok=False)`.

Callers can distinguish "malformed" from "empty on purpose" using the returned
flag. This matches plan.md §Reward wrapper which docks -1.0 on parse failure.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from .edit_schema import EditList

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class ParseResult:
    edits: EditList
    parse_ok: bool
    error: str | None = None


def _strip_fences(s: str) -> str:
    m = _FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    return s.strip()


def _first_json_object(s: str) -> str | None:
    """Return the substring of the first balanced {...} block in s, or None."""
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start:i + 1]
        start = s.find("{", start + 1)
    return None


def parse_edits(completion: str) -> ParseResult:
    body = _strip_fences(completion)
    for candidate in (body, _first_json_object(body) or ""):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):  # tolerate bare list, wrap in {"edits": ...}
            data = {"edits": data}
        try:
            return ParseResult(EditList.model_validate(data), True)
        except ValidationError as e:
            last_err = str(e)
            continue
    return ParseResult(
        EditList(edits=[]), False,
        error=f"could not parse edits from completion: {locals().get('last_err', 'no JSON found')}",
    )
