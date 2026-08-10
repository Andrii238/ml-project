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
_TRAILING_COMMA_RE = re.compile(r",(\s*[\]}])")


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


def _drop_mismatched_closers(s: str) -> str:
    """Drop `]` or `}` that don't match the current open bracket.

    Fixes the common small-model failure where the model closes the edit
    list as `{[{}}]` instead of `{[{}]}` — brackets balanced by count but
    wrong nesting.
    """
    stack: list[str] = []
    out: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "[{":
            stack.append(ch)
            out.append(ch)
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
                out.append(ch)
            # else: drop this mismatched closer
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
                out.append(ch)
            # else: drop this mismatched closer
        else:
            out.append(ch)
    return "".join(out)


def _repair_truncated_json(s: str) -> str:
    """Close unbalanced strings/brackets/braces at the tail of `s`.

    Handles the common failure mode where max_new_tokens truncates the model's
    reply mid-object. Walks the string once, tracking the open-bracket stack,
    then appends matching closers in reverse. Trailing commas are stripped
    before closing. Does not touch valid input.
    """
    stack: list[str] = []
    in_str = False
    esc = False
    last_non_ws = 0
    for i, ch in enumerate(s):
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
            elif ch in "[{":
                stack.append(ch)
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
        if not ch.isspace():
            last_non_ws = i
    if not stack and not in_str:
        return s
    out = s[: last_non_ws + 1]
    if in_str:
        # If we truncated inside a string with a dangling backslash, drop it.
        if out.endswith("\\"):
            out = out[:-1]
        out += '"'
    # Drop a trailing comma or partial key like `,\n  "` before closing.
    out = re.sub(r",\s*$", "", out)
    out = re.sub(r',\s*"[^"]*$', "", out)
    for opener in reversed(stack):
        out += "]" if opener == "[" else "}"
    return out


def parse_edits(completion: str) -> ParseResult:
    body = _strip_fences(completion)
    # Strip trailing commas which small models emit constantly.
    body_clean = _TRAILING_COMMA_RE.sub(r"\1", body)
    body_dropmis = _drop_mismatched_closers(body_clean)
    candidates = [
        body,
        body_clean,
        _first_json_object(body) or "",
        _first_json_object(body_clean) or "",
        _repair_truncated_json(body_clean),
        body_dropmis,
        _repair_truncated_json(body_dropmis),
        _first_json_object(body_dropmis) or "",
    ]
    last_err = "no JSON found"
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = f"json.loads: {e}"
            continue
        if isinstance(data, list):  # tolerate bare list, wrap in {"edits": ...}
            data = {"edits": data}
        elif isinstance(data, dict) and "op" in data and "edits" not in data:
            # A single edit dict — wrap so it's not silently dropped.
            data = {"edits": [data]}
        elif not (isinstance(data, dict) and "edits" in data):
            # Missing the top-level "edits" key — reject rather than let
            # pydantic silently produce an empty EditList (extras are ignored).
            last_err = "top-level object has no 'edits' key"
            continue
        try:
            return ParseResult(EditList.model_validate(data), True)
        except ValidationError as e:
            last_err = str(e)
            continue
    return ParseResult(
        EditList(edits=[]), False,
        error=f"could not parse edits from completion: {last_err}",
    )
