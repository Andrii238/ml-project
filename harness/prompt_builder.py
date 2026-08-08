"""Assemble the prompt sent to the LLM.

Sections:
1. Game rules — recipe database, machine specs, belt/inserter constants.
   Static across prompts. Kept short so the layout dominates the token budget.
2. Edit schema — JSON grammar the model must emit.
3. Task — task statement plus the current layout JSON.
4. Reminder — output-format constraint.

The prompt is designed for `Qwen2.5-Coder-1.5B-Instruct` in instruction/chat
form. The `build_prompt` return is a plain string with a system+user layout;
call `build_chat_messages` for the messages list version.
"""
from __future__ import annotations

import json

from mini_factorio.entities import (
    BELT_SPEED,
    INSERTER_THROUGHPUT,
    MACHINES,
    PLACEABLES,
    RESOURCE_TYPES,
)
from mini_factorio.layout import Layout
from mini_factorio.recipes import GREEN_SCIENCE_ITEM, RECIPES


def _rules_block() -> str:
    lines = ["## Game rules"]
    lines.append(f"Goal: maximize `{GREEN_SCIENCE_ITEM}` production per second.")
    lines.append("")
    lines.append("### Machines")
    for mid, spec in MACHINES.items():
        cost = ", ".join(f"{v} {k}" for k, v in spec.cost.items())
        fuel = "coal fuel required" if spec.fuel_category == "chemical" else "no fuel"
        lines.append(
            f"- `{mid}` size {spec.size[0]}x{spec.size[1]}, speed {spec.crafting_speed}, "
            f"cost [{cost}], {fuel}"
        )
    lines.append("")
    lines.append("### Placeables (1x1 each)")
    for pid, spec in PLACEABLES.items():
        cost = ", ".join(f"{v:g} {k}" for k, v in spec.cost_per_place.items())
        lines.append(f"- `{pid}` cost per tile [{cost}]")
    lines.append(
        f"- Belt throughput: {BELT_SPEED} items/sec. "
        f"Inserter throughput: {INSERTER_THROUGHPUT} items/sec."
    )
    lines.append("")
    lines.append("### Recipes (time in seconds at speed=1.0)")
    for rid, r in RECIPES.items():
        ing = ", ".join(f"{v} {k}" for k, v in r.ingredients.items()) or "-"
        out = ", ".join(f"{v} {k}" for k, v in r.products.items())
        lines.append(f"- `{rid}` [{r.kind}, {r.time}s]: {ing} -> {out}")
    lines.append("")
    lines.append(f"### Resource types on the map: {', '.join(RESOURCE_TYPES)}")
    lines.append("Miners must sit on a resource patch of their target_resource type.")
    lines.append("Furnaces need both an ore input and a coal input (via inserters).")
    lines.append(
        "Inserter with direction=`east` picks from the tile to its west and drops "
        "on the tile to its east; other directions analogous. Pickup/drop tile must "
        "be either a machine footprint tile or a belt tile."
    )
    return "\n".join(lines)


def _edit_schema_block() -> str:
    return (
        "## Edit schema\n"
        "Reply with a JSON object `{\"edits\": [ ... ]}` where each edit is one of:\n"
        "```\n"
        "{\"op\": \"add_entity\", \"id\": str, \"type\": str, \"x\": int, \"y\": int, "
        "\"recipe\"?: str, \"target_resource\"?: str}\n"
        "{\"op\": \"remove_entity\", \"id\": str}\n"
        "{\"op\": \"add_inserter\", \"id\": str, \"x\": int, \"y\": int, \"direction\": "
        "\"north\"|\"east\"|\"south\"|\"west\"}\n"
        "{\"op\": \"add_belt\", \"id\": str, \"item\": str, \"tiles\": [[x, y, dir], ...]}\n"
        "{\"op\": \"remove_belt\", \"id\": str}\n"
        "{\"op\": \"extend_belt\", \"id\": str, \"tiles\": [[x, y, dir], ...]}\n"
        "```\n"
        "Rules: unique ids, entities in-bounds, no footprint overlap, belt tiles must "
        "be contiguous in the flow direction."
    )


def _layout_block(layout: Layout) -> str:
    return (
        "## Current layout\n"
        "```json\n" + layout.to_json(indent=2) + "\n```"
    )


SYSTEM_PROMPT = (
    "You are a Factorio factory-design assistant. Given a partial factory layout, "
    "propose a JSON list of edits that will increase green-science-pack production "
    "per second. Only output the JSON, no prose."
)


def build_prompt(layout: Layout) -> str:
    return "\n\n".join([
        _rules_block(),
        _edit_schema_block(),
        _layout_block(layout),
        "## Your reply\nReturn only the edits JSON.",
    ])


def build_chat_messages(layout: Layout) -> list[dict[str, str]]:
    """Chat-format version for Qwen2.5-Coder-Instruct."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(layout)},
    ]
