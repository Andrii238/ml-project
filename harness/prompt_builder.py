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
        "\"direction\"?: \"north\"|\"east\"|\"south\"|\"west\", "
        "\"recipe\"?: str, \"target_resource\"?: str}\n"
        "{\"op\": \"remove_entity\", \"id\": str}\n"
        "{\"op\": \"add_inserter\", \"id\": str, \"x\": int, \"y\": int, \"direction\": "
        "\"north\"|\"east\"|\"south\"|\"west\"}\n"
        "{\"op\": \"add_belt\", \"id\": str, \"item\": str, \"tiles\": [[x, y, dir], ...]}\n"
        "{\"op\": \"remove_belt\", \"id\": str}\n"
        "{\"op\": \"extend_belt\", \"id\": str, \"tiles\": [[x, y, dir], ...]}\n"
        "```\n"
        "Rules: unique ids, entities in-bounds, no footprint overlap, belt tiles must "
        "be contiguous in the flow direction.\n"
        "\n"
        "IMPORTANT recipe → machine mapping (common source of invalid edits):\n"
        "- `stone-furnace` ONLY runs `iron-plate` or `copper-plate`. Nothing else.\n"
        "- `assembling-machine-1` runs everything else: `iron-gear-wheel`, "
        "`copper-cable`, `electronic-circuit`, `transport-belt`, `inserter`, "
        "`logistic-science-pack`.\n"
        "- Putting a non-smelting recipe on a furnace = invalid edit (zero reward).\n"
        "\n"
        "MINER direction matters: a miner's `direction` is the drop side. A "
        "north-facing miner drops on the tile above (y-1); east drops to the right "
        "(x+size). Belt tile catching the drop must be adjacent in that direction."
    )


def _examples_block() -> str:
    """Two worked examples showing valid edit lists that improve reward.

    Kept short but full: each example shows an input layout snippet and the
    edit list that produces green-science-relevant output. Purpose: teach the
    model (a) the JSON shape, (b) that non-empty edit lists are expected,
    (c) the miner → belt → inserter → machine → inserter → belt pattern.
    """
    return (
        "## Two worked examples\n"
        "\n"
        "### Example 1 — empty grid, build an iron-plate producer\n"
        "Input layout has: iron-ore patch at (0,0) size 3, coal patch at (0,8) size 3, "
        "no machines yet. A minimal producing edit list:\n"
        "```json\n"
        "{\"edits\": [\n"
        "  {\"op\": \"add_entity\", \"id\": \"mi\", \"type\": \"electric-mining-drill\", "
        "\"x\": 0, \"y\": 0, \"target_resource\": \"iron-ore\"},\n"
        "  {\"op\": \"add_entity\", \"id\": \"mc\", \"type\": \"electric-mining-drill\", "
        "\"x\": 0, \"y\": 8, \"target_resource\": \"coal\"},\n"
        "  {\"op\": \"add_entity\", \"id\": \"f1\", \"type\": \"stone-furnace\", "
        "\"x\": 7, \"y\": 4, \"recipe\": \"iron-plate\"},\n"
        "  {\"op\": \"add_belt\", \"id\": \"b_ore\", \"item\": \"iron-ore\", "
        "\"tiles\": [[3,1,\"east\"],[4,1,\"east\"],[5,1,\"south\"],[5,2,\"south\"],"
        "[5,3,\"south\"],[5,4,\"east\"]]},\n"
        "  {\"op\": \"add_belt\", \"id\": \"b_coal\", \"item\": \"coal\", "
        "\"tiles\": [[3,9,\"east\"],[4,9,\"east\"],[5,9,\"north\"],[5,8,\"north\"],"
        "[5,7,\"north\"],[5,6,\"north\"],[5,5,\"north\"]]},\n"
        "  {\"op\": \"add_inserter\", \"id\": \"i_ore\", \"x\": 6, \"y\": 4, "
        "\"direction\": \"east\"},\n"
        "  {\"op\": \"add_inserter\", \"id\": \"i_coal\", \"x\": 6, \"y\": 5, "
        "\"direction\": \"east\"},\n"
        "  {\"op\": \"add_inserter\", \"id\": \"i_plate\", \"x\": 9, \"y\": 4, "
        "\"direction\": \"east\"},\n"
        "  {\"op\": \"add_belt\", \"id\": \"b_plate\", \"item\": \"iron-plate\", "
        "\"tiles\": [[10,4,\"east\"],[11,4,\"east\"]]}\n"
        "]}\n"
        "```\n"
        "Result: ~0.31 iron-plate/sec (fed by iron miner, coal fuel path complete, "
        "output extractor drops onto a belt so backpressure rule is satisfied).\n"
        "\n"
        "### Example 2 — extend an iron-plate chain to make gears\n"
        "Input layout already has iron-plate production running (like Example 1's end "
        "state, but on a 20x20 grid with iron-plate belt going south from (10,4)). "
        "Adding a gear assembler:\n"
        "```json\n"
        "{\"edits\": [\n"
        "  {\"op\": \"add_entity\", \"id\": \"a_gear\", \"type\": \"assembling-machine-1\", "
        "\"x\": 12, \"y\": 4, \"recipe\": \"iron-gear-wheel\"},\n"
        "  {\"op\": \"add_inserter\", \"id\": \"i_plate_in\", \"x\": 11, \"y\": 4, "
        "\"direction\": \"east\"},\n"
        "  {\"op\": \"add_inserter\", \"id\": \"i_gear_out\", \"x\": 15, \"y\": 5, "
        "\"direction\": \"east\"},\n"
        "  {\"op\": \"add_belt\", \"id\": \"b_gear\", \"item\": \"iron-gear-wheel\", "
        "\"tiles\": [[16,5,\"east\"],[17,5,\"east\"]]}\n"
        "]}\n"
        "```\n"
        "Result: ~0.16 gear/sec (bottlenecked by upstream iron-plate rate). Note the "
        "gear recipe runs on `assembling-machine-1`, not a furnace."
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
        _examples_block(),
        _layout_block(layout),
        "## Your reply\nReturn only the edits JSON. Do not return an empty edit list "
        "unless the layout is already saturated — always try to add or improve something.",
    ])


def build_chat_messages(layout: Layout) -> list[dict[str, str]]:
    """Chat-format version for Qwen2.5-Coder-Instruct."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(layout)},
    ]
