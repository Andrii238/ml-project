"""Prompt builder for the simplified green-science env.

Produces a system + user message pair for the LLM. Content per plan (locked):
- Grid contents (ASCII map, tier + direction visible per cell).
- Current chest emission rates.
- Task instructions (goal, recipe, edit vocabulary summary).

The precise edit-JSON schema is defined in Chunk 6 (harness/edit_schema.py).
The prompt references the vocabulary at a summary level; the parser accepts
the schema described there.
"""
from __future__ import annotations

from mini_factorio.entities import ASSEMBLERS
from mini_factorio.layout import Layout


# --------------------------------------------------------------- ASCII grid

# Single-character rendering per cell.
# '.'      empty tile
# 'B'      input-belts chest
# 'I'      input-inserters chest
# 'O'      output-science chest
# '1'/'2'/'3'   assembler footprint of tier 1/2/3
# '>' '<' '^' 'v'   conveyor pointing east/west/north/south (all tiers)
# '+'      two conveyors on the same tile (a perpendicular crossing)
# '?'      unexpected (should not normally happen)

CHEST_CHAR = {
    "input-belts": "B",
    "input-inserters": "I",
    "output-science": "O",
}

DIR_ARROW = {
    "north": "^",
    "south": "v",
    "east":  ">",
    "west":  "<",
}


def render_grid(layout: Layout) -> str:
    w, h = layout.grid_size
    grid: list[list[str]] = [["." for _ in range(w)] for _ in range(h)]

    # Assemblers first — 3x3 footprints, filled with tier digit.
    for a in layout.assemblers:
        ch = str(a.tier)
        for (x, y) in a.footprint:
            if 0 <= x < w and 0 <= y < h:
                grid[y][x] = ch

    # Chests.
    for c in layout.chests:
        if 0 <= c.x < w and 0 <= c.y < h:
            grid[c.y][c.x] = CHEST_CHAR.get(c.kind, "?")

    # Conveyors — arrow. If a tile already has a conveyor arrow, mark '+'.
    for cv in layout.conveyors:
        if not (0 <= cv.x < w and 0 <= cv.y < h):
            continue
        cur = grid[cv.y][cv.x]
        arrow = DIR_ARROW.get(cv.direction, "?")
        if cur in ("^", "v", ">", "<"):
            grid[cv.y][cv.x] = "+"
        elif cur == ".":
            grid[cv.y][cv.x] = arrow
        # If cell already has a chest or assembler, that's an invalid layout;
        # leave the cell showing the higher-priority entity for readability.

    return "\n".join("".join(row) for row in grid)


# --------------------------------------------------------------- entity list

def render_entity_list(layout: Layout) -> str:
    """Compact JSON-like list of placed entities, giving id + fields the ASCII
    grid can't show (tier, id, exact chest position when covered)."""
    lines: list[str] = []
    for c in layout.chests:
        lines.append(f'  {{"id":"{c.id}","kind":"{c.kind}","x":{c.x},"y":{c.y}}}')
    for a in layout.assemblers:
        lines.append(
            f'  {{"id":"{a.id}","type":"assembler","tier":{a.tier},"x":{a.x},"y":{a.y}}}'
        )
    for cv in layout.conveyors:
        lines.append(
            f'  {{"id":"{cv.id}","type":"conveyor","tier":{cv.tier},'
            f'"x":{cv.x},"y":{cv.y},"direction":"{cv.direction}"}}'
        )
    return "[\n" + ",\n".join(lines) + "\n]" if lines else "[]"


# --------------------------------------------------------------- messages

SYSTEM_MESSAGE = (
    "You design a factory layout to produce green science packs. "
    "You emit JSON edits that place entities on the provided grid. "
    "Follow the schema exactly. Use no more than 20 edits. "
    "Reply with the JSON array only, no prose."
)


def _render_recipe_and_map_facts(layout: Layout) -> str:
    w, h = layout.grid_size
    speeds = {t: ASSEMBLERS[t].crafts_per_sec_green_science for t in (1, 2, 3)}
    return (
        f"Grid: {w}x{h}. Coordinates (x, y): x east, y south, origin top-left.\n"
        "Green science recipe: 1 transport-belt + 1 inserter -> 1 pack, "
        "6-second craft time.\n"
        f"Assembler crafts/sec on this recipe: "
        f"asm-1={speeds[1]:.4f}, asm-2={speeds[2]:.4f}, asm-3={speeds[3]:.4f}.\n"
        "Conveyor per-lane throughput: T1=15, T2=30, T3=45 items/sec. "
        "Two lanes per tile; each lane carries at most one item type. "
        "Two conveyors may share a tile only if perpendicular (crossing).\n"
        "Required chests are already present: input-belts, input-inserters, and output-science. "
        "Do not place, remove, or duplicate required chests. Only add assemblers and conveyors.\n"
        "In the active task, all required chests are fixed in the top-left corner: "
        "output-science at (0,0), input-belts at (2,0), input-inserters at (3,0).\n"
        "Assemblers consume required inputs from any adjacent conveyor carrying them. "
        "Assemblers output science onto adjacent conveyors that are empty or already carrying science, not onto input conveyors carrying recipe ingredients. "
        "No inserter entity exists in this simplified environment.\n"
    )


def _render_rates(layout: Layout) -> str:
    return (
        f"Chest emission rates (items/sec):\n"
        f"  input-belts     -> transport-belt: {layout.chest_rates.belts:.3f}\n"
        f"  input-inserters -> inserter:       {layout.chest_rates.inserters:.3f}\n"
    )


EDIT_VOCAB_SUMMARY = (
    "Edit vocabulary (details in schema):\n"
    "  {\"op\":\"place_assembler\", \"tier\":1|2|3, \"x\":int, \"y\":int, \"id\":str} "
    "where x,y are the 3x3 top-left anchor and must keep the whole footprint inside the grid\n"
    "  {\"op\":\"place_conveyor_line\", \"tier\":1|2|3, \"from_x\":int, \"from_y\":int,"
    " \"to_x\":int, \"to_y\":int, \"id\":str} for one straight horizontal/vertical belt line. "
    "The endpoints are excluded; conveyor direction is inferred from from->to.\n"
    "Conveyor-line rules: use one line per straight route; do not split one route into repeated sub-lines. "
    "Never output a line where from_x/from_y equals to_x/to_y. "
    "A complete bus factory usually needs exactly five conveyor-line edits: output trunk, output bus, "
    "belt-input trunk, inserter-input trunk, and input bus.\n"
)


GOAL_MESSAGE = (
    "Goal: use the existing chests, then place assemblers and conveyors so green-science packs "
    "flow from producing assemblers to the output-science chest. Maximizing delivered rate "
    "of green science to the output chest is the main and most important objective. "
    "Higher tiers cost more and unlock a one-time penalty. Use no more than 15 edits.\n"
)


GRID_LEGEND = (
    "Grid legend: '.' empty | 'B' input-belts | 'I' input-inserters | "
    "'O' output-science | '1'/'2'/'3' assembler footprint tier 1/2/3 | "
    "'>' east '<' west '^' north 'v' south conveyor | "
    "'+' two perpendicular conveyors on same tile (crossing).\n"
)


def build_user_message(layout: Layout) -> str:
    # The <<LAYOUT>>…<</LAYOUT>> envelope is machine-readable and used by
    # `training.reward_wrapper.layout_from_prompt` to recover the exact layout
    # the model saw. It's placed at the end so the human-facing view above
    # dominates the model's context.
    envelope = "<<LAYOUT>>" + layout.to_json() + "<</LAYOUT>>"
    return (
        _render_recipe_and_map_facts(layout)
        + "\n"
        + _render_rates(layout)
        + "\n"
        + GRID_LEGEND
        + "Current layout (ASCII grid, y increases downward):\n"
        + "```\n"
        + render_grid(layout)
        + "\n```\n\n"
        + "Placed entities:\n"
        + render_entity_list(layout)
        + "\n\n"
        + EDIT_VOCAB_SUMMARY
        + "\n"
        + GOAL_MESSAGE
        + "\nReply with the JSON array of edits. Maximum 100 edits.\n\n"
        + envelope
    )


def build_chat_messages(layout: Layout) -> list[dict[str, str]]:
    """OpenAI-style chat message list — usable with transformers.apply_chat_template."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": build_user_message(layout)},
    ]
