"""Decode Factorio blueprint strings and convert to our Layout schema.

Blueprint string format (per wiki.factorio.com/Blueprint):
    "0" + base64( zlib_deflate( json.dumps(blueprint_dict) ) )

Stdlib only (base64 + zlib + json).

Entities we support (per tier):
    tier-1: electric-mining-drill, stone-furnace, assembling-machine-1,
            transport-belt, inserter, small-electric-pole (skipped),
            substation (skipped), electric-energy-interface (skipped)
    tier-2: steel-furnace, assembling-machine-2, fast-inserter, fast-transport-belt
    tier-3: electric-furnace, assembling-machine-3, stack-inserter, express-transport-belt

Splitters and underground belts are simplified to plain transport-belts (we
lose their splitting/gap behavior, which is fine for our validation goals — the
layouts still produce comparable throughput in most cases).

Small electric poles / substations / EEIs are dropped (we skip electricity).
"""
from __future__ import annotations

import base64
import json
import zlib

from mini_factorio.entities import DIRECTIONS
from mini_factorio.layout import Belt, BeltTile, Inserter, Layout, Machine, Resource

# Factorio 2.0: N=0, E=4, S=8, W=12 (16-way encoding).
FACTORIO_TO_DIR: dict[int, str] = {0: "north", 4: "east", 8: "south", 12: "west",
                                    2: "east", 6: "south", 10: "west", 14: "north"}
# ^ old-style 8-way (0.17): N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7 also seen.
# We map non-cardinals to nearest cardinal (some blueprints use 0.17-era numbers).

# Recipe name migration (0.17 → 2.0).
RECIPE_ALIAS: dict[str, str] = {
    "science-pack-1": "automation-science-pack",
    "science-pack-2": "logistic-science-pack",
    "science-pack-3": "chemical-science-pack",
    "high-tech-science-pack": "utility-science-pack",
    "military-science-pack": "military-science-pack",
    "production-science-pack": "production-science-pack",
}

# Entities we intentionally skip (drop from the layout — power infra, decorations).
SKIP_ENTITIES: set[str] = {
    "small-electric-pole", "medium-electric-pole", "big-electric-pole",
    "substation", "electric-energy-interface", "solar-panel", "accumulator",
    "steam-engine", "steam-turbine", "boiler", "nuclear-reactor",
    "lamp", "programmable-speaker", "constant-combinator", "arithmetic-combinator",
    "decider-combinator", "power-switch", "roboport",
}


def decode_blueprint_string(bp_string: str) -> dict:
    """`"0<base64...>"` → decoded JSON dict."""
    if not bp_string:
        raise ValueError("empty blueprint string")
    if bp_string[0] != "0":
        raise ValueError(f"unexpected blueprint version prefix: {bp_string[0]!r}")
    b = base64.b64decode(bp_string[1:])
    j = zlib.decompress(b)
    return json.loads(j)


def _norm_direction(d: int | None) -> str:
    if d is None:
        return "north"  # Factorio default
    return FACTORIO_TO_DIR.get(d, "north")


def _norm_recipe(recipe: str | None) -> str | None:
    if recipe is None:
        return None
    return RECIPE_ALIAS.get(recipe, recipe)


def blueprint_dict_to_layout(bp: dict, *, grid_pad: int = 2) -> Layout:
    """Translate a Factorio blueprint dict into our Layout schema.

    Coordinates are normalized: original blueprint entities may sit at negative
    or fractional positions; we shift so the min tile is at (grid_pad, grid_pad)
    and infer grid_size from the extent.

    Splitters and underground-belts collapse to plain transport-belt tiles.
    Power infra and non-production entities are dropped. Fast/stack inserters
    and asm-2/3 are preserved by type name (require tier-2/3 in entities.py).
    """
    if "blueprint" in bp:
        bp = bp["blueprint"]
    entities: list[dict] = bp.get("entities", [])

    # First pass: filter + normalize to (name, x_int, y_int, direction, recipe, extra).
    # Blueprint positions are TILE-CENTERED floats. Entity anchor for our schema
    # is TOP-LEFT integer. For a footprint (w, h), top-left = (round(px - w/2), round(py - h/2)).
    # Tile centers are at .5 for 1x1 entities; furnaces (2x2) sit on integer centers.
    from mini_factorio.entities import MACHINES  # local import to avoid cycle

    normalized: list[dict] = []
    xs: list[float] = []
    ys: list[float] = []

    def _footprint_size(name: str) -> tuple[int, int]:
        if name in MACHINES:
            return MACHINES[name].size
        # splitters are 2x1 in the flow direction, but we drop them anyway.
        return (1, 1)

    for e in entities:
        name = e.get("name")
        if not name or name in SKIP_ENTITIES:
            continue
        px = e.get("position", {}).get("x", 0)
        py = e.get("position", {}).get("y", 0)
        w, h = _footprint_size(name)
        # Factorio blueprint position = center of the entity's footprint.
        # For a w×h footprint (top-left at grid (tx, ty)), the center in world
        # coordinates is (tx + (w-1)/2, ty + (h-1)/2). So top-left = pos - (w-1)/2.
        # Result is integer when pos follows Factorio's convention (integer for
        # odd w, half-integer for even w). Use round-half-up (not Python's
        # banker's rounding) to break ties consistently.
        import math
        tx = math.floor(px - (w - 1) / 2 + 0.5)
        ty = math.floor(py - (h - 1) / 2 + 0.5)
        normalized.append({
            "raw": e, "name": name, "tx": tx, "ty": ty, "w": w, "h": h,
        })
        # Track extents (each footprint tile).
        for dx in range(w):
            for dy in range(h):
                xs.append(tx + dx)
                ys.append(ty + dy)

    if not xs:
        return Layout(grid_size=(4, 4), resources=[], budget={}, machines=[],
                      inserters=[], belts=[])

    min_x, min_y = min(xs), min(ys)
    max_x, max_y = max(xs), max(ys)
    shift_x = grid_pad - min_x
    shift_y = grid_pad - min_y
    width = (max_x - min_x + 1) + 2 * grid_pad
    height = (max_y - min_y + 1) + 2 * grid_pad

    machines: list[Machine] = []
    inserters: list[Inserter] = []
    # Belts: we treat each blueprint transport-belt / underground / splitter as
    # a SEPARATE single-tile belt. Our schema requires each Belt to specify an
    # item, but blueprints don't tag belts by item. Group post-hoc by contiguity
    # + assume unknown item ("mixed") — the sim will warn but validation passes.
    belt_tiles_by_item: dict[str, list[BeltTile]] = {}

    m_counter = i_counter = b_counter = 0

    for n in normalized:
        name = n["name"]
        tx = n["tx"] + shift_x
        ty = n["ty"] + shift_y
        e = n["raw"]
        direction = _norm_direction(e.get("direction"))
        recipe = _norm_recipe(e.get("recipe"))

        # --- Machines (miners, furnaces, assemblers) ---
        if name in {"electric-mining-drill", "burner-mining-drill",
                    "stone-furnace", "steel-furnace", "electric-furnace",
                    "assembling-machine-1", "assembling-machine-2",
                    "assembling-machine-3"}:
            m_counter += 1
            kwargs = dict(
                id=f"bp_m{m_counter}",
                type=name,
                x=tx, y=ty,
                direction=direction,
            )
            if "mining-drill" in name:
                # Blueprint doesn't specify target_resource; caller may need to
                # supply it based on nearby resource entities. For now leave None.
                pass
            else:
                kwargs["recipe"] = recipe
            machines.append(Machine(**kwargs))

        # --- Inserters. Preserve tier via Inserter.type. Unknown tiers (long-
        # handed, filter, bulk) collapse to basic 'inserter' since we don't
        # model their special behaviors — sim uses basic throughput for those.
        elif name in {"inserter", "burner-inserter", "long-handed-inserter",
                      "filter-inserter"}:
            i_counter += 1
            inserters.append(Inserter(
                id=f"bp_i{i_counter}", x=tx, y=ty, direction=direction,
                type="inserter",
            ))
        elif name == "fast-inserter":
            i_counter += 1
            inserters.append(Inserter(
                id=f"bp_i{i_counter}", x=tx, y=ty, direction=direction,
                type="fast-inserter",
            ))
        elif name in {"stack-inserter", "bulk-inserter", "stack-filter-inserter"}:
            i_counter += 1
            inserters.append(Inserter(
                id=f"bp_i{i_counter}", x=tx, y=ty, direction=direction,
                type="stack-inserter",
            ))

        # --- Belts / splitters / undergrounds. Group by BELT TIER so different
        # tiers stay distinct (they have different throughput). Item type
        # unknown — sim requires a name; we tag "unknown" and let inserters
        # resolve based on what upstream feeds the belt.
        elif name == "transport-belt":
            belt_tiles_by_item.setdefault(("unknown", "transport-belt"), []).append(
                BeltTile(x=tx, y=ty, direction=direction))
        elif name == "fast-transport-belt":
            belt_tiles_by_item.setdefault(("unknown", "fast-transport-belt"), []).append(
                BeltTile(x=tx, y=ty, direction=direction))
        elif name == "express-transport-belt":
            belt_tiles_by_item.setdefault(("unknown", "express-transport-belt"), []).append(
                BeltTile(x=tx, y=ty, direction=direction))
        elif "underground-belt" in name:
            # Match belt tier by name prefix.
            if name.startswith("fast-"):
                tier = "fast-transport-belt"
            elif name.startswith("express-"):
                tier = "express-transport-belt"
            else:
                tier = "transport-belt"
            belt_tiles_by_item.setdefault(("unknown", tier), []).append(
                BeltTile(x=tx, y=ty, direction=direction))
        elif "splitter" in name:
            if name.startswith("fast-"):
                tier = "fast-transport-belt"
            elif name.startswith("express-"):
                tier = "express-transport-belt"
            else:
                tier = "transport-belt"
            belt_tiles_by_item.setdefault(("unknown", tier), []).append(
                BeltTile(x=tx, y=ty, direction=direction))

        # else: silently skip (chest, pipe, oil, radar, etc.)

    belts: list[Belt] = []
    for (item, belt_type), tiles in belt_tiles_by_item.items():
        b_counter += 1
        belts.append(Belt(id=f"bp_b{b_counter}", item=item, tiles=tiles,
                          type=belt_type))

    return Layout(
        grid_size=(width, height),
        resources=[],
        budget={},
        machines=machines,
        inserters=inserters,
        belts=belts,
    )


def decode_and_translate(bp_string: str) -> Layout:
    """Convenience: blueprint string → Layout."""
    return blueprint_dict_to_layout(decode_blueprint_string(bp_string))
