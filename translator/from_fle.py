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
from mini_factorio.layout import DIR_DELTA, OPPOSITE, Belt, BeltTile, Inserter, Layout, Machine, Resource
from mini_factorio.recipes import RECIPES

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


def _split_belt_tiles(tiles: list[BeltTile]) -> list[list[BeltTile]]:
    """Group belt tiles into linear connected chains along their flow direction.

    A chain breaks at forks/merges (a tile with in-degree > 1 starts a new belt).
    Tiles in a loop or unreachable from a start are emitted as separate belts.
    """
    tile_map = {(t.x, t.y): t for t in tiles}
    in_degree: dict[tuple[int, int], int] = {k: 0 for k in tile_map}
    for (x, y), t in tile_map.items():
        dx, dy = DIR_DELTA[t.direction]
        nxt = (x + dx, y + dy)
        if nxt in tile_map:
            in_degree[nxt] += 1

    visited: set[tuple[int, int]] = set()
    chains: list[list[BeltTile]] = []

    def _walk(start: tuple[int, int]) -> list[BeltTile]:
        chain: list[BeltTile] = []
        cur: tuple[int, int] | None = start
        while cur is not None and cur not in visited:
            visited.add(cur)
            t = tile_map[cur]
            chain.append(t)
            dx, dy = DIR_DELTA[t.direction]
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt not in tile_map or in_degree[nxt] > 1:
                cur = None
            else:
                cur = nxt
        return chain

    for start in [k for k in tile_map if in_degree[k] != 1]:
        if start not in visited:
            chains.append(_walk(start))
    for k in tile_map:
        if k not in visited:
            chains.append(_walk(k))
    return chains


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
        for chain in _split_belt_tiles(tiles):
            b_counter += 1
            belts.append(Belt(id=f"bp_b{b_counter}", item=item, tiles=chain,
                              type=belt_type))

    return Layout(
        grid_size=(width, height),
        resources=[],
        budget={},
        machines=machines,
        inserters=inserters,
        belts=belts,
    )


def _machine_output_item(m: Machine) -> str | None:
    if "mining-drill" in m.type:
        return m.target_resource
    if m.recipe and m.recipe in RECIPES:
        products = RECIPES[m.recipe].products
        if len(products) == 1:
            return next(iter(products))
    return None


def infer_belt_items(layout: Layout) -> Layout:
    """Infer belt items by walking connections. Sets each belt.item from
    'unknown' to the inferred item name where possible.

    Forward: machine output → picking inserter → belt.
    Backward: belt → dropping inserter → consumer machine (recipe input inference).
    """
    machines_by_id = {m.id: m for m in layout.machines}
    tile_owner: dict[tuple[int, int], tuple[str, str]] = {}
    for m in layout.machines:
        for t in layout.machine_footprint(m):
            tile_owner[t] = ("machine", m.id)
    for b in layout.belts:
        for bt in b.tiles:
            tile_owner[(bt.x, bt.y)] = ("belt", b.id)
    belts_by_id = {b.id: b for b in layout.belts}

    def _inserter_source_sink(i: Inserter) -> tuple[tuple[str, str] | None,
                                                     tuple[str, str] | None]:
        dx, dy = DIR_DELTA[i.direction]
        drop = (i.x + dx, i.y + dy)
        odx, ody = DIR_DELTA[OPPOSITE[i.direction]]
        pickup = (i.x + odx, i.y + ody)
        return tile_owner.get(pickup), tile_owner.get(drop)

    # Mutable belt items.
    belt_items: dict[str, str] = {b.id: b.item for b in layout.belts}
    changed = True
    while changed:
        changed = False
        for i in layout.inserters:
            src, snk = _inserter_source_sink(i)
            # Forward: from known-item source → set drop-target belt item.
            item = None
            if src and src[0] == "machine":
                item = _machine_output_item(machines_by_id[src[1]])
            elif src and src[0] == "belt":
                bi = belt_items.get(src[1])
                if bi and bi != "unknown":
                    item = bi
            if item and snk and snk[0] == "belt":
                if belt_items[snk[1]] == "unknown":
                    belt_items[snk[1]] = item
                    changed = True

        # Backward: a belt whose consumer inserter drops into a known-recipe
        # machine, and the machine has only one input ingredient not yet
        # supplied by another known source → item = that ingredient.
        for b in layout.belts:
            if belt_items[b.id] != "unknown":
                continue
            belt_tiles = {(t.x, t.y) for t in b.tiles}
            for i in layout.inserters:
                odx, ody = DIR_DELTA[OPPOSITE[i.direction]]
                pickup = (i.x + odx, i.y + ody)
                if pickup not in belt_tiles:
                    continue
                dx, dy = DIR_DELTA[i.direction]
                drop = (i.x + dx, i.y + dy)
                sink = tile_owner.get(drop)
                if not sink or sink[0] != "machine":
                    continue
                m = machines_by_id[sink[1]]
                if not m.recipe or m.recipe not in RECIPES:
                    continue
                needed = set(RECIPES[m.recipe].ingredients.keys())
                # Items already supplied by other inserters (with known source).
                supplied: set[str] = set()
                for other in layout.inserters:
                    if other.id == i.id:
                        continue
                    o_dx, o_dy = DIR_DELTA[other.direction]
                    o_drop = (other.x + o_dx, other.y + o_dy)
                    if o_drop != drop:
                        continue
                    o_src, _ = _inserter_source_sink(other)
                    if o_src and o_src[0] == "machine":
                        it = _machine_output_item(machines_by_id[o_src[1]])
                        if it:
                            supplied.add(it)
                    elif o_src and o_src[0] == "belt":
                        it = belt_items.get(o_src[1])
                        if it and it != "unknown":
                            supplied.add(it)
                remaining = needed - supplied
                if len(remaining) == 1:
                    belt_items[b.id] = next(iter(remaining))
                    changed = True
                    break

    new_belts = [b.model_copy(update={"item": belt_items[b.id]}) for b in layout.belts]
    return layout.model_copy(update={"belts": new_belts})


def decode_and_translate(bp_string: str) -> Layout:
    """Convenience: blueprint string → Layout with items inferred."""
    lay = blueprint_dict_to_layout(decode_blueprint_string(bp_string))
    return infer_belt_items(lay)
