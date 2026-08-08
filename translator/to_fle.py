"""Convert a Mini-Factorio Layout to Factorio-compatible outputs.

Two output modes:

- `layout_to_lua_commands(layout)` — a list of Lua strings suitable for RCON via
  FLE's `instance.eval(...)`. The first command clears the surface; the rest
  create entities via `game.surfaces[1].create_entity{...}`. Also inserts a
  substation plus an electric-energy-interface (creative-mode infinite power
  source) since our simulator skips the electricity subsystem — plan.md
  §Electricity.

- `layout_to_blueprint_dict(layout)` — the raw dict form of Factorio's
  blueprint schema. This is what (JSON → zlib deflate → base64 → '0' prefix)
  becomes a shareable blueprint string. We return the dict for inspection; the
  binary encoding is orthogonal.

Factorio 2.0 conventions:
- Directions are 16-way indices; cardinals are N=0, E=4, S=8, W=12.
- Positions are tile-centered floats. A footprint anchored at layout top-left
  `(x, y)` with size `(w, h)` sits at Factorio position `(x + w/2, y + h/2)`.
- Force: `'player'` for our factory, `'neutral'` for resource ore tiles.

Not covered here: actually running the commands (needs FLE + Docker), reading
production stats back (see `get_production_lua`), or the base64/zlib encoding
of the blueprint dict. Plan.md §FLE integration covers the run-side.
"""
from __future__ import annotations

from mini_factorio.entities import MACHINES
from mini_factorio.layout import DIR_DELTA, OPPOSITE, Belt, Inserter, Layout, Machine, Resource, _machine_kind
from mini_factorio.recipes import GREEN_SCIENCE_ITEM, RECIPES

# Factorio 2.0 uses doubled direction indices (16-way for rails). Cardinal
# entities like belts / inserters / assemblers use these values.
FACTORIO_DIRECTION: dict[str, int] = {
    "north": 0,
    "east": 4,
    "south": 8,
    "west": 12,
}


def _factorio_inserter_direction(our_direction: str) -> int:
    """Factorio's `LuaInserter.direction` is the PICKUP direction, while our
    schema uses the DROP direction. Verified live via `pickup_position`
    read-back on a running server. So Factorio direction = opposite of ours."""
    return FACTORIO_DIRECTION[OPPOSITE[our_direction]]

# Resource tile amount — high enough that a 10-minute FLE run never depletes.
RESOURCE_ORE_AMOUNT = 100_000

# Power infra names.
SUBSTATION = "substation"                        # 2x2, 18-tile supply radius
POWER_SOURCE = "electric-energy-interface"       # 1x1, creative infinite power
FACTORIO_VERSION_INT = 281479275675649           # Factorio 2.0 blueprint version int


def _center(x: int, y: int, size: tuple[int, int]) -> tuple[float, float]:
    w, h = size
    return (x + w / 2, y + h / 2)


def _lua_position(pos: tuple[float, float]) -> str:
    return f"{{{pos[0]:g}, {pos[1]:g}}}"


def _create_entity_lua(
    name: str,
    pos: tuple[float, float],
    *,
    direction: int | None = None,
    recipe: str | None = None,
    force: str = "player",
    amount: int | None = None,
) -> str:
    parts = [f"name='{name}'", f"position={_lua_position(pos)}", f"force='{force}'"]
    if direction is not None:
        parts.append(f"direction={direction}")
    if amount is not None:
        parts.append(f"amount={amount}")
    base = "game.surfaces[1].create_entity{" + ", ".join(parts) + "}"
    if recipe is None:
        return base
    # Assemblers/furnaces need their recipe applied after creation.
    return f"local e = {base}; if e then e.set_recipe('{recipe}') end"


def _clear_surface_lua() -> str:
    return (
        "for _, e in pairs(game.surfaces[1].find_entities()) do "
        "if e.name ~= 'character' and e.name ~= 'player' then e.destroy() end "
        "end"
    )


SUBSTATION_SPACING = 16  # tiles between substation anchors; supply radius 18, wire reach 18.


def _power_grid_commands(layout: Layout) -> list[str]:
    """Cover the layout with a grid of substations + one infinite power source.

    Substations are placed every SUBSTATION_SPACING tiles so wires reach and
    supply areas overlap. Skips grid positions where the 2×2 footprint
    collides with existing entities. Also tries a small perturbation search
    to place any substation that would otherwise miss its grid slot.
    """
    w, h = layout.grid_size
    occupied = set(layout.occupied_tiles().keys())
    cmds: list[str] = []
    placed_positions: list[tuple[int, int]] = []

    def _try_place(cx: int, cy: int) -> bool:
        tiles = {(cx + dx, cy + dy) for dx in range(2) for dy in range(2)}
        if any(not (0 <= t[0] and 0 <= t[1]) for t in tiles):
            return False
        if tiles & occupied:
            return False
        cmds.append(_create_entity_lua(SUBSTATION, _center(cx, cy, (2, 2))))
        occupied.update(tiles)
        placed_positions.append((cx, cy))
        return True

    # Grid pattern covering the layout, plus a margin so edge machines are lit.
    ys = list(range(0, h + SUBSTATION_SPACING, SUBSTATION_SPACING))
    xs = list(range(0, w + SUBSTATION_SPACING, SUBSTATION_SPACING))
    for cy in ys:
        for cx in xs:
            if _try_place(cx, cy):
                continue
            # Small perturbation search: try neighbors within +/-3 tiles.
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    if _try_place(cx + dx, cy + dy):
                        break
                else:
                    continue
                break

    if not placed_positions:
        cmds.append("-- WARNING: could not place any substation")
        return cmds

    # One infinite power source (EEI) next to the first substation.
    sx, sy = placed_positions[0]
    for cx, cy in [(sx + 2, sy), (sx - 1, sy), (sx, sy + 2), (sx, sy - 1)]:
        if 0 <= cx and 0 <= cy and (cx, cy) not in occupied:
            eei_pos = _center(cx, cy, (1, 1))
            cmds.append(_create_entity_lua(POWER_SOURCE, eei_pos))
            cmds.append(
                f"local eei = game.surfaces[1].find_entity('{POWER_SOURCE}', "
                f"{_lua_position(eei_pos)}); "
                "if eei then eei.electric_buffer_size = 1e12; "
                "eei.power_production = 1e9 end"
            )
            occupied.add((cx, cy))
            break
    return cmds


def _infinity_chest_lua(pos: tuple[float, float], item: str) -> list[str]:
    """Create an infinity-chest at pos configured to spawn `item` at count 1000."""
    lua_pos = _lua_position(pos)
    return [
        _create_entity_lua("infinity-chest", pos),
        f"local c = game.surfaces[1].find_entity('infinity-chest', {lua_pos}); "
        f"if c then c.set_infinity_container_filter(1, "
        f"{{name='{item}', count=1000, mode='at-least', index=1}}) end",
    ]


def _machine_output_item(m: Machine) -> str | None:
    if "mining-drill" in m.type:
        return m.target_resource
    if m.recipe and m.recipe in RECIPES:
        products = RECIPES[m.recipe].products
        if len(products) == 1:
            return next(iter(products))
    return None


def _machine_input_feed_commands(layout: Layout) -> list[str]:
    """For each assembler/furnace with a recipe, feed every ingredient via an
    infinity-chest + inserter placed adjacent to the machine. Skips ingredients
    that would collide with existing entities or the map edge. This bypasses
    the layout's own input belts — used when blueprints omit miners/inputs.
    """
    machines_by_id = {m.id: m for m in layout.machines}
    tile_owner: dict[tuple[int, int], tuple[str, str]] = {}
    for m in layout.machines:
        for t in layout.machine_footprint(m):
            tile_owner[t] = ("machine", m.id)
    for b in layout.belts:
        for bt in b.tiles:
            tile_owner[(bt.x, bt.y)] = ("belt", b.id)
    for i in layout.inserters:
        tile_owner[(i.x, i.y)] = ("inserter", i.id)

    def _adj_tiles(m: Machine) -> list[tuple[int, int]]:
        w, h = MACHINES[m.type].size
        adj: list[tuple[int, int]] = []
        for dx in range(w):
            adj.append((m.x + dx, m.y - 1))
            adj.append((m.x + dx, m.y + h))
        for dy in range(h):
            adj.append((m.x - 1, m.y + dy))
            adj.append((m.x + w, m.y + dy))
        return adj

    # For each machine input inserter, determine what item it already delivers.
    # If the source item can be inferred (from a machine or previously-inferred
    # belt), we don't need to feed that ingredient again.
    already_fed: dict[str, set[str]] = {m.id: set() for m in layout.machines}
    for i in layout.inserters:
        dx, dy = DIR_DELTA[i.direction]
        drop = (i.x + dx, i.y + dy)
        snk = tile_owner.get(drop)
        if not snk or snk[0] != "machine":
            continue
        odx, ody = DIR_DELTA[OPPOSITE[i.direction]]
        pickup = (i.x + odx, i.y + ody)
        src = tile_owner.get(pickup)
        if src and src[0] == "machine":
            it = _machine_output_item(machines_by_id[src[1]])
            if it:
                already_fed[snk[1]].add(it)
        elif src and src[0] == "belt":
            for b in layout.belts:
                if b.id == src[1] and b.item != "unknown":
                    already_fed[snk[1]].add(b.item)
                    break

    cmds: list[str] = []
    occupied = set(tile_owner.keys())
    for m in layout.machines:
        if "mining-drill" in m.type:
            continue
        if not m.recipe or m.recipe not in RECIPES:
            continue
        ingredients = set(RECIPES[m.recipe].ingredients.keys())
        # Furnaces additionally need coal for fuel.
        if _machine_kind(m.type) == "furnace":
            ingredients.add("coal")
        missing = ingredients - already_fed[m.id]
        for item in missing:
            placed = False
            for ax, ay in _adj_tiles(m):
                # Need two free tiles: inserter at (ax,ay), chest at outward.
                ox, oy = ax - m.x, ay - m.y
                w, h = MACHINES[m.type].size
                # Determine outward direction from machine.
                if oy == -1:
                    step = (0, -1); insr_dir = "south"
                elif oy == h:
                    step = (0, 1); insr_dir = "north"
                elif ox == -1:
                    step = (-1, 0); insr_dir = "east"
                elif ox == w:
                    step = (1, 0); insr_dir = "west"
                else:
                    continue
                chest_pos = (ax + step[0], ay + step[1])
                if (ax, ay) in occupied or chest_pos in occupied:
                    continue
                # Inserter with our-direction = insr_dir (drop into machine).
                cmds.append(_create_entity_lua(
                    "inserter",
                    _center(ax, ay, (1, 1)),
                    direction=FACTORIO_DIRECTION[OPPOSITE[insr_dir]],
                ))
                cmds.extend(_infinity_chest_lua(_center(*chest_pos, (1, 1)), item))
                occupied.add((ax, ay))
                occupied.add(chest_pos)
                placed = True
                break
            if not placed:
                cmds.append(
                    f"-- WARNING: no room to feed {item!r} into machine {m.id!r} at ({m.x},{m.y})"
                )
    return cmds


def _machine_output_sink_commands(layout: Layout) -> list[str]:
    """For each machine whose output isn't extracted by an existing inserter
    (or the inserter picks up but its item leads nowhere useful), place a
    void-configured infinity-chest + inserter adjacent to remove products.
    Ensures measured production reflects steady-state, not a one-shot buffer fill.
    """
    machines_by_id = {m.id: m for m in layout.machines}
    tile_owner: dict[tuple[int, int], tuple[str, str]] = {}
    for m in layout.machines:
        for t in layout.machine_footprint(m):
            tile_owner[t] = ("machine", m.id)
    for b in layout.belts:
        for bt in b.tiles:
            tile_owner[(bt.x, bt.y)] = ("belt", b.id)
    for i in layout.inserters:
        tile_owner[(i.x, i.y)] = ("inserter", i.id)

    # Machines with at least one inserter picking from them.
    extracted: set[str] = set()
    for i in layout.inserters:
        odx, ody = DIR_DELTA[OPPOSITE[i.direction]]
        pickup = (i.x + odx, i.y + ody)
        src = tile_owner.get(pickup)
        if src and src[0] == "machine":
            extracted.add(src[1])

    def _adj_tiles(m: Machine) -> list[tuple[int, int]]:
        w, h = MACHINES[m.type].size
        adj: list[tuple[int, int]] = []
        for dx in range(w):
            adj.append((m.x + dx, m.y - 1))
            adj.append((m.x + dx, m.y + h))
        for dy in range(h):
            adj.append((m.x - 1, m.y + dy))
            adj.append((m.x + w, m.y + dy))
        return adj

    cmds: list[str] = []
    occupied = set(tile_owner.keys())
    for m in layout.machines:
        if "mining-drill" in m.type:
            continue
        out_item = _machine_output_item(m)
        if not out_item:
            continue
        # Always add void sink for green-science producers; existing extractors
        # in the blueprint may lead to dead-end belts that stall the output.
        # For other outputs, only add sink if nothing extracts from it.
        if out_item != GREEN_SCIENCE_ITEM and m.id in extracted:
            continue
        placed = False
        for ax, ay in _adj_tiles(m):
            ox, oy = ax - m.x, ay - m.y
            w, h = MACHINES[m.type].size
            if oy == -1:
                step = (0, -1); insr_dir = "north"
            elif oy == h:
                step = (0, 1); insr_dir = "south"
            elif ox == -1:
                step = (-1, 0); insr_dir = "west"
            elif ox == w:
                step = (1, 0); insr_dir = "east"
            else:
                continue
            chest_pos = (ax + step[0], ay + step[1])
            if (ax, ay) in occupied or chest_pos in occupied:
                continue
            # Inserter our_direction = insr_dir (drops outward, away from machine).
            cmds.append(_create_entity_lua(
                "inserter",
                _center(ax, ay, (1, 1)),
                direction=FACTORIO_DIRECTION[OPPOSITE[insr_dir]],
            ))
            chest_center = _center(*chest_pos, (1, 1))
            cmds.append(_create_entity_lua("infinity-chest", chest_center))
            cmds.append(
                f"local c = game.surfaces[1].find_entity('infinity-chest', "
                f"{_lua_position(chest_center)}); "
                "if c then c.remove_unfiltered_items = true end"
            )
            occupied.add((ax, ay))
            occupied.add(chest_pos)
            placed = True
            break
        if not placed:
            cmds.append(f"-- WARNING: no room for output sink at machine {m.id!r}")
    return cmds


def _dead_end_sink_commands(layout: Layout) -> list[str]:
    """For each belt with no consumer inserter, place an inserter + steel-chest
    two tiles downstream of the belt tip. Mirrors the sim, which treats such
    belts as feeding an implicit sink. Chests may be placed outside the layout
    grid (Factorio's surface is unbounded).
    """
    inserter_pickup_tiles: set[tuple[int, int]] = set()
    for i in layout.inserters:
        odx, ody = DIR_DELTA[OPPOSITE[i.direction]]
        inserter_pickup_tiles.add((i.x + odx, i.y + ody))

    occupied = set(layout.occupied_tiles().keys())
    cmds: list[str] = []
    for b in layout.belts:
        if any((t.x, t.y) in inserter_pickup_tiles for t in b.tiles):
            continue
        tip = b.tiles[-1]
        dx, dy = DIR_DELTA[tip.direction]
        insr_pos = (tip.x + dx, tip.y + dy)
        chest_pos = (tip.x + 2 * dx, tip.y + 2 * dy)
        if insr_pos in occupied or chest_pos in occupied:
            cmds.append(
                f"-- WARNING: dead-end sink for belt {b.id!r} not placed; "
                f"tiles {insr_pos} or {chest_pos} occupied"
            )
            continue
        inserter_dir_factorio = FACTORIO_DIRECTION[OPPOSITE[tip.direction]]
        cmds.append(_create_entity_lua(
            "inserter",
            _center(insr_pos[0], insr_pos[1], (1, 1)),
            direction=inserter_dir_factorio,
        ))
        cmds.append(_create_entity_lua(
            "steel-chest",
            _center(chest_pos[0], chest_pos[1], (1, 1)),
        ))
        occupied.add(insr_pos)
        occupied.add(chest_pos)
    return cmds


def _resource_commands(res: Resource) -> list[str]:
    cmds: list[str] = []
    for dx in range(res.size):
        for dy in range(res.size):
            pos = _center(res.x + dx, res.y + dy, (1, 1))
            cmds.append(_create_entity_lua(
                res.type, pos, force="neutral", amount=RESOURCE_ORE_AMOUNT,
            ))
    return cmds


def _machine_command(m: Machine) -> str:
    spec = MACHINES[m.type]
    pos = _center(m.x, m.y, spec.size)
    # Only assembling machines accept set_recipe(). Furnaces smelt whatever ore is
    # inserted into them (no explicit recipe), and miners have no recipe concept.
    recipe = m.recipe if m.type.startswith("assembling-machine-") else None
    # For miners, `direction` is the drop direction and Factorio's convention
    # matches ours (no inversion, unlike inserters). For furnaces/assemblers
    # the direction is cosmetic — Factorio inserters can access any side.
    direction = FACTORIO_DIRECTION[m.direction] if m.type == "electric-mining-drill" else None
    return _create_entity_lua(m.type, pos, recipe=recipe, direction=direction)


def _inserter_command(i: Inserter) -> str:
    pos = _center(i.x, i.y, (1, 1))
    # `i.type` carries the tier (inserter / fast-inserter / stack-inserter).
    return _create_entity_lua(i.type, pos, direction=_factorio_inserter_direction(i.direction))


def _belt_commands(b: Belt) -> list[str]:
    # `b.type` carries the belt tier (transport-belt / fast- / express-).
    return [
        _create_entity_lua(
            b.type,
            _center(t.x, t.y, (1, 1)),
            direction=FACTORIO_DIRECTION[t.direction],
        )
        for t in b.tiles
    ]


def layout_to_lua_commands(layout: Layout, *, add_power: bool = True,
                            feed_missing_inputs: bool = False) -> list[str]:
    """Return the ordered Lua commands to build this layout on FLE surface 1.

    Order: clear surface → power infra → resources → machines → inserters → belts.
    Each list element is a single Lua statement (may include semicolons).
    """
    cmds: list[str] = [_clear_surface_lua()]
    if add_power:
        cmds.extend(_power_grid_commands(layout))
    for r in layout.resources:
        cmds.extend(_resource_commands(r))
    for m in layout.machines:
        cmds.append(_machine_command(m))
    for i in layout.inserters:
        cmds.append(_inserter_command(i))
    for b in layout.belts:
        cmds.extend(_belt_commands(b))
    cmds.extend(_dead_end_sink_commands(layout))
    if feed_missing_inputs:
        cmds.extend(_machine_input_feed_commands(layout))
        cmds.extend(_machine_output_sink_commands(layout))
    return cmds


def read_production_count_lua(item_id: str) -> str:
    """Return a Lua one-liner that rcon-prints the total-ever-produced count.

    Caller pattern (in fle_driver): capture count before the timing window,
    sleep, capture count after, subtract in Python, divide by wall time.

    Factorio 2.0 renamed the stats API: `force.item_production_statistics`
    (attribute) → `force.get_item_production_statistics(surface)` (method).
    See FLE `fle/env/tools/admin/get_production_stats/server.lua`.
    """
    return (
        f"local stats = game.forces['player']"
        f".get_item_production_statistics(game.surfaces[1]); "
        f"rcon.print(stats.input_counts['{item_id}'] or 0)"
    )


def set_game_speed_lua(speed: float) -> str:
    return f"game.speed = {speed:g}"


# --------- Blueprint dict output ---------


def layout_to_blueprint_dict(layout: Layout) -> dict:
    """Return the layout in Factorio's blueprint schema (dict form).

    See https://wiki.factorio.com/Blueprint_string_format. The full blueprint
    string is `'0' + base64(zlib(json(this dict)))`; we return only the dict so
    tests can inspect it without binary encoding.
    """
    entities: list[dict] = []
    idx = 1

    for m in layout.machines:
        spec = MACHINES[m.type]
        e: dict = {
            "entity_number": idx,
            "name": m.type,
            "position": {"x": m.x + spec.size[0] / 2, "y": m.y + spec.size[1] / 2},
        }
        if m.recipe:
            e["recipe"] = m.recipe
        entities.append(e)
        idx += 1

    for ins in layout.inserters:
        entities.append({
            "entity_number": idx,
            "name": "inserter",
            "position": {"x": ins.x + 0.5, "y": ins.y + 0.5},
            "direction": _factorio_inserter_direction(ins.direction),
        })
        idx += 1

    for b in layout.belts:
        for t in b.tiles:
            entities.append({
                "entity_number": idx,
                "name": "transport-belt",
                "position": {"x": t.x + 0.5, "y": t.y + 0.5},
                "direction": FACTORIO_DIRECTION[t.direction],
            })
            idx += 1

    return {
        "blueprint": {
            "icons": [
                {"signal": {"type": "item", "name": "logistic-science-pack"}, "index": 1}
            ],
            "entities": entities,
            "item": "blueprint",
            "version": FACTORIO_VERSION_INT,
        }
    }
