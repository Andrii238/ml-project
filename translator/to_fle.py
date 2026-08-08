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
from mini_factorio.layout import DIR_DELTA, OPPOSITE, Belt, Inserter, Layout, Machine, Resource

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


def _power_grid_commands(layout: Layout) -> list[str]:
    """Place a substation + EEI so electric machines can actually run in FLE."""
    w, h = layout.grid_size
    occupied = set(layout.occupied_tiles().keys())
    cmds: list[str] = []

    # Substation (2×2): try common spots in preference order.
    sub_pos: tuple[int, int] | None = None
    for cx, cy in [
        (w // 2 - 1, h // 2 - 1),
        (0, 0),
        (w - 2, h - 2),
        (0, h - 2),
        (w - 2, 0),
    ]:
        if cx < 0 or cy < 0 or cx + 2 > w or cy + 2 > h:
            continue
        tiles = {(cx + dx, cy + dy) for dx in range(2) for dy in range(2)}
        if not tiles & occupied:
            sub_pos = (cx, cy)
            occupied |= tiles
            break
    if sub_pos is None:
        cmds.append("-- WARNING: no free 2x2 for substation; FLE run will lack power")
        return cmds
    cmds.append(_create_entity_lua(SUBSTATION, _center(*sub_pos, (2, 2))))

    # Infinite power source (1×1) somewhere adjacent to the substation.
    candidates = [
        (sub_pos[0] + 2, sub_pos[1]),
        (sub_pos[0] - 1, sub_pos[1]),
        (sub_pos[0], sub_pos[1] + 2),
        (sub_pos[0], sub_pos[1] - 1),
    ]
    for cx, cy in candidates:
        if 0 <= cx < w and 0 <= cy < h and (cx, cy) not in occupied:
            eei_pos = _center(cx, cy, (1, 1))
            cmds.append(_create_entity_lua(POWER_SOURCE, eei_pos))
            cmds.append(
                f"local eei = game.surfaces[1].find_entity('{POWER_SOURCE}', "
                f"{_lua_position(eei_pos)}); "
                "if eei then eei.electric_buffer_size = 1e12; "
                "eei.power_production = 1e9 end"
            )
            occupied.add((cx, cy))
            return cmds
    cmds.append("-- WARNING: no free 1x1 next to substation for EEI")
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
    recipe = m.recipe if m.type == "assembling-machine-1" else None
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


def layout_to_lua_commands(layout: Layout, *, add_power: bool = True) -> list[str]:
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
