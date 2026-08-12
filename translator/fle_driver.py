"""Drive a live Factorio server (via RCON) to build and measure a translated
layout, then compare against our sim.

Uses `factorio-rcon-py.RCONClient` directly. Every command is wrapped in
Factorio's `/sc` (silent-command) prefix; read-back values use `rcon.print()`
inside the Lua chunk.

Expected server:
- Factorio 2.0.x (matches factoriolab data version we use).
- RCON on port 27000, password 'factorio' (FLE cluster defaults).
- Container started via `fle cluster start` or manually.

Typical usage:
    from mini_factorio.random_layouts import empty_episode
    from translator.to_fle import translate
    from translator.fle_driver import validate_and_measure

    lay = empty_episode(seed=42)
    tr = translate(lay)
    result = validate_and_measure(tr, host='localhost', port=27000, password='factorio')
    print(result)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterable

try:
    import factorio_rcon
except ImportError:  # pragma: no cover
    factorio_rcon = None  # type: ignore

from .to_fle import TranslatedEntity, TranslationResult


# --------------------------------------------------------------- config

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 27000
DEFAULT_PASSWORD = "factorio"

# Origin offset — placement is centered around this map tile. We stay away
# from (0, 0) because the starting character sits near the origin.
ORIGIN_X = 100
ORIGIN_Y = 100


# --------------------------------------------------------------- Lua helpers

def _lua_str(s: str) -> str:
    """Quote a Python string for embedding in Lua source. Handles quotes and
    backslashes only — good enough for entity names and simple values."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _lua_position(x: float, y: float) -> str:
    return "{x=" + f"{x:.4f}" + ",y=" + f"{y:.4f}" + "}"


def _entity_to_lua(e: TranslatedEntity, offset_x: float, offset_y: float) -> str:
    """Emit a Lua `create_entity{...}` call for a single translated entity."""
    x = e.position["x"] + offset_x
    y = e.position["y"] + offset_y
    parts = [
        f"name={_lua_str(e.name)}",
        f"position={_lua_position(x, y)}",
        'force="player"',
    ]
    if e.direction is not None:
        parts.append(f"direction=defines.direction.{_dir_name(e.direction)}")
    if e.type is not None:  # underground-belt type: "input" or "output"
        parts.append(f"type={_lua_str(e.type)}")
    if e.recipe is not None:
        parts.append(f"recipe={_lua_str(e.recipe)}")
    return "surface.create_entity{" + ",".join(parts) + "}"


def _dir_name(d: int) -> str:
    return {0: "north", 4: "east", 8: "south", 12: "west"}.get(d, "north")


# --------------------------------------------------------------- driver

@dataclass
class BuildResult:
    build_success: bool
    entities_requested: int
    entities_placed: int
    build_errors: list[str] = field(default_factory=list)


@dataclass
class MeasureResult:
    fle_science_rate: float          # green-science packs / sec measured live
    fle_ticks_elapsed: int
    fle_seconds_elapsed: float


@dataclass
class CrossCheckResult:
    build: BuildResult
    measure: MeasureResult | None
    sim_science_rate: float | None
    error: float | None              # (sim - fle) / max(fle, 1e-6)

    def to_dict(self) -> dict:
        return {
            "build_success": self.build.build_success,
            "entities_requested": self.build.entities_requested,
            "entities_placed": self.build.entities_placed,
            "build_errors": self.build.build_errors,
            "fle_rate": self.measure.fle_science_rate if self.measure else None,
            "fle_ticks": self.measure.fle_ticks_elapsed if self.measure else None,
            "fle_seconds": self.measure.fle_seconds_elapsed if self.measure else None,
            "sim_rate": self.sim_science_rate,
            "rel_error": self.error,
        }


def _connect(host: str, port: int, password: str):
    if factorio_rcon is None:
        raise RuntimeError("factorio-rcon-py not installed; pip install factorio-rcon-py")
    return factorio_rcon.RCONClient(host, port, password)


def _clear_and_research(rc) -> None:
    rc.send_command(
        '/sc local s = game.surfaces["nauvis"]; '
        'for _, e in pairs(s.find_entities()) do '
        '  if e.name ~= "character" then e.destroy() end '
        'end; rcon.print("cleared")'
    )
    rc.send_command('/sc game.forces["player"].research_all_technologies(); '
                     'rcon.print("researched")')


def _place_infinite_power(rc, origin_x: int, origin_y: int,
                            grid_w: int, grid_h: int) -> None:
    """Our sim skips electricity. Add an infinite-power source + enough
    substations to cover the whole translated grid.

    - `electric-energy-interface`: 1x1, infinite production. Place one at the
      origin corner (offset by a few tiles so it doesn't overlap layout entities).
    - `substation`: 2x2, 18-tile supply area, propagates power.

    We drop a substation grid at 14-tile intervals so overlaps guarantee full coverage.
    """
    # EEI at (origin - 3, origin - 3) so it's well outside the layout footprint.
    eei_x = origin_x - 3
    eei_y = origin_y - 3
    rc.send_command(
        f'/sc local eei = game.surfaces["nauvis"].create_entity{{'
        f'name="electric-energy-interface", position={{x={eei_x + 0.5},y={eei_y + 0.5}}}, '
        f'force="player"}}; '
        f'eei.electric_buffer_size = 1e9; eei.power_production = 1e9; '
        f'rcon.print("eei ok")')
    # Grid of substations.
    step = 14  # supply area of substation is 18x18, so 14-tile spacing gives overlap.
    for x in range(origin_x - 2, origin_x + grid_w + 2, step):
        for y in range(origin_y - 2, origin_y + grid_h + 2, step):
            rc.send_command(
                f'/sc local ok = pcall(function() '
                f'game.surfaces["nauvis"].create_entity{{'
                f'name="substation", position={{x={x + 1.0},y={y + 1.0}}}, '
                f'force="player"}} end); rcon.print(tostring(ok))'
            )


def build_layout(rc, tr: TranslationResult,
                 origin_x: int = ORIGIN_X, origin_y: int = ORIGIN_Y) -> BuildResult:
    """Clear the surface, then place every entity in `tr` at (position + origin).
    Returns per-entity success counts and any error strings from Factorio."""
    _clear_and_research(rc)
    _place_infinite_power(rc, origin_x, origin_y,
                            tr.grid_size[0], tr.grid_size[1])
    requested = len(tr.entities)
    placed = 0
    errors: list[str] = []

    # Place entities one at a time so we get per-entity error messages.
    # Batching all into one /sc chunk would surface only the first error.
    for e in tr.entities:
        lua = _entity_to_lua(e, origin_x, origin_y)
        chunk = f'/sc local surface = game.surfaces["nauvis"]; '
        chunk += 'local ok, err = pcall(function() '
        chunk += f'local ent = {lua}; '
        # Apply infinity_settings filters immediately after creation (they
        # are not accepted as a create_entity kwarg in Factorio 2.0).
        if e.infinity_settings and e.infinity_settings.get("filters"):
            for i, f in enumerate(e.infinity_settings["filters"], start=1):
                chunk += (f'ent.set_infinity_container_filter({i}, '
                          f'{{name={_lua_str(f["name"])}, '
                          f'count={f["count"]}, '
                          f'mode={_lua_str(f.get("mode", "exactly"))}, '
                          f'index={i}}}); ')
            chunk += ('ent.remove_unfiltered_items = '
                      + str(bool(e.infinity_settings.get("remove_unfiltered_items", False))).lower() + '; ')
        chunk += 'rcon.print(ent ~= nil and "OK" or "NIL") end); '
        chunk += 'if not ok then rcon.print("ERR:" .. tostring(err)) end'
        reply = rc.send_command(chunk).strip()
        if reply == "OK":
            placed += 1
        elif reply == "NIL":
            errors.append(f"{e.name}@{e.position}: create_entity returned nil "
                          f"(likely position blocked or invalid)")
        elif reply.startswith("ERR:"):
            errors.append(f"{e.name}@{e.position}: {reply[4:]}")
        else:
            errors.append(f"{e.name}@{e.position}: unexpected reply {reply!r}")
    return BuildResult(
        build_success=(placed == requested and not errors),
        entities_requested=requested,
        entities_placed=placed,
        build_errors=errors,
    )


def measure_science_rate(rc, duration_seconds: float = 30.0,
                          game_speed: int = 20) -> MeasureResult:
    """Un-throttled measurement — chest rate is NOT enforced.

    Use `measure_science_rate_throttled` for rate-controlled experiments."""
    rc.send_command(f'/sc game.speed = {game_speed}; rcon.print("speed set")')

    def _tick() -> int:
        r = rc.send_command('/sc rcon.print(game.tick)').strip()
        return int(r)

    def _science_count() -> int:
        r = rc.send_command(
            '/sc local st = game.forces["player"].get_item_production_statistics(game.surfaces["nauvis"]); '
            'rcon.print(st.input_counts["logistic-science-pack"] or 0)'
        ).strip()
        return int(r)

    t0 = _tick()
    sci0 = _science_count()
    time.sleep(duration_seconds)
    t1 = _tick()
    sci1 = _science_count()
    rc.send_command('/sc game.speed = 1; rcon.print("speed reset")')

    ticks = t1 - t0
    seconds = ticks / 60.0
    delta = sci1 - sci0
    rate = delta / max(seconds, 1e-9)
    return MeasureResult(
        fle_science_rate=rate,
        fle_ticks_elapsed=ticks,
        fle_seconds_elapsed=seconds,
    )


def measure_science_rate_throttled(rc, chest_map: dict[tuple[float, float], str],
                                     chest_rates: dict[str, float],
                                     duration_seconds: float = 30.0) -> MeasureResult:
    """Throttle input chests to enforce the sim's per-second emission rate.

    `chest_map` maps translated (x, y) tile position (top-left, not tile-center)
    to chest kind. `chest_rates` maps chest kind → items/sec.

    Every wall-second, the driver tops up each input chest so the average
    emission matches the target rate. Fractional rates use an accumulator.
    Runs at game.speed=1 so 1 wall-second = 1 game-second.
    """
    kind_to_item = {"input-belts": "transport-belt", "input-inserters": "inserter"}

    # Map position -> (rate, item) for chests we'll throttle.
    pos_to_rate: dict[tuple[float, float], float] = {}
    pos_to_item: dict[tuple[float, float], str] = {}
    for (px, py), kind in chest_map.items():
        if kind not in chest_rates:
            continue
        item = kind_to_item.get(kind)
        if item is None:
            continue
        pos_to_rate[(px, py)] = chest_rates[kind]
        pos_to_item[(px, py)] = item

    accum: dict[tuple[float, float], float] = {p: 0.0 for p in pos_to_rate}
    for (px, py) in pos_to_rate:
        rc.send_command(
            f'/sc local es = game.surfaces["nauvis"].find_entities_filtered'
            f'{{name="infinity-chest",position={{x={px},y={py}}},radius=0.5}}; '
            f'if es[1] then es[1].get_inventory(defines.inventory.chest).clear() end; '
            f'rcon.print("ok")')

    rc.send_command('/sc game.speed = 1; rcon.print("speed set")')

    def _tick() -> int:
        return int(rc.send_command('/sc rcon.print(game.tick)').strip())

    def _science() -> int:
        return int(rc.send_command(
            '/sc local st = game.forces["player"].get_item_production_statistics(game.surfaces["nauvis"]); '
            'rcon.print(st.input_counts["logistic-science-pack"] or 0)').strip())

    tick0 = _tick()
    sci0 = _science()

    end_at = time.time() + duration_seconds
    while time.time() < end_at:
        for (px, py), rate in pos_to_rate.items():
            accum[(px, py)] += rate
            items_to_add = int(accum[(px, py)])
            if items_to_add <= 0:
                continue
            accum[(px, py)] -= items_to_add
            item = pos_to_item[(px, py)]
            rc.send_command(
                f'/sc local es = game.surfaces["nauvis"].find_entities_filtered'
                f'{{name="infinity-chest",position={{x={px},y={py}}},radius=0.5}}; '
                f'if es[1] then es[1].insert{{name="{item}", count={items_to_add}}} end; '
                f'rcon.print("ok")')
        time.sleep(1.0)

    tick1 = _tick()
    sci1 = _science()

    ticks = tick1 - tick0
    seconds = ticks / 60.0
    delta = sci1 - sci0
    rate = delta / max(seconds, 1e-9)
    return MeasureResult(
        fle_science_rate=rate,
        fle_ticks_elapsed=ticks,
        fle_seconds_elapsed=seconds,
    )


def validate_and_measure(tr: TranslationResult, *,
                          host: str = DEFAULT_HOST,
                          port: int = DEFAULT_PORT,
                          password: str = DEFAULT_PASSWORD,
                          duration_seconds: float = 30.0,
                          game_speed: int = 20,
                          sim_rate: float | None = None,
                          chest_rates: dict[str, float] | None = None,
                          chest_map: dict[tuple[float, float], str] | None = None,
                          origin_x: int = ORIGIN_X,
                          origin_y: int = ORIGIN_Y) -> CrossCheckResult:
    """If `chest_rates` + `chest_map` are provided, uses the throttled
    measurement loop that enforces the sim's per-second chest emission rate.
    `chest_map` maps chest tile-center (x, y) in the SAME coordinate frame as
    `tr.entities[*].position` (i.e., NOT offset by origin) to kind."""
    rc = _connect(host, port, password)
    build = build_layout(rc, tr, origin_x=origin_x, origin_y=origin_y)
    measure = None
    if build.build_success:
        if chest_rates and chest_map:
            # Offset chest_map into the placement coordinate system.
            offset_map = {(cx + origin_x, cy + origin_y): kind
                           for (cx, cy), kind in chest_map.items()}
            measure = measure_science_rate_throttled(
                rc, chest_map=offset_map, chest_rates=chest_rates,
                duration_seconds=duration_seconds)
        else:
            measure = measure_science_rate(rc, duration_seconds=duration_seconds,
                                            game_speed=game_speed)
    rel_error = None
    if measure is not None and sim_rate is not None:
        rel_error = (sim_rate - measure.fle_science_rate) / max(
            abs(measure.fle_science_rate), 1e-6)
    return CrossCheckResult(build=build, measure=measure,
                             sim_science_rate=sim_rate, error=rel_error)


def chest_map_from_layout(layout) -> dict[tuple[float, float], str]:
    """Build a `chest_map` (tile-center → kind) for a Layout, suitable for
    passing to `validate_and_measure(chest_map=...)`."""
    return {(c.x + 0.5, c.y + 0.5): c.kind for c in layout.chests}


def smoke_test(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                password: str = DEFAULT_PASSWORD) -> str:
    """Minimal reachability check. Places a single assembler, verifies it
    exists, clears it. Returns a status string."""
    rc = _connect(host, port, password)
    _clear_and_research(rc)
    rc.send_command('/sc local a = game.surfaces["nauvis"].create_entity{'
                     'name="assembling-machine-1", position={x=100.5,y=100.5}, '
                     'force="player", recipe="logistic-science-pack"}; '
                     'rcon.print(tostring(a ~= nil))')
    n = int(rc.send_command(
        '/sc local e = game.surfaces["nauvis"].find_entities_filtered{'
        'name="assembling-machine-1"}; rcon.print(#e)').strip())
    rc.send_command(
        '/sc for _, e in pairs(game.surfaces["nauvis"].find_entities()) do '
        'if e.name ~= "character" then e.destroy() end end; rcon.print("ok")')
    if n != 1:
        return f"FAIL: expected 1 assembler after placement, found {n}"
    return f"OK: FLE reachable on {host}:{port}, placed and cleared 1 assembler"
