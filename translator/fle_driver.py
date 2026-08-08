"""Drive FLE (Factorio Learning Environment) from Python for cross-validation.

Three entry points:

- `smoke_test()` — minimal connectivity check. Places one furnace via RCON and
  reads it back. Use this first after `fle cluster start`.
- `validate_and_measure(layout)` — builds one Layout in FLE, times its
  green-science production, returns `FLEValidation` with build status + measured
  rate.
- `cross_check(entries)` — runs a batch, computes Pearson r + MAPE against our
  simulator, checks plan.md §FLE integration ship gates.

**Wire model.** We use the raw `factorio-rcon-py` client rather than FLE's
`FactorioInstance`. FLE's `FactorioInstance.eval()` runs Python (their agent
REPL), not raw Lua. We already emit Lua, so it's simpler to talk RCON directly.
`factorio-rcon-py` is a FLE dep, so `uv sync --extra fle` gives us both.

**Server management.** The Factorio dedicated server runs in a Docker container
managed by FLE's `fle cluster start` CLI. Default RCON port is 27000
(FLE's `START_RCON_PORT`), password `'factorio'`. See translator/FLE_NOTES.md.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from mini_factorio.layout import Layout
from mini_factorio.recipes import GREEN_SCIENCE_ITEM

from .to_fle import layout_to_lua_commands, read_production_count_lua, set_game_speed_lua

DEFAULT_ADDRESS = "localhost"
DEFAULT_RCON_PORT = 27000        # FLE's START_RCON_PORT
DEFAULT_RCON_PASSWORD = "factorio"  # FLE's RCON_PASSWORD
DEFAULT_MEASUREMENT_SECONDS = 120.0
DEFAULT_WARMUP_SECONDS = 60.0    # skip this much game-time before starting the rate window
DEFAULT_GAME_SPEED = 100         # so 60 in-game seconds runs in 0.6 wall-time seconds


@dataclass
class FLEValidation:
    """Outcome of building + measuring one layout in FLE."""
    layout_id: str
    build_ok: bool
    build_errors: list[str] = field(default_factory=list)
    fle_rate: float = 0.0
    sim_rate: float | None = None

    def to_dict(self) -> dict:
        return {
            "layout_id": self.layout_id,
            "build_ok": self.build_ok,
            "build_errors": self.build_errors,
            "fle_rate": self.fle_rate,
            "sim_rate": self.sim_rate,
        }


@dataclass
class CrossCheckReport:
    per_layout: list[FLEValidation]
    build_success_rate: float
    pearson_r: float | None
    mape: float | None
    ship_gates: dict[str, bool]

    def to_dict(self) -> dict:
        return {
            "build_success_rate": self.build_success_rate,
            "pearson_r": self.pearson_r,
            "mape": self.mape,
            "ship_gates": self.ship_gates,
            "per_layout": [v.to_dict() for v in self.per_layout],
        }


# ------------------------- RCON plumbing (lazy import) -------------------------


def _import_rcon():
    try:
        from factorio_rcon import RCONClient
    except ImportError as e:
        raise ImportError(
            "factorio-rcon-py not installed. Install FLE with "
            "`uv sync --extra fle` (pulls factorio-rcon-py as a transitive dep) "
            "and start the Factorio server with `fle cluster start`. "
            "See translator/FLE_NOTES.md."
        ) from e
    return RCONClient


def _connect(
    address: str = DEFAULT_ADDRESS,
    tcp_port: int = DEFAULT_RCON_PORT,
    password: str = DEFAULT_RCON_PASSWORD,
) -> Any:
    RCONClient = _import_rcon()
    client = RCONClient(address, tcp_port, password)
    client.connect()
    return client


def _sc(client: Any, lua: str) -> str:
    """Send one Lua chunk via Factorio's /sc (silent command) RCON call.

    Returns the RCON reply string (contents of any `rcon.print(...)` calls in
    the chunk, concatenated). Empty string for fire-and-forget commands.
    Raises any exception raised by the RCON client (usually connection loss).
    """
    return client.send_command("/sc " + lua)


def _research_all(client: Any) -> None:
    """Unlock every recipe/technology so 'locked' recipes (e.g. logistic-science-pack) work."""
    _sc(client, (
        "for _, tech in pairs(game.forces['player'].technologies) do "
        "tech.researched = true end"
    ))


# ------------------------- Smoke test -------------------------


def smoke_test(
    address: str = DEFAULT_ADDRESS,
    tcp_port: int = DEFAULT_RCON_PORT,
    password: str = DEFAULT_RCON_PASSWORD,
) -> str:
    """Minimal end-to-end check. Returns a short human-readable status string.

    Sequence: connect → clear surface → place one stone-furnace → count via
    rcon.print → clear surface. Raises on any failure with a diagnostic message.
    """
    client = _connect(address, tcp_port, password)
    _sc(client, (
        "for _, e in pairs(game.surfaces[1].find_entities()) do "
        "if e.name ~= 'character' and e.name ~= 'player' then e.destroy() end end"
    ))
    _sc(client, (
        "game.surfaces[1].create_entity{name='stone-furnace', "
        "position={5, 5}, force='player'}"
    ))
    reply = _sc(client, (
        "rcon.print(#game.surfaces[1].find_entities_filtered{name='stone-furnace'})"
    ))
    _sc(client, (
        "for _, e in pairs(game.surfaces[1].find_entities()) do "
        "if e.name ~= 'character' and e.name ~= 'player' then e.destroy() end end"
    ))
    try:
        count = int(reply.strip())
    except (AttributeError, ValueError) as e:
        raise RuntimeError(
            f"unexpected RCON reply while reading furnace count: {reply!r}"
        ) from e
    if count != 1:
        raise RuntimeError(f"expected 1 furnace after placement, got {count}")
    return f"FLE reachable on {address}:{tcp_port}; test furnace placed and cleared"


# ------------------------- Single-layout validation -------------------------


def validate_and_measure(
    layout: Layout,
    *,
    layout_id: str = "unnamed",
    sim_rate: float | None = None,
    target_item: str = GREEN_SCIENCE_ITEM,
    measurement_seconds: float = DEFAULT_MEASUREMENT_SECONDS,
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
    game_speed: int = DEFAULT_GAME_SPEED,
    address: str = DEFAULT_ADDRESS,
    tcp_port: int = DEFAULT_RCON_PORT,
    password: str = DEFAULT_RCON_PASSWORD,
    add_power: bool = True,
    unlock_technologies: bool = True,
) -> FLEValidation:
    """Build the layout in FLE, run for `measurement_seconds` game-seconds,
    read green-science production rate.

    Steps:
        1. Connect over RCON.
        2. Unlock all technologies (once per call — needed for locked recipes).
        3. Send each Lua command from `layout_to_lua_commands()` in order.
           Any command that raises → recorded in build_errors, others still run.
        4. If any command errored, return with `build_ok=False` (don't trust the
           production number from a half-built layout).
        5. Set `game.speed = game_speed`, snapshot science count, sleep
           `measurement_seconds / game_speed` wall-time, snapshot again, return
           items/sec.
    """
    client = _connect(address, tcp_port, password)
    result = FLEValidation(layout_id=layout_id, build_ok=True, sim_rate=sim_rate)

    if unlock_technologies:
        try:
            _research_all(client)
        except Exception as e:
            result.build_errors.append(f"research_all: {e}")

    for i, cmd in enumerate(layout_to_lua_commands(layout, add_power=add_power)):
        try:
            reply = _sc(client, cmd)
            # Factorio returns a non-empty reply if the /sc raised inside Lua.
            # `rcon.print` output is fine; error output usually contains 'Error:'
            # or 'error running'.
            if reply and ("Error:" in reply or "error running" in reply):
                result.build_ok = False
                result.build_errors.append(f"cmd[{i}]: {reply.strip()}")
        except Exception as e:
            result.build_ok = False
            result.build_errors.append(f"cmd[{i}]: {e}")

    if not result.build_ok:
        return result

    # Measure with a warmup phase to eliminate transients (furnace/belt fill,
    # assembler output buffer fill). The rate window starts AFTER warmup.
    # Container CPU-limits game speed on Apple Silicon (~22× at game.speed=100),
    # so we use game.tick deltas — not wall time — to compute the rate.
    read_cmd = read_production_count_lua(target_item)
    _sc(client, set_game_speed_lua(game_speed))

    def _wait_ticks(target_delta_seconds: float) -> None:
        """Sleep until at least `target_delta_seconds` game-seconds have elapsed
        since the current game.tick. Cap wall time so we don't hang forever."""
        try:
            t0 = int(_sc(client, "rcon.print(game.tick)").strip())
        except (AttributeError, ValueError):
            return
        target = t0 + int(target_delta_seconds * 60)
        wall_deadline = time.time() + target_delta_seconds * 2.0 + 5.0
        while time.time() < wall_deadline:
            time.sleep(0.1)
            try:
                cur = int(_sc(client, "rcon.print(game.tick)").strip())
            except (AttributeError, ValueError):
                continue
            if cur >= target:
                return

    # Phase 1: warmup. Let the layout reach steady state.
    if warmup_seconds > 0:
        _wait_ticks(warmup_seconds)

    # Phase 2: measurement window. Snapshot count+tick, wait, snapshot again.
    try:
        tick_before = int(_sc(client, "rcon.print(game.tick)").strip())
        before = float(_sc(client, read_cmd).strip())
    except (AttributeError, ValueError) as e:
        result.build_ok = False
        result.build_errors.append(f"pre-count read: {e}")
        return result
    _wait_ticks(measurement_seconds)
    try:
        tick_after = int(_sc(client, "rcon.print(game.tick)").strip())
        after = float(_sc(client, read_cmd).strip())
    except (AttributeError, ValueError) as e:
        result.build_ok = False
        result.build_errors.append(f"post-count read: {e}")
        return result
    game_seconds_elapsed = (tick_after - tick_before) / 60.0
    if game_seconds_elapsed <= 0:
        result.build_ok = False
        result.build_errors.append("no game time elapsed during measurement")
        return result
    result.fle_rate = (after - before) / game_seconds_elapsed
    return result


# ------------------------- Batch cross-check -------------------------


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def _mape(ys_pred: list[float], ys_true: list[float]) -> float | None:
    pairs = [(p, t) for p, t in zip(ys_pred, ys_true) if t > 0]
    if not pairs:
        return None
    return sum(abs(p - t) / t for p, t in pairs) / len(pairs)


def summarize(results: list[FLEValidation]) -> CrossCheckReport:
    build_ok = [r for r in results if r.build_ok]
    build_success_rate = len(build_ok) / len(results) if results else 0.0
    paired = [r for r in build_ok if r.sim_rate is not None]
    sim_rates = [r.sim_rate for r in paired]
    fle_rates = [r.fle_rate for r in paired]
    r = _pearson_r(sim_rates, fle_rates)
    m = _mape(sim_rates, fle_rates)
    gates = {
        "build_success_100pct": build_success_rate >= 1.0,
        "pearson_r_ge_0_9": r is not None and r >= 0.9,
        "mape_le_0_20": m is not None and m <= 0.20,
    }
    return CrossCheckReport(
        per_layout=results,
        build_success_rate=build_success_rate,
        pearson_r=r,
        mape=m,
        ship_gates=gates,
    )


def cross_check(
    entries: list[tuple[str, Layout, float]],
    **kwargs: Any,
) -> CrossCheckReport:
    """Run validate_and_measure on each (id, layout, sim_rate) tuple; summarize."""
    results: list[FLEValidation] = []
    for lid, lay, sim_r in entries:
        v = validate_and_measure(lay, layout_id=lid, sim_rate=sim_r, **kwargs)
        results.append(v)
    return summarize(results)
