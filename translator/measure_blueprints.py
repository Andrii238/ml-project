"""Measure green-science rate of every valid pasted blueprint in FLE.

Reads translator/user_blueprints.txt, decodes each blueprint string, converts
to Layout, then builds in FLE with feed_missing_inputs=True (infinity-chest
supplies every recipe ingredient directly at each assembler/furnace, since
these blueprints omit miners/inputs). Reports FLE-measured green-science
production rate for each.

Usage: uv run python -m translator.measure_blueprints
"""
from __future__ import annotations

import sys
import time

from mini_factorio.recipes import GREEN_SCIENCE_ITEM
from translator.fle_driver import (
    DEFAULT_ADDRESS, DEFAULT_RCON_PASSWORD, DEFAULT_RCON_PORT,
    _connect, _research_all, _sc,
)
from translator.from_fle import (
    blueprint_dict_to_layout, decode_blueprint_string, infer_belt_items,
)
from translator.to_fle import (
    layout_to_lua_commands, read_production_count_lua, set_game_speed_lua,
)

WARMUP_SECONDS = 300.0
MEASUREMENT_SECONDS = 120.0
GAME_SPEED = 100


def _wait_ticks(client, target_delta_seconds: float) -> None:
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


def measure_layout(client, layout, label: str) -> dict:
    result = {"label": label, "build_errors": 0, "fle_rate": None}
    cmds = layout_to_lua_commands(layout, add_power=True, feed_missing_inputs=True)
    for cmd in cmds:
        reply = _sc(client, cmd)
        if reply and ("Error:" in reply or "error running" in reply):
            result["build_errors"] += 1
    read_cmd = read_production_count_lua(GREEN_SCIENCE_ITEM)
    _sc(client, set_game_speed_lua(GAME_SPEED))
    _wait_ticks(client, WARMUP_SECONDS)
    try:
        tb = int(_sc(client, "rcon.print(game.tick)").strip())
        cb = float(_sc(client, read_cmd).strip())
    except (AttributeError, ValueError):
        return result
    _wait_ticks(client, MEASUREMENT_SECONDS)
    try:
        ta = int(_sc(client, "rcon.print(game.tick)").strip())
        ca = float(_sc(client, read_cmd).strip())
    except (AttributeError, ValueError):
        return result
    elapsed = (ta - tb) / 60.0
    if elapsed > 0:
        result["fle_rate"] = (ca - cb) / elapsed
    return result


def main() -> int:
    with open("translator/user_blueprints.txt") as f:
        lines = [(i + 1, ln.strip()) for i, ln in enumerate(f)]
    lines = [(i, s) for i, s in lines if s and not s.startswith("#")]

    client = _connect(DEFAULT_ADDRESS, DEFAULT_RCON_PORT, DEFAULT_RCON_PASSWORD)
    _research_all(client)

    results: list[dict] = []
    for lineno, s in lines:
        try:
            bp = decode_blueprint_string(s)
            if "blueprint" not in bp:
                continue
            lay = blueprint_dict_to_layout(bp)
            if lay.validate_layout():
                continue
            lay = infer_belt_items(lay)
            label = bp["blueprint"].get("label", f"line{lineno}")
        except Exception as e:
            print(f"[{lineno}] decode/translate failed: {e}")
            continue
        print(f"[{lineno}] measuring {label!r} ...", flush=True)
        r = measure_layout(client, lay, f"{lineno}:{label}")
        results.append(r)
        rate = r["fle_rate"]
        print(f"  build_errors={r['build_errors']} fle_rate={rate}")

    print("\n" + "=" * 72)
    print(f"{'layout':<50} {'build_err':>10} {'fle_rate':>10}")
    print("-" * 72)
    for r in results:
        rate = f"{r['fle_rate']:.4f}" if r["fle_rate"] is not None else "  FAIL"
        print(f"{r['label']:<50.50} {r['build_errors']:>10} {rate:>10}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
