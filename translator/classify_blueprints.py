"""Classify each pasted blueprint by what it produces best.

For every blueprint in `translator/user_blueprints.txt`:
1. Decode → translate to Layout → build in FLE (infinity-chest feeds missing inputs).
2. Warm up, then measure per-second production rate for EVERY item in the
   green-science chain, not just green science itself.
3. Tag each blueprint with its `top_item` = the recipe-chain product with the
   highest measured rate. Write results to `results/blueprint_classification.json`.

Usage:
    uv run python -m translator.classify_blueprints
    uv run python -m translator.classify_blueprints --start-line 69   # only new ones
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

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

WARMUP_SECONDS = 120.0
MEASUREMENT_SECONDS = 60.0
GAME_SPEED = 100

# Chain items we track. Order = rough dependency depth so the "top" item is the
# deepest one produced.
CHAIN_ITEMS = [
    "iron-plate",
    "copper-plate",
    "iron-gear-wheel",
    "copper-cable",
    "electronic-circuit",
    "transport-belt",
    "inserter",
    "logistic-science-pack",
]


def _wait_ticks(client, target_delta_seconds: float) -> None:
    try:
        t0 = int(_sc(client, "rcon.print(game.tick)").strip())
    except (AttributeError, ValueError):
        return
    target = t0 + int(target_delta_seconds * 60)
    wall_deadline = time.time() + target_delta_seconds * 3.0 + 10.0
    while time.time() < wall_deadline:
        time.sleep(0.1)
        try:
            cur = int(_sc(client, "rcon.print(game.tick)").strip())
        except (AttributeError, ValueError):
            continue
        if cur >= target:
            return


def measure_all_items(client, layout, label: str) -> dict:
    """Return {'label', 'build_errors', 'rates': {item: rate_per_sec}}."""
    result = {"label": label, "build_errors": 0, "rates": {i: None for i in CHAIN_ITEMS}}
    cmds = layout_to_lua_commands(layout, add_power=True, feed_missing_inputs=True)
    for cmd in cmds:
        reply = _sc(client, cmd)
        if reply and ("Error:" in reply or "error running" in reply):
            result["build_errors"] += 1
    _sc(client, set_game_speed_lua(GAME_SPEED))
    _wait_ticks(client, WARMUP_SECONDS)
    try:
        tb = int(_sc(client, "rcon.print(game.tick)").strip())
    except (AttributeError, ValueError):
        return result
    counts_before: dict[str, float] = {}
    for item in CHAIN_ITEMS:
        try:
            counts_before[item] = float(_sc(client, read_production_count_lua(item)).strip())
        except (AttributeError, ValueError):
            counts_before[item] = 0.0
    _wait_ticks(client, MEASUREMENT_SECONDS)
    try:
        ta = int(_sc(client, "rcon.print(game.tick)").strip())
    except (AttributeError, ValueError):
        return result
    elapsed = (ta - tb) / 60.0
    if elapsed <= 0:
        return result
    for item in CHAIN_ITEMS:
        try:
            ca = float(_sc(client, read_production_count_lua(item)).strip())
        except (AttributeError, ValueError):
            continue
        result["rates"][item] = (ca - counts_before[item]) / elapsed
    return result


def classify(rates: dict) -> str:
    """Deepest chain item produced above threshold; 'none' if all near zero."""
    THRESHOLD = 0.01
    top = "none"
    for item in CHAIN_ITEMS:  # in dependency-depth order
        r = rates.get(item)
        if r is not None and r >= THRESHOLD:
            top = item
    return top


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-line", type=int, default=1,
                    help="Only classify blueprints on lines >= this (1-indexed).")
    ap.add_argument("--out", type=str, default="results/blueprint_classification.json")
    args = ap.parse_args()

    bp_path = pathlib.Path("translator/user_blueprints.txt")
    with open(bp_path) as f:
        lines = [(i + 1, ln.strip()) for i, ln in enumerate(f)]
    lines = [(i, s) for i, s in lines if s and not s.startswith("#") and i >= args.start_line]
    print(f"loaded {len(lines)} blueprint strings (from line {args.start_line})", flush=True)

    client = _connect(DEFAULT_ADDRESS, DEFAULT_RCON_PORT, DEFAULT_RCON_PASSWORD)
    _research_all(client)

    results = []
    for idx, (lineno, s) in enumerate(lines, 1):
        try:
            bp = decode_blueprint_string(s)
            if "blueprint" not in bp:
                print(f"[{idx}/{len(lines)}] line {lineno}: not a blueprint, skipping")
                continue
            lay = blueprint_dict_to_layout(bp)
            v_err = lay.validate_layout()
            if v_err:
                print(f"[{idx}/{len(lines)}] line {lineno}: layout invalid: {v_err[:1]}")
                results.append({
                    "lineno": lineno,
                    "label": bp["blueprint"].get("label", f"line{lineno}"),
                    "build_errors": None,
                    "rates": {i: None for i in CHAIN_ITEMS},
                    "top_item": "invalid",
                    "invalid_reason": v_err[:3],
                })
                continue
            lay = infer_belt_items(lay)
            label = bp["blueprint"].get("label", f"line{lineno}")
        except Exception as e:
            print(f"[{idx}/{len(lines)}] line {lineno}: decode/translate failed: {e}")
            continue
        t0 = time.time()
        print(f"[{idx}/{len(lines)}] measuring {label!r} ...", flush=True)
        r = measure_all_items(client, lay, f"line{lineno}:{label}")
        r["lineno"] = lineno
        r["top_item"] = classify(r["rates"])
        results.append(r)
        wall = time.time() - t0
        rates_str = ", ".join(
            f"{i}={r['rates'][i]:.3f}" for i in CHAIN_ITEMS if r["rates"][i] and r["rates"][i] > 0.01
        )
        print(f"    build_errors={r['build_errors']}  top={r['top_item']}  "
              f"({rates_str or 'nothing produced'})  [{wall:.1f}s]", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print(f"{'#':>3}  {'top item':<25} {'label':<40}")
    print("-" * 78)
    for r in results:
        print(f"{r['lineno']:>3}  {r['top_item']:<25} {r['label'][:38]:<40}")
    print("=" * 78)
    print(f"\nresults written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
