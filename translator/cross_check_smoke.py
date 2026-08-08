"""Small batch cross-check: run each handcrafted layout in FLE, compare rates.

Usage:
    python -m translator.cross_check_smoke

Requires FLE cluster running (`fle cluster start`). Reports per-layout
sim rate vs FLE rate for each measured item, plus percent error. Exits nonzero
if any absolute error > 20% (plan.md's MAPE ship gate).
"""
from __future__ import annotations

import sys

from mini_factorio import handcrafted_layouts as hl
from mini_factorio.simulator import simulate

from .fle_driver import validate_and_measure


# One entry per (layout_id, layout_factory, target_item_for_fle_readout).
# We measure the item that best characterizes the sim's non-trivial output for
# that layout — e.g., iron-plate for the smelting-only smoke.
CASES: list[tuple[str, callable, str, str]] = [
    ("iron_plate_smoke", hl.iron_plate_smoke, "iron-plate", "f_iron"),
    ("iron_gear_with_extractor", hl.iron_gear_with_extractor, "iron-gear-wheel", "a_gear"),
    ("iron_gear_no_extractor", hl.iron_gear_no_extractor, "iron-gear-wheel", "a_gear"),
    ("belt_asm_chain", hl.belt_asm_chain, "iron-gear-wheel", "a_gear"),
]

MEASUREMENT_SECONDS = 120.0
WARMUP_SECONDS = 300.0
GAME_SPEED = 100


def main() -> int:
    print(f"Running {len(CASES)} layouts against FLE. "
          f"Measurement window: {MEASUREMENT_SECONDS}s game-time each.\n")
    rows: list[dict] = []
    max_err = 0.0
    for lid, factory, target_item, target_machine in CASES:
        lay = factory()
        errs = lay.validate_layout()
        if errs:
            print(f"[{lid}] SKIPPED (validation errors): {errs}")
            continue
        sim_result = simulate(lay)
        sim_rate_for_item = sim_result.machine_rate.get(target_machine, 0.0)
        print(f"[{lid}] target item={target_item!r} (from machine {target_machine!r}), "
              f"sim rate={sim_rate_for_item:.4f}/sec")
        fle_result = validate_and_measure(
            lay,
            layout_id=lid,
            sim_rate=sim_rate_for_item,
            target_item=target_item,
            measurement_seconds=MEASUREMENT_SECONDS,
            warmup_seconds=WARMUP_SECONDS,
            game_speed=GAME_SPEED,
        )
        if not fle_result.build_ok:
            print(f"  BUILD FAILED: {fle_result.build_errors}")
            rows.append({"layout": lid, "target": target_item,
                         "sim": sim_rate_for_item, "fle": None,
                         "err_pct": None, "build_ok": False})
            max_err = float("inf")
            continue
        fle_rate = fle_result.fle_rate
        if sim_rate_for_item == 0 and fle_rate == 0:
            err_pct = 0.0
        elif sim_rate_for_item == 0:
            err_pct = float("inf")  # sim=0, FLE>0 — sim under-predicted
        else:
            err_pct = 100.0 * abs(fle_rate - sim_rate_for_item) / sim_rate_for_item
        max_err = max(max_err, err_pct)
        print(f"  FLE rate={fle_rate:.4f}/sec, err={err_pct:.1f}%")
        rows.append({
            "layout": lid, "target": target_item,
            "sim": sim_rate_for_item, "fle": fle_rate,
            "err_pct": err_pct, "build_ok": True,
        })

    print("\n" + "=" * 78)
    print(f"{'layout':<28} {'target':<20} {'sim':>10} {'fle':>10} {'err %':>8}")
    print("-" * 78)
    for r in rows:
        fle_str = f"{r['fle']:.4f}" if r['fle'] is not None else "  FAIL"
        err_str = f"{r['err_pct']:.1f}" if r['err_pct'] is not None and r['err_pct'] != float("inf") else "  ∞"
        print(f"{r['layout']:<28} {r['target']:<20} {r['sim']:>10.4f} {fle_str:>10} {err_str:>8}")
    print("=" * 78)
    print(f"\nmax |err|: {max_err:.1f}%   (ship-gate MAPE ≤ 20%)")
    return 0 if max_err <= 20.0 else 1


if __name__ == "__main__":
    sys.exit(main())
