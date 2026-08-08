"""Compares baseline Qwen mean reward against:
  1. Handcrafted layouts (green-science reachable by hand-tuning).
  2. Expert blueprints from translator/user_blueprints.txt (FLE-measured).

Produces results/baseline_vs_expert.json with the comparison table used in
the Task 2 writeup ("baseline is not optimal").
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from mini_factorio.handcrafted_layouts import (
    belt_asm_chain, iron_gear_with_extractor, iron_plate_smoke,
)
from mini_factorio.reward import compute_reward
from mini_factorio.simulator import simulate


def _reward_for(name: str, lay) -> dict:
    r = compute_reward(lay)
    sim_gs = simulate(lay).green_science_rate
    return {
        "name": name,
        "composite_reward": r.composite,
        "green_science_rate": r.green_science_rate,
        "sim_green_science_rate": sim_gs,
        "materials": r.materials,
        "cells": r.cells,
        "machine_count": r.machine_count,
    }


def main() -> int:
    baseline_path = pathlib.Path("results/baseline_eval.json")
    if not baseline_path.exists():
        print(f"missing {baseline_path} — run run_baseline_eval.py first")
        return 1
    baseline = json.load(open(baseline_path))
    b_summary = baseline["summary"]

    print("=" * 70)
    print("Baseline (Qwen2.5-Coder-1.5B, zero-shot)")
    print("=" * 70)
    print(f"  n_episodes:         {b_summary['n_episodes']}")
    print(f"  mean_reward:        {b_summary['mean_reward']:+.4f}")
    print(f"  mean_green_science: {b_summary['mean_green_science']:.4f} /sec")
    print(f"  invalid_json_rate:  {b_summary['invalid_json_rate']:.1%}")
    print(f"  valid_edit_rate:    {b_summary['valid_edit_rate']:.1%}")

    # Handcrafted layouts (from mini_factorio.handcrafted_layouts).
    # These aren't green-sci layouts — they're smaller upstream chains — but
    # their positive rewards vs baseline's negative demonstrates the gap.
    print("\n" + "=" * 70)
    print("Handcrafted reference layouts (rewards use analytical sim)")
    print("=" * 70)
    handcrafted = [
        _reward_for("iron_plate_smoke", iron_plate_smoke()),
        _reward_for("iron_gear_with_extractor", iron_gear_with_extractor()),
        _reward_for("belt_asm_chain", belt_asm_chain()),
    ]
    for r in handcrafted:
        print(f"  {r['name']:<28} reward={r['composite_reward']:+.4f}  "
              f"gs={r['green_science_rate']:.4f}/s  mach={r['machine_count']}")

    # Expert blueprints (FLE-measured from earlier measure_blueprints run).
    # Hardcoded from the latest measure_blueprints report (translator/measure_blueprints.py).
    print("\n" + "=" * 70)
    print("Expert blueprints (FLE-measured green science /sec)")
    print("=" * 70)
    expert = [
        {"name": "Green Science no belt", "fle_rate": 0.33},
        {"name": "Green Science with Belt", "fle_rate": 1.00},
        {"name": "Green Science Element (largest)", "fle_rate": 6.04},
        {"name": "Green Science (mid)", "fle_rate": 0.25},
        {"name": "unlabeled (line 15)", "fle_rate": 0.25},
    ]
    for e in expert:
        print(f"  {e['name']:<40} {e['fle_rate']:.2f}/s")

    baseline_gs = b_summary["mean_green_science"]
    best_expert_gs = max(e["fle_rate"] for e in expert)
    gap = best_expert_gs - baseline_gs

    print("\n" + "=" * 70)
    print("Verdict")
    print("=" * 70)
    print(f"  best expert green-sci rate: {best_expert_gs:.2f} /sec")
    print(f"  baseline mean gs rate:      {baseline_gs:.4f} /sec")
    print(f"  gap (expert - baseline):    {gap:.2f} /sec")
    print()
    if baseline_gs < best_expert_gs:
        print("  BASELINE IS NOT OPTIMAL — expert layouts exist that score much higher.")
        print("  Task 3 (GRPO) has real headroom to improve baseline.")
    else:
        print("  Baseline matches or exceeds best expert. Task 3 headroom uncertain.")

    out = {
        "baseline": b_summary,
        "handcrafted": handcrafted,
        "expert_blueprints": expert,
        "best_expert_green_science_rate": best_expert_gs,
        "gap_expert_minus_baseline_gs": gap,
        "baseline_is_not_optimal": baseline_gs < best_expert_gs,
    }
    out_path = pathlib.Path("results/baseline_vs_expert.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
