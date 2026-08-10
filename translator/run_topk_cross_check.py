"""FLE cross-check on the top-K layouts of each named checkpoint.

Inputs:
    - A directory of per-checkpoint completions JSON (produced by
      `training/evaluate.py` after we extend it to save completions), OR
    - A pair of eval JSON files with per-episode `completion` + `layout_index`.

For MVP the script accepts a simple layout-list format: a JSON with
`{"checkpoint_name": [{"layout_json": ..., "sim_rate": float}, ...]}`. That way
we decouple this script from evaluate.py's schema.

For each layout in each checkpoint:
    1. Build the layout in FLE (`translator.fle_driver.validate_and_measure`).
    2. Record build_ok, sim_rate, fle_rate.
Then summarize per-checkpoint: build success rate, Pearson r, MAPE, ship gates.

Usage:
    uv run python -m translator.run_topk_cross_check \
        --input results/topk_for_fle.json \
        --out results/fle_cross_check.json

Ship gates (plan.md §Verification): build_success 100%, Pearson r ≥ 0.9,
MAPE ≤ 20%. Any gate failure gets flagged in the report but the script still
completes so we get a full picture rather than an early exit.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from mini_factorio.layout import Layout
from translator.fle_driver import cross_check


def build_topk_input(
    eval_json_path: str,
    val_layouts_json_path: str | None,
    k: int,
) -> dict[str, list[dict]]:
    """Convenience helper: if we later save per-episode completions with reward,
    we can pick top-K here. Left as a stub because the current evaluate.py
    aggregates and doesn't save per-episode. Users can build the input JSON
    manually or from `results/baseline_eval.with_completions.json` shape.
    """
    raise NotImplementedError(
        "Build the input JSON manually for now — supply "
        "{\"checkpoint_name\": [{\"layout_json\": ..., \"sim_rate\": ...}, ...]}"
    )


def run(input_path: str, out_path: str, **fle_kwargs) -> None:
    with open(input_path) as f:
        by_checkpoint: dict[str, list[dict]] = json.load(f)

    report_by_ckpt: dict[str, dict] = {}
    for ckpt_name, layouts in by_checkpoint.items():
        print(f"\n=== {ckpt_name}: {len(layouts)} layouts ===", flush=True)
        entries = []
        for i, item in enumerate(layouts):
            lay = Layout.model_validate_json(item["layout_json"])
            sim_rate = float(item["sim_rate"])
            entries.append((f"{ckpt_name}[{i}]", lay, sim_rate))
        report = cross_check(entries, **fle_kwargs)
        report_by_ckpt[ckpt_name] = {
            "per_layout": [asdict(r) for r in report.per_layout],
            "build_success_rate": report.build_success_rate,
            "pearson_r": report.pearson_r,
            "mape": report.mape,
            "ship_gates": report.ship_gates,
        }
        print(
            f"  build_ok={report.build_success_rate:.0%}  "
            f"r={report.pearson_r}  MAPE={report.mape}  gates={report.ship_gates}",
            flush=True,
        )

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report_by_ckpt, f, indent=2)
    print(f"\nwrote {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="JSON: {ckpt_name: [{layout_json, sim_rate}, ...]}")
    p.add_argument("--out", default="results/fle_cross_check.json")
    # Passthrough FLE knobs (defaults match plan §FLE cross-check).
    p.add_argument("--measurement-seconds", type=float, default=60.0)
    p.add_argument("--warmup-seconds", type=float, default=120.0)
    p.add_argument("--game-speed", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.input, args.out,
        measurement_seconds=args.measurement_seconds,
        warmup_seconds=args.warmup_seconds,
        game_speed=args.game_speed,
    )
