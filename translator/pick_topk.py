"""Pick top-K episodes from an evaluate.py JSON, format for FLE cross-check.

Input: eval JSON produced by `training/evaluate.py` (a list of CheckpointMetrics
dicts, each with `episodes: [...]`).
Output: JSON expected by `translator/run_topk_cross_check.py`:
    {ckpt_name: [{layout_json, sim_rate}, ...]}

Ranked by composite reward, ties broken by green_science_rate then -materials.
Skips episodes where the layout after edits is empty (nothing to build in FLE).

Usage:
    uv run python -m translator.pick_topk \
        --eval results/checkpoint_eval.json \
        --k 10 \
        --out results/topk_for_fle.json
"""
from __future__ import annotations

import argparse
import json
import pathlib


def pick(eval_path: str, k: int) -> dict[str, list[dict]]:
    data = json.loads(pathlib.Path(eval_path).read_text())
    out: dict[str, list[dict]] = {}
    for ckpt in data:
        name = ckpt["name"]
        eps = ckpt.get("episodes") or []
        eps = [e for e in eps if e["parse_ok"] and e["n_edits_applied"] > 0]
        eps.sort(
            key=lambda e: (e["composite"], e["green_science_rate"], -e["materials"]),
            reverse=True,
        )
        top = eps[:k]
        out[name] = [
            {"layout_json": e["layout_after_json"], "sim_rate": e["green_science_rate"]}
            for e in top
        ]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default="results/topk_for_fle.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = pick(args.eval, args.k)
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2))
    for name, layouts in result.items():
        print(f"{name}: {len(layouts)} layouts")
    print(f"wrote {p}")
