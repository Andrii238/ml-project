"""Evaluate what happens across GRPO checkpoints.

This is the direct diagnostic for "does GRPO actually change the policy?".
It evaluates:

- SFT start policy,
- every GRPO checkpoint directory,
- final GRPO adapter directory.

For each one it reports deterministic evaluation and sampled evaluation.
If deterministic metrics are flat but sampled metrics move, GRPO changed the
distribution but not the greedy output. If both are flat, GRPO did not improve.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from training.evaluate import evaluate_checkpoints, save_results


def _checkpoint_step(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def _adapter_specs(sft_dir: str, grpo_dir: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"name": "policy_sft", "adapter": sft_dir}]
    root = Path(grpo_dir)
    checkpoints = sorted(
        [p for p in root.glob("checkpoint-*") if p.is_dir()],
        key=_checkpoint_step,
    )
    for p in checkpoints:
        specs.append({"name": f"grpo_{_checkpoint_step(p)}", "adapter": str(p)})
    specs.append({"name": "policy_grpo_final", "adapter": grpo_dir})
    return specs


def _rows(results) -> list[dict[str, Any]]:
    return [r.as_row() for r in results]


def _merge(det_rows: list[dict[str, Any]], sampled_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {r["name"]: r for r in sampled_rows}
    out = []
    for d in det_rows:
        s = by_name[d["name"]]
        out.append({
            "name": d["name"],
            "adapter": d["adapter"],
            "det_reward": d["mean_reward"],
            "det_green": d["mean_green_science"],
            "det_valid_pct": d["valid_output_pct"],
            "sample_reward": s["mean_reward"],
            "sample_std": s["std_reward"],
            "sample_green": s["mean_green_science"],
            "sample_valid_pct": s["valid_output_pct"],
            "n_det": d["n"],
            "n_sample": s["n"],
        })
    return out


def _print_table(rows: list[dict[str, Any]]) -> None:
    cols = [
        "name", "det_reward", "det_green", "sample_reward", "sample_std",
        "sample_green", "det_valid_pct", "sample_valid_pct",
    ]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate SFT and GRPO checkpoints over training.")
    ap.add_argument("--sft-dir", default="./ckpts/sft")
    ap.add_argument("--grpo-dir", default="./ckpts/grpo")
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--sampled-per-layout", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default="results/grpo_progress.json")
    args = ap.parse_args()

    specs = _adapter_specs(args.sft_dir, args.grpo_dir)
    det = evaluate_checkpoints(
        specs,
        samples_per_layout=1,
        n_val=args.n_val,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    sampled = evaluate_checkpoints(
        specs,
        samples_per_layout=args.sampled_per_layout,
        n_val=args.n_val,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
    )
    rows = _merge(_rows(det), _rows(sampled))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    _print_table(rows)
    print(f"wrote {args.out}")

    # Also save full deterministic/sample rows for deeper inspection.
    stem = Path(args.out)
    save_results(det, str(stem.with_suffix(".deterministic.json")))
    save_results(sampled, str(stem.with_suffix(".sampled.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
