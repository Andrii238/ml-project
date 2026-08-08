"""Runs the same logic as notebooks/task2_baseline_eval.ipynb but as a script
so we can execute + monitor + interrupt cleanly. Writes results/baseline_eval.json
and prints a progress line for each layout so we know it's alive.

Usage: uv run python notebooks/run_baseline_eval.py [--n-val N] [--samples K]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

# Repo root on path (script sits in notebooks/).
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from harness.evaluator import EpisodeResult, evaluate_policy
from harness.qwen_policy import QwenPolicy
from mini_factorio.random_layouts import train_val_split
from mini_factorio.reward import compute_reward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", type=str, default="results/baseline_eval.json")
    args = ap.parse_args()

    _, val_layouts = train_val_split(n_train=60, n_val=20)
    val_layouts = val_layouts[: args.n_val]
    print(f"val layouts: {len(val_layouts)}", flush=True)

    print("loading Qwen model ...", flush=True)
    t0 = time.time()
    qp = QwenPolicy(
        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
    )
    # Force load now so timing is honest.
    qp._load()
    print(f"model loaded in {time.time() - t0:.1f}s on {qp.device}", flush=True)

    episodes: list[EpisodeResult] = []
    per_layout_rewards: list[list[float]] = []
    for li, layout in enumerate(val_layouts):
        t_lay = time.time()
        group: list[EpisodeResult] = []
        for si in range(args.samples):
            report = evaluate_policy(qp, [layout], samples_per_layout=1)
            group.append(report.episodes[0])
        rewards = [e.reward.composite for e in group]
        per_layout_rewards.append(rewards)
        episodes.extend(group)
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        print(
            f"[{li + 1}/{len(val_layouts)}] {args.samples} samples in "
            f"{time.time() - t_lay:.1f}s  mean_reward={mean_r:.4f}",
            flush=True,
        )

    n = len(episodes)
    mean_reward = sum(e.reward.composite for e in episodes) / n if n else 0.0
    mean_gs = sum(e.reward.green_science_rate for e in episodes) / n if n else 0.0
    invalid_json = sum(1 for e in episodes if not e.parse_ok) / n if n else 0.0
    total_edits = sum(e.n_edits_attempted for e in episodes)
    valid_edits = sum(e.n_edits_applied for e in episodes)
    valid_edit_rate = valid_edits / total_edits if total_edits else 0.0

    summary = {
        "n_episodes": n,
        "mean_reward": mean_reward,
        "mean_green_science": mean_gs,
        "invalid_json_rate": invalid_json,
        "valid_edit_rate": valid_edit_rate,
    }
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": qp.model_name,
            "samples_per_layout": args.samples,
            "n_layouts": args.n_val,
            "summary": summary,
            "per_layout_rewards": per_layout_rewards,
            "episodes": [
                {
                    "parse_ok": e.parse_ok,
                    "n_edits_applied": e.n_edits_applied,
                    "n_edits_attempted": e.n_edits_attempted,
                    "reward": e.reward.to_dict(),
                }
                for e in episodes
            ],
        }, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
