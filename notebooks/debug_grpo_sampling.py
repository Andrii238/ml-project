"""Diagnose whether GRPO has useful exploration signal.

This script samples multiple completions for the same prompts, scores them
with the exact GRPO reward function, and reports:

- how many unique completions the policy produced,
- whether rewards differ inside each group,
- parse/apply success,
- green-science rates.

If each group has one unique completion or zero reward spread, GRPO cannot
meaningfully improve because all candidates look the same.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from harness.qwen_policy import DEFAULT_MODEL, QwenPolicy
from training.reward_wrapper import reward_breakdown
from training.train_grpo import prepare_prompt_dataset


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _round(x: float) -> float:
    if math.isfinite(x):
        return round(x, 4)
    return x


def _sample_group(policy: QwenPolicy, prompt: str, *, group_size: int,
                  temperature: float, max_new_tokens: int) -> list[dict[str, Any]]:
    completions = policy.generate(
        [prompt] * group_size,
        batch_size=group_size,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    rows: list[dict[str, Any]] = []
    for i, completion in enumerate(completions):
        br = reward_breakdown(prompt, completion)
        rows.append({
            "sample": i,
            "hash": _hash(completion),
            "reward": br.get("total", -50.0),
            "green_science_rate": br.get("green_science_rate", 0.0),
            "edits_parsed": br.get("edits_parsed", 0),
            "edits_applied": br.get("edits_applied", 0),
            "error": br.get("error"),
            "edit_errors": br.get("edit_errors", []),
            "completion": completion,
            "reward_breakdown": br,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug GRPO sample diversity/reward spread")
    ap.add_argument("--adapter", default="./ckpts/sft",
                    help="LoRA adapter to inspect, or BASE for no adapter")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default="results/grpo_sampling_debug.json")
    ap.add_argument("--show-completions", action="store_true")
    args = ap.parse_args()

    adapter = None if args.adapter == "BASE" else args.adapter
    policy = QwenPolicy(model_name=args.model_name, adapter_path=adapter)
    prompts = list(prepare_prompt_dataset()["prompt"][:args.n_prompts])

    report: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(prompts):
        rows = _sample_group(
            policy, prompt,
            group_size=args.group_size,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        rewards = [float(r["reward"]) for r in rows]
        greens = [float(r["green_science_rate"]) for r in rows]
        unique_completions = len({r["completion"] for r in rows})
        unique_rewards = len({round(float(r["reward"]), 6) for r in rows})
        prompt_report = {
            "prompt_idx": prompt_idx,
            "unique_completions": unique_completions,
            "unique_rewards": unique_rewards,
            "mean_reward": fmean(rewards),
            "std_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "mean_green_science": fmean(greens),
            "max_green_science": max(greens),
            "parse_ok": sum(1 for r in rows if not r["error"]) / len(rows),
            "valid_apply": sum(1 for r in rows if r["edits_applied"] > 0) / len(rows),
            "samples": rows,
        }
        report.append(prompt_report)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"adapter: {args.adapter}")
    print(f"prompts: {args.n_prompts} | group_size: {args.group_size} | temp: {args.temperature}")
    print("prompt | uniq_completion | uniq_reward | mean_reward | std_reward | min..max_reward | mean_green | max_green | parse_ok | valid_apply")
    print("-------+-----------------+-------------+-------------+------------+----------------+------------+-----------+----------+------------")
    for r in report:
        print(
            f"{r['prompt_idx']:>6} | "
            f"{r['unique_completions']:>15} | "
            f"{r['unique_rewards']:>11} | "
            f"{_round(r['mean_reward']):>11} | "
            f"{_round(r['std_reward']):>10} | "
            f"{_round(r['min_reward'])}..{_round(r['max_reward'])} | "
            f"{_round(r['mean_green_science']):>10} | "
            f"{_round(r['max_green_science']):>9} | "
            f"{_round(r['parse_ok']):>8} | "
            f"{_round(r['valid_apply']):>10}"
        )
        best = max(r["samples"], key=lambda s: s["reward"])
        worst = min(r["samples"], key=lambda s: s["reward"])
        print(
            f"       best hash={best['hash']} reward={_round(best['reward'])} "
            f"green={_round(best['green_science_rate'])} edits={best['edits_applied']}"
        )
        print(
            f"       worst hash={worst['hash']} reward={_round(worst['reward'])} "
            f"green={_round(worst['green_science_rate'])} edits={worst['edits_applied']}"
        )
        if args.show_completions:
            print("       BEST COMPLETION:")
            print(best["completion"])
            print("       WORST COMPLETION:")
            print(worst["completion"])

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
