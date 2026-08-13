"""Per-checkpoint evaluation on the val split.

Loads each checkpoint (policy_0 = base, policy_1 = SFT, policy_2..N = GRPO
steps), runs it on the 20 val layouts × K samples, and produces:

- composite reward (mean, std)
- green_science_rate (mean)
- machine_count, conveyor_count, total_cells (means)
- parse_ok_rate, valid_rate

Usage:

    from training.evaluate import evaluate_checkpoints
    rows = evaluate_checkpoints([
        {'name': 'policy_0',      'adapter': None},
        {'name': 'policy_1_sft',  'adapter': '/content/ckpts/sft'},
        {'name': 'policy_2',      'adapter': '/content/ckpts/grpo/checkpoint-50'},
        {'name': 'policy_final',  'adapter': '/content/ckpts/grpo/checkpoint-200'},
    ], samples_per_layout=4)
    print(rows_to_table(rows))
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from harness.evaluator import EvalSummary, evaluate_policy
from harness.qwen_policy import DEFAULT_MODEL, QwenPolicy
from training.data import val_samples


@dataclass
class CheckpointResult:
    name: str
    adapter: str | None
    summary: EvalSummary

    def as_row(self) -> dict[str, Any]:
        s = self.summary
        return {
            "name": self.name,
            "adapter": self.adapter,
            "mean_reward": round(s.mean_reward, 3),
            "mean_composite": round(s.mean_reward, 3),
            "std_reward":  round(s.std_reward, 3),
            "mean_green_science": round(s.mean_green_science, 4),
            "mean_machines": round(s.mean_machine_count, 2),
            "mean_conveyors": round(s.mean_conveyor_count, 2),
            "mean_cells": round(s.mean_total_cells, 2),
            "mean_materials": round(s.mean_machine_count + s.mean_conveyor_count, 2),
            "parse_ok_rate": round(s.parse_ok_rate, 3),
            "parse_ok_pct": round(100 * s.parse_ok_rate, 1),
            "valid_rate": round(s.valid_rate, 3),
            "valid_output_pct": round(100 * s.valid_rate, 1),
            "n": s.n,
        }


def evaluate_checkpoints(specs: list[dict[str, Any]], *,
                          samples_per_layout: int = 4,
                          model_name: str | None = None,
                          batch_size: int = 4,
                          max_new_tokens: int = 512,
                          temperature: float = 0.0,
                          n_val: int | None = None) -> list[CheckpointResult]:
    val = val_samples()
    if n_val is not None:
        val = val[:n_val]
    prompts = [s.prompt for s in val]
    seeds = [s.seed for s in val]

    results: list[CheckpointResult] = []
    for spec in specs:
        pol = QwenPolicy(
            model_name=model_name or DEFAULT_MODEL,
            adapter_path=spec.get("adapter"),
            load_in_4bit=spec.get("load_in_4bit", False),
        )
        summary = evaluate_policy(
            pol.generate, prompts, seeds=seeds,
            samples_per_prompt=samples_per_layout,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        results.append(CheckpointResult(
            name=spec["name"], adapter=spec.get("adapter"), summary=summary,
        ))
    return results


def rows_to_table(results: list[CheckpointResult]) -> str:
    rows = [r.as_row() for r in results]
    if not rows:
        return "(no results)"
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    body = "\n".join(" | ".join(str(r[c]).ljust(widths[c]) for c in cols)
                     for r in rows)
    return f"{header}\n{sep}\n{body}"


def save_results(results: list[CheckpointResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([r.as_row() for r in results], f, indent=2)


def _parse_checkpoint_specs(raw: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"checkpoint must be NAME=PATH or NAME=BASE, got {item!r}")
        name, path = item.split("=", 1)
        specs.append({"name": name, "adapter": None if path == "BASE" else path})
    return specs


def _main() -> int:
    import argparse
    import pathlib

    ap = argparse.ArgumentParser(description="Evaluate base/SFT/GRPO policy checkpoints")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="entries like policy_0=BASE policy_1=./ckpts/sft")
    ap.add_argument("--samples-per-layout", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    results = evaluate_checkpoints(
        _parse_checkpoint_specs(args.checkpoints),
        samples_per_layout=args.samples_per_layout,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        n_val=args.n_val,
    )
    print(rows_to_table(results))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, str(out))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
