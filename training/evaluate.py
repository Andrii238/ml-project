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
            "std_reward":  round(s.std_reward, 3),
            "mean_green_science": round(s.mean_green_science, 4),
            "mean_machines": round(s.mean_machine_count, 2),
            "mean_conveyors": round(s.mean_conveyor_count, 2),
            "mean_cells": round(s.mean_total_cells, 2),
            "parse_ok_rate": round(s.parse_ok_rate, 3),
            "valid_rate": round(s.valid_rate, 3),
            "n": s.n,
        }


def evaluate_checkpoints(specs: list[dict[str, Any]], *,
                          samples_per_layout: int = 4,
                          model_name: str | None = None,
                          batch_size: int = 4,
                          max_new_tokens: int = 1024,
                          temperature: float = 0.8) -> list[CheckpointResult]:
    val = val_samples()
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
