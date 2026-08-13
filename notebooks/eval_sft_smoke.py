import json
import os

import pandas as pd

from harness.evaluator import evaluate_policy
from harness.qwen_policy import QwenPolicy
from training.data import VAL_SEEDS
from training.evaluate import CheckpointResult, save_results, rows_to_table
from training.template_sft_generator import _build_large_bus_full_pair

os.makedirs("results", exist_ok=True)

# Smoke eval intentionally uses the locked structured task distribution:
# all chests are clustered near one corner/side and the target pattern is a
# compact multi-assembler bus. This is the first check before full eval/GRPO.
rows = []
for i, seed in enumerate(VAL_SEEDS[:2]):
    pair = _build_large_bus_full_pair(seed, variant=i)
    if pair is None:
        raise RuntimeError(f"failed to build large-bus smoke pair for seed={seed}")
    rows.append({"seed": pair.seed, "prompt": pair.prompt, "target_rate": pair.sim_gs_rate})
prompts = [r["prompt"] for r in rows]
seeds = [r["seed"] for r in rows]
print("smoke target rates:", [round(r["target_rate"], 4) for r in rows])

specs = [
    {"name": "policy_0", "adapter": None},
    {"name": "policy_1", "adapter": "./ckpts/sft"},
]

results = []
for spec in specs:
    policy = QwenPolicy(adapter_path=spec["adapter"], load_in_4bit=False)
    summary = evaluate_policy(
        policy.generate,
        prompts,
        seeds=seeds,
        samples_per_prompt=1,
        batch_size=2,
        max_new_tokens=1024,
        temperature=0.2,
    )
    results.append(CheckpointResult(
        name=spec["name"], adapter=spec["adapter"], summary=summary,
    ))

print(rows_to_table(results))
save_results(results, "results/eval_sft_vs_base_smoke.json")
print("wrote results/eval_sft_vs_base_smoke.json")

d = [r.as_row() for r in results]
table = [{
    "ckpt": c["name"],
    "composite": round(c["mean_composite"], 4),
    "green_sci/s": round(c["mean_green_science"], 4),
    "valid %": round(c["valid_output_pct"], 1),
    "parse_ok %": round(c["parse_ok_pct"], 1),
    "materials": round(c["mean_materials"], 1),
    "cells": round(c["mean_cells"], 1),
    "machines": round(c["mean_machines"], 2),
} for c in d]
print(pd.DataFrame(table).set_index("ckpt").to_string())
print("Smoke purpose: quick locked-distribution check that SFT can produce valid productive large-bus layouts.")
