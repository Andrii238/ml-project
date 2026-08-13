import json
import os

import pandas as pd

from harness.evaluator import evaluate_policy
from harness.qwen_policy import QwenPolicy
from training.data import VAL_SEEDS
from training.evaluate import CheckpointResult, save_results, rows_to_table
from training.template_sft_generator import build_template_dataset

os.makedirs("results", exist_ok=True)

# Smoke eval intentionally uses the same locked structured distribution as SFT:
# all chests are clustered in one corner/side and the model only places
# assemblers/conveyors. Use 4 examples: small full/partial + large full/partial.
rows = build_template_dataset(VAL_SEEDS[:1], variants_per_seed=4)
prompts = [r["prompt"] for r in rows]
seeds = [r["seed"] for r in rows]
print("smoke target rates:", [round(r["sim_gs_rate"], 4) for r in rows])
print("smoke target assemblers:", [r["n_assemblers"] for r in rows])

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
        max_new_tokens=4096,
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
print("Smoke purpose: quick locked-distribution check that SFT can produce valid productive clustered-chest layouts.")
