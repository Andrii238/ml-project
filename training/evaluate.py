"""Checkpoint evaluation for the Task 3 comparison.

Produces the per-checkpoint raw-metrics table plan.md §Evaluation requires:

    | checkpoint | green_science/s | materials | area | machines | valid% | composite |

Evaluation reward is the **absolute composite reward** on the post-edit layout,
not the training-time delta. Rationale: the plan's headline claim is
`mean_reward(policy_final) > mean_reward(policy_0)`, which is an absolute
comparison; delta would double-count the input layout baseline.

Runs K=4 completions per val layout for each checkpoint, aggregates into a
`CheckpointMetrics` row, and returns the list of rows so the notebook can plot.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.edit_applier import apply_edits
from harness.edit_parser import parse_edits
from harness.prompt_builder import build_chat_messages
from mini_factorio.layout import Layout
from mini_factorio.reward import compute_reward
from training.data import SplitSizes, build_val_layouts
from training.train_grpo import BASE_MODEL


@dataclass
class Episode:
    layout_index: int
    completion: str
    parse_ok: bool
    n_edits_attempted: int
    n_edits_applied: int
    layout_after_json: str        # post-apply layout, for top-K FLE cross-check
    composite: float
    green_science_rate: float
    materials: float
    cells: int
    machines: int
    valid: bool


@dataclass
class CheckpointMetrics:
    name: str
    mean_composite: float
    mean_green_science: float
    mean_materials: float
    mean_cells: float
    mean_machines: float
    valid_output_pct: float
    parse_ok_pct: float
    n_samples: int
    episodes: list[Episode] = None  # populated when save_episodes=True


def _load(adapter_dir: str | None):
    """adapter_dir=None → base model (policy_0)."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


RESPONSE_PREFIX = '{"edits": ['


@torch.no_grad()
def _generate(model, tokenizer, prompt_text: str, max_new_tokens: int, temperature: float) -> str:
    # Assistant prefill: append `{"edits": [` to the templated prompt so the
    # model can only continue an edit-list JSON. Prepend the prefix back to
    # the decoded completion so the parser sees the whole thing.
    inputs = tokenizer(prompt_text + RESPONSE_PREFIX, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        pad_token_id=tokenizer.pad_token_id,
    )
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return RESPONSE_PREFIX + completion


def evaluate_checkpoint(
    name: str,
    adapter_dir: str | None,
    val_layouts: list[Layout],
    *,
    samples_per_layout: int = 4,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    save_episodes: bool = True,
) -> CheckpointMetrics:
    print(f"[{name}] loading model{' + adapter' if adapter_dir else ''} ...", flush=True)
    t_load = time.time()
    model, tokenizer = _load(adapter_dir)
    print(f"[{name}] loaded in {time.time() - t_load:.1f}s", flush=True)

    episodes: list[Episode] = []
    t_start = time.time()

    for li, layout in enumerate(val_layouts):
        t_layout = time.time()
        messages = build_chat_messages(layout)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for _ in range(samples_per_layout):
            completion = _generate(model, tokenizer, prompt_text, max_new_tokens, temperature)
            parse = parse_edits(completion)
            if not parse.parse_ok:
                episodes.append(Episode(
                    layout_index=li, completion=completion,
                    parse_ok=False, n_edits_attempted=0, n_edits_applied=0,
                    layout_after_json=layout.to_json(),
                    composite=-1.0, green_science_rate=0.0, materials=0.0,
                    cells=0, machines=0, valid=False,
                ))
                continue
            applied = apply_edits(layout, parse.edits)
            rb = compute_reward(applied.layout)
            episodes.append(Episode(
                layout_index=li, completion=completion,
                parse_ok=True,
                n_edits_attempted=len(parse.edits.edits),
                n_edits_applied=applied.n_applied,
                layout_after_json=applied.layout.to_json(),
                composite=rb.composite,
                green_science_rate=rb.green_science_rate,
                materials=rb.materials,
                cells=rb.cells,
                machines=rb.machine_count,
                valid=rb.valid,
            ))
        last = episodes[-samples_per_layout:]
        mean_r = sum(e.composite for e in last) / len(last)
        parse_ok = sum(1 for e in last if e.parse_ok) / len(last)
        print(
            f"[{name}] [{li + 1}/{len(val_layouts)}] {samples_per_layout} samples "
            f"in {time.time() - t_layout:.1f}s  mean_r={mean_r:+.4f}  parse_ok={parse_ok:.0%}",
            flush=True,
        )

    print(f"[{name}] done in {time.time() - t_start:.1f}s ({len(episodes)} episodes)", flush=True)
    n = len(episodes)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return CheckpointMetrics(
        name=name,
        mean_composite=sum(e.composite for e in episodes) / n,
        mean_green_science=sum(e.green_science_rate for e in episodes) / n,
        mean_materials=sum(e.materials for e in episodes) / n,
        mean_cells=sum(e.cells for e in episodes) / n,
        mean_machines=sum(e.machines for e in episodes) / n,
        valid_output_pct=100 * sum(1 for e in episodes if e.valid) / n,
        parse_ok_pct=100 * sum(1 for e in episodes if e.parse_ok) / n,
        n_samples=n,
        episodes=episodes if save_episodes else None,
    )


def evaluate_all(
    checkpoints: list[tuple[str, str | None]],
    *,
    samples_per_layout: int = 4,
    n_val: int = 20,
) -> list[CheckpointMetrics]:
    val_layouts = build_val_layouts(SplitSizes(train=60, val=n_val))
    return [
        evaluate_checkpoint(name, adapter_dir, val_layouts, samples_per_layout=samples_per_layout)
        for name, adapter_dir in checkpoints
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Pairs like NAME=path or NAME=BASE for the un-adapted base model. "
                        "Example: policy_0=BASE policy_1=./ckpts/run1/checkpoint-50")
    p.add_argument("--samples-per-layout", type=int, default=4)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--out", type=str, default="./eval_results.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpts: list[tuple[str, str | None]] = []
    for spec in args.checkpoints:
        name, _, path = spec.partition("=")
        ckpts.append((name, None if path.upper() == "BASE" else path))
    metrics = evaluate_all(ckpts, samples_per_layout=args.samples_per_layout, n_val=args.n_val)
    Path(args.out).write_text(json.dumps([asdict(m) for m in metrics], indent=2))
    for m in metrics:
        print(m)
