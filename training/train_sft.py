"""SFT warmup for Stage 1 (plan.md §Task 3 → Stage 1).

Purpose: bootstrap the raw baseline out of the "output empty list or fail parse"
regime so that GRPO has non-zero reward variance to learn from.

Data source (two-track, matches plan §Stage 1):
1. **Rejection sampling** — for each of the 60 training layouts, sample K=16
   completions from the raw baseline (temperature 0.8). Keep the highest-reward
   one if it improves composite reward over the input layout; discard otherwise.
2. **Synthetic seeds** — hand-crafted or blueprint-sourced good layouts with
   derived (starting_layout, correct_edit_list) pairs. Used when rejection
   sampling produces nothing usable for a given training layout.

Output: a LoRA adapter at ./ckpts/sft/, plus config.json capturing every
hyperparameter used (base model, LoRA config, LR, epochs, seeds, git commit).

Run mode:
    --dry-run   : 2 examples, 1 epoch. Smoke test. Confirms trainer starts,
                  loss decreases, adapter saves.
    (default)   : full SFT run per plan.

Usage:
    python -m training.train_sft --dry-run --output-dir ./ckpts/sft_dry
    python -m training.train_sft --output-dir ./ckpts/sft

STATUS: SCAFFOLD. The data-loading and seed-derivation steps are TODOs — they
depend on decisions we haven't made yet (which seed layouts, augmentation
strategy). The training loop itself is complete and testable via --dry-run
once we plug real data in.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from training.train_grpo import BASE_MODEL


@dataclass
class SFTHyperparams:
    """Every knob we set. Serialized to config.json for the writeup."""
    base_model: str = BASE_MODEL
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    seed: int = 42
    n_train_layouts: int = 60
    rejection_samples_per_layout: int = 16
    rejection_temperature: float = 0.8
    # Populated at runtime.
    n_examples_from_rejection: int = 0
    n_examples_from_synthetic: int = 0
    n_examples_total: int = 0
    git_commit: str = ""


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _lora_config(hp: SFTHyperparams) -> LoraConfig:
    return LoraConfig(
        r=hp.lora_r,
        lora_alpha=hp.lora_alpha,
        lora_dropout=hp.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def build_sft_dataset(
    tokenizer,
    hp: SFTHyperparams,
    *,
    dry_run: bool,
) -> Dataset:
    """Assemble the SFT training set as {prompt, completion} rows.

    Data source: 13 FLE-validated blueprints (see training/sft_data.py), each
    turned into 5 pairs via strip-and-re-add augmentation → 65 pairs total.
    Dry-run mode returns just the first 2 pairs so the trainer smoke-test runs
    in seconds.
    """
    from training.sft_data import build_pairs

    pairs = build_pairs(seed=hp.seed)
    if dry_run:
        pairs = pairs[:2]

    rows = []
    for pair in pairs:
        from harness.prompt_builder import build_chat_messages
        messages = build_chat_messages(pair.stripped)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Match the runtime assistant-prefill: model emits everything after
        # `{"edits": [`. Training completion is the whole JSON so the trainer
        # sees the target the way the model will produce it.
        completion = json.dumps({"edits": pair.edits})
        rows.append({"prompt": prompt, "completion": completion})

    hp.n_examples_from_synthetic = len(rows)
    hp.n_examples_total = len(rows)
    return Dataset.from_list(rows)


def train(args: argparse.Namespace) -> None:
    hp = SFTHyperparams(
        learning_rate=args.learning_rate,
        num_train_epochs=1 if args.dry_run else args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )
    hp.git_commit = _git_commit()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=_pick_dtype(),
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    train_ds = build_sft_dataset(tokenizer, hp, dry_run=args.dry_run)
    hp.n_examples_total = len(train_ds)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=hp.num_train_epochs,
        learning_rate=hp.learning_rate,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        max_length=hp.max_seq_length,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        seed=hp.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        peft_config=_lora_config(hp),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    (output_dir / "config.json").write_text(json.dumps(asdict(hp), indent=2))
    print(f"SFT complete. Adapter + config.json written to {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="./ckpts/sft", type=str)
    p.add_argument("--dry-run", action="store_true",
                   help="Smoke test with 2 trivial examples.")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--num-train-epochs", type=int, default=3)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
