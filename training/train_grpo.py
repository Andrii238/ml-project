"""GRPO training entrypoint.

Runs `trl.GRPOTrainer` with a LoRA adapter over `Qwen2.5-Coder-1.5B-Instruct`,
using our simulator-backed reward (`reward_wrapper.reward_fn`). Saves a
checkpoint per outer iteration so `evaluate.py` can compare `policy_0` (base) to
`policy_1..policy_N` (adapters).

Two run modes:
    --dry-run   : G=2, 2 steps, 1 iteration. ~1 minute. Confirms the trainer
                  starts, reward returns floats, KL logs, and a checkpoint
                  saves. Required smoke test before any real run.
    (default)   : G=8, N iterations x M steps per plan §Task 3 Config.

Usage:
    python -m training.train_grpo --dry-run --output-dir ./ckpts/dry
    python -m training.train_grpo --output-dir ./ckpts/run1

On a T4 (Colab): use default. On a Mac (local): use --dry-run only; full runs
will be very slow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from training.data import SplitSizes, build_datasets
from training.reward_wrapper import reward_fn

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def _lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
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
    # Mac / CPU fallback
    return torch.float32


def _grpo_config(args: argparse.Namespace) -> GRPOConfig:
    return GRPOConfig(
        output_dir=args.output_dir,
        # Sampling
        num_generations=args.group_size,          # G in plan (paper: 64, ours: 8)
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        # Optimization
        learning_rate=args.learning_rate,
        beta=args.kl_beta,                         # KL coeff (paper eq 3)
        num_iterations=args.mu,                    # inner GRPO updates per batch
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        # Logging / saving
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=None,
        report_to=[],                              # no wandb; keep it local
        # Memory
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )


def train(args: argparse.Namespace) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=_pick_dtype(),
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # If an SFT adapter is provided, merge it into the base model so GRPO
    # continues from π_1 rather than the raw base. GRPO then attaches its own
    # trainable LoRA on top of the merged weights.
    peft_config = _lora_config()
    if args.init_adapter is not None:
        print(f"Loading SFT adapter from {args.init_adapter} and merging ...", flush=True)
        model = PeftModel.from_pretrained(model, args.init_adapter)
        model = model.merge_and_unload()

    train_ds, _ = build_datasets(
        tokenizer,
        sizes=SplitSizes(train=args.n_train, val=args.n_val),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=_grpo_config(args),
        train_dataset=train_ds,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="./ckpts/run", type=str)
    p.add_argument("--dry-run", action="store_true",
                   help="Small smoke test: G=2, 2 steps. Use before any real run.")
    p.add_argument("--init-adapter", type=str, default=None,
                   help="Path to SFT adapter to merge into base before GRPO. "
                        "If unset, GRPO starts from raw base model (policy_0).")

    # Data
    p.add_argument("--n-train", type=int, default=60)
    p.add_argument("--n-val", type=int, default=20)

    # GRPO
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--kl-beta", type=float, default=0.04)
    p.add_argument("--mu", type=int, default=1, help="Inner GRPO updates per batch")
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-completion-length", type=int, default=1024)

    # Optimizer / batching
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=50)

    args = p.parse_args()
    if args.dry_run:
        args.group_size = 2
        args.max_steps = 2
        args.save_steps = 2
        args.per_device_train_batch_size = 1
        args.gradient_accumulation_steps = 1
        args.n_train = 4
        args.max_completion_length = 256
    return args


if __name__ == "__main__":
    train(parse_args())
