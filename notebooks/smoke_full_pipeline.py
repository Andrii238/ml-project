"""Small end-to-end Colab smoke pipeline.

Purpose: catch integration bugs before spending time on a full SFT + GRPO run.

Default runtime target: quick, not final-quality.
Runs:
1. SFT into ./ckpts/smoke_sft
2. eval base vs smoke_sft
3. short GRPO into ./ckpts/smoke_grpo
4. eval base vs smoke_sft vs smoke_grpo
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a small SFT -> GRPO -> eval smoke pipeline.")
    ap.add_argument("--sft-epochs", type=int, default=1)
    ap.add_argument("--grpo-steps", type=int, default=12)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--samples-per-layout", type=int, default=1)
    ap.add_argument("--sft-dir", default="./ckpts/smoke_sft")
    ap.add_argument("--grpo-dir", default="./ckpts/smoke_grpo")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    Path("results").mkdir(exist_ok=True)

    if args.clean:
        shutil.rmtree(args.sft_dir, ignore_errors=True)
        shutil.rmtree(args.grpo_dir, ignore_errors=True)

    run([
        sys.executable, "-m", "training.train_sft",
        "--output-dir", args.sft_dir,
        "--num-train-epochs", str(args.sft_epochs),
    ])

    run([
        sys.executable, "-m", "training.evaluate",
        "--checkpoints", "policy_0=BASE", f"policy_sft={args.sft_dir}",
        "--samples-per-layout", str(args.samples_per_layout),
        "--n-val", str(args.n_val),
        "--max-new-tokens", "512",
        "--out", "results/smoke_after_sft.json",
    ])

    if args.group_size % 2 != 0:
        raise ValueError("smoke expects an even group size because per-device batch is fixed at 2")
    grad_accum = args.group_size // 2

    run([
        sys.executable, "-m", "training.train_grpo",
        "--sft-adapter", args.sft_dir,
        "--output-dir", args.grpo_dir,
        "--max-steps", str(args.grpo_steps),
        "--save-steps", str(args.grpo_steps),
        "--group-size", str(args.group_size),
        "--per-device-batch-size", "2",
        "--gradient-accumulation-steps", str(grad_accum),
        "--max-new-tokens", "512",
        "--max-prompt-length", "3500",
    ])

    run([
        sys.executable, "-m", "training.evaluate",
        "--checkpoints", "policy_0=BASE", f"policy_sft={args.sft_dir}",
        f"policy_grpo={args.grpo_dir}",
        "--samples-per-layout", str(args.samples_per_layout),
        "--n-val", str(args.n_val),
        "--max-new-tokens", "512",
        "--out", "results/smoke_full_pipeline.json",
    ])

    print("\nSmoke pipeline finished.")
    print("Main file: results/smoke_full_pipeline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
