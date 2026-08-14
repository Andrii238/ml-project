"""LoRA SFT warmup for the simplified env.

Feeds the base Qwen model (prompt, completion) pairs produced by
`training.sft_data.build_sft_dataset(TRAIN_SEEDS)`. Uses TRL's SFTTrainer
with a LoRA adapter for T4-friendly memory footprint.

Run this after the SFT dataset has been generated. In Colab:

    from training.train_sft import train
    train(output_dir='/content/ckpts/sft', epochs=3)

Requires: transformers, trl, peft, bitsandbytes, accelerate, datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.prompt_builder import SYSTEM_MESSAGE
from training.data import TRAIN_SEEDS, VAL_SEEDS
from training.sft_data import build_sft_dataset


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@dataclass
class SFTConfig:
    model_name: str = DEFAULT_MODEL
    output_dir: str = "./ckpts/sft"
    epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    max_steps: int = -1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 4096
    load_in_4bit: bool = False   # Colab bitsandbytes often broken; 1.5B + LoRA fits on T4
    dtype: str = "float16"       # T4 supports fp16, not bf16
    seed: int = 42


def _to_chat_row(pair: dict, tokenizer: Any) -> dict:
    """Format as a prompt-completion row matching inference exactly.

    TRL masks prompt tokens automatically for prompt-completion datasets when
    completion_only_loss=True. This prevents the long Factorio prompt from
    dominating the loss while the short JSON answer remains poorly learned.
    """
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": pair["prompt"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt": prompt,
        "completion": pair["completion"] + tokenizer.eos_token,
    }


def prepare_datasets(tokenizer: Any):
    """Build HF Datasets for train + val. Returns (train_ds, val_ds)."""
    from datasets import Dataset  # lazy import

    train = [_to_chat_row(p, tokenizer) for p in build_sft_dataset(TRAIN_SEEDS)]
    val   = [_to_chat_row(p, tokenizer) for p in build_sft_dataset(VAL_SEEDS)]
    return Dataset.from_list(train), Dataset.from_list(val)


def train(config: SFTConfig | None = None, **overrides: Any) -> None:
    if config is None:
        config = SFTConfig(**overrides)
    else:
        for k, v in overrides.items():
            setattr(config, k, v)

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[config.dtype]
    model_kwargs: dict[str, Any] = {"dtype": dtype, "device_map": "auto"}
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig  # lazy — Colab's bitsandbytes may be broken
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(config.model_name,
                                                    trust_remote_code=True,
                                                    **model_kwargs)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_ds, val_ds = prepare_datasets(tokenizer)

    import inspect

    trl_kwargs: dict[str, Any] = dict(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        max_seq_length=config.max_seq_length,
        bf16=config.dtype == "bfloat16",
        fp16=config.dtype == "float16",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        seed=config.seed,
        completion_only_loss=True,
    )
    # TRL 0.24 renamed max_seq_length to max_length.
    trl_kwargs["max_length"] = config.max_seq_length
    accepted = set(inspect.signature(TRLSFTConfig.__init__).parameters)
    trl_args = TRLSFTConfig(**{k: v for k, v in trl_kwargs.items() if k in accepted})

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=trl_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    if "max_seq_length" not in accepted:
        trainer_accepted = set(inspect.signature(SFTTrainer.__init__).parameters)
        if "max_seq_length" in trainer_accepted:
            trainer_kwargs["max_seq_length"] = config.max_seq_length

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"SFT adapter saved to {config.output_dir}")


def _parse_args() -> SFTConfig:
    import argparse

    ap = argparse.ArgumentParser(description="LoRA SFT warmup for Qwen policy")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", default="./ckpts/sft")
    ap.add_argument("--epochs", "--num-train-epochs", dest="epochs", type=int, default=3)
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ns = ap.parse_args()
    return SFTConfig(**vars(ns))


if __name__ == "__main__":
    train(_parse_args())
