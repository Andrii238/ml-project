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
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 4096
    load_in_4bit: bool = False   # Colab bitsandbytes often broken; bf16 fits on T4
    seed: int = 42


def _to_chat_row(pair: dict) -> dict:
    """Chat-format the (prompt, completion) as [system, user, assistant]."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["completion"]},
        ]
    }


def prepare_datasets():
    """Build HF Datasets for train + val. Returns (train_ds, val_ds)."""
    from datasets import Dataset  # lazy import

    train = [_to_chat_row(p) for p in build_sft_dataset(TRAIN_SEEDS)]
    val   = [_to_chat_row(p) for p in build_sft_dataset(VAL_SEEDS)]
    return Dataset.from_list(train), Dataset.from_list(val)


def train(config: SFTConfig | None = None, **overrides: Any) -> None:
    if config is None:
        config = SFTConfig(**overrides)
    else:
        for k, v in overrides.items():
            setattr(config, k, v)

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16,
                                     "device_map": "auto"}
    if config.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
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

    train_ds, val_ds = prepare_datasets()

    trl_args = TRLSFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_seq_length=config.max_seq_length,
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        seed=config.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=trl_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"SFT adapter saved to {config.output_dir}")


if __name__ == "__main__":
    train()
