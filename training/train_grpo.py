"""GRPO training entry point.

TRL's `GRPOTrainer` + LoRA on the SFT-warmed Qwen model. Uses our
`training.reward_wrapper.reward_fn` as the reward function.

Colab usage:

    from training.train_grpo import train
    train(sft_adapter='/content/ckpts/sft', output_dir='/content/ckpts/grpo')

Hyperparameters mirror `plan.md` §Task 3 (group size 8, β 0.04, LR 5e-5,
~200 optimizer steps per iteration). Adjustable via `GRPOConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.prompt_builder import SYSTEM_MESSAGE
from training.data import TRAIN_SEEDS
from training.reward_wrapper import reward_fn


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@dataclass
class GRPOConfig:
    model_name: str = DEFAULT_MODEL
    sft_adapter: str | None = None
    output_dir: str = "./ckpts/grpo"

    # GRPO hyperparameters
    num_generations: int = 8            # G — group size
    temperature: float = 0.8
    max_new_tokens: int = 1024
    beta: float = 0.04
    learning_rate: float = 5e-5
    max_steps: int = 200
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    load_in_4bit: bool = False   # Colab bitsandbytes often broken; bf16 fits on T4
    seed: int = 42


def prepare_prompt_dataset():
    """HF Dataset with one column `prompt`, seeded by TRAIN_SEEDS."""
    from datasets import Dataset
    from mini_factorio.random_layouts import empty_episode
    from harness.prompt_builder import build_user_message

    rows = [{"prompt": build_user_message(empty_episode(seed=s))}
            for s in TRAIN_SEEDS]
    return Dataset.from_list(rows)


def train(config: GRPOConfig | None = None, **overrides: Any) -> None:
    if config is None:
        config = GRPOConfig(**overrides)
    else:
        for k, v in overrides.items():
            setattr(config, k, v)

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

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

    if config.sft_adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config.sft_adapter, is_trainable=True)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = prepare_prompt_dataset()

    trl_args = TRLGRPOConfig(
        output_dir=config.output_dir,
        num_generations=config.num_generations,
        temperature=config.temperature,
        max_completion_length=config.max_new_tokens,
        beta=config.beta,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=6,
        seed=config.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        args=trl_args,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config if config.sft_adapter is None else None,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"GRPO adapter saved to {config.output_dir}")


if __name__ == "__main__":
    train()
