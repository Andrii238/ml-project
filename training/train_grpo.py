"""GRPO training entry point.

TRL's `GRPOTrainer` + LoRA on the SFT-warmed Qwen model. Uses our
`training.reward_wrapper.reward_fn` as the reward function.

Colab usage:

    from training.train_grpo import train
    train(sft_adapter='/content/ckpts/sft', output_dir='/content/ckpts/grpo')

Hyperparameters mirror DeepSeekMath-style conservative RL (group size 8,
β 0.04, LR 1e-6,
~200 optimizer steps per iteration). Adjustable via `GRPOConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.prompt_builder import SYSTEM_MESSAGE
from harness.qwen_policy import has_complete_json_array
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
    max_new_tokens: int = 512
    max_prompt_length: int | None = None
    beta: float = 0.04
    learning_rate: float = 1e-6
    max_steps: int = 200
    save_steps: int = 50
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    load_in_4bit: bool = False   # Colab bitsandbytes often broken; 1.5B + LoRA fits on T4
    dtype: str = "float16"       # T4 supports fp16, not bf16
    seed: int = 42


def prepare_prompt_dataset():
    """HF Dataset with one column `prompt`, seeded by TRAIN_SEEDS.

    Uses the same generated-task distribution as SFT: full empty-build prompts
    and partial-repair prompts. This gives GRPO both construction and editing
    practice instead of only chest-empty layouts.
    """
    from datasets import Dataset
    from training.template_sft_generator import build_template_dataset

    prompts = [{"prompt": r["prompt"]}
               for r in build_template_dataset(TRAIN_SEEDS, variants_per_seed=4)]
    return Dataset.from_list(prompts)


def _install_json_array_stopper(model: Any, tokenizer: Any) -> None:
    """Make TRL generation stop once every sample has closed its JSON array.

    `QwenPolicy.generate` already has this guard for evaluation. GRPOTrainer,
    however, calls `model.generate(...)` directly, so without this hook GRPO can
    waste tokens up to max_completion_length even when a valid JSON array was
    already produced.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    original_generate = model.generate

    class StopAfterJsonArray(StoppingCriteria):
        def __init__(self, start_len: int):
            self.start_len = start_len

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            for seq in input_ids:
                text = tokenizer.decode(seq[self.start_len:],
                                        skip_special_tokens=True)
                if not has_complete_json_array(text):
                    return False
            return True

    def generate_with_json_stop(*args: Any, **kwargs: Any):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            return original_generate(*args, **kwargs)

        stopper = StopAfterJsonArray(start_len=input_ids.shape[1])
        existing = kwargs.get("stopping_criteria")
        if existing is None:
            kwargs["stopping_criteria"] = StoppingCriteriaList([stopper])
        else:
            existing.append(stopper)
            kwargs["stopping_criteria"] = existing
        return original_generate(*args, **kwargs)

    model.generate = generate_with_json_stop


def train(config: GRPOConfig | None = None, **overrides: Any) -> None:
    if config is None:
        config = GRPOConfig(**overrides)
    else:
        for k, v in overrides.items():
            setattr(config, k, v)

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

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

    if config.sft_adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config.sft_adapter, is_trainable=True)

    _install_json_array_stopper(model, tokenizer)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = prepare_prompt_dataset()

    trl_kwargs: dict[str, Any] = dict(
        output_dir=config.output_dir,
        num_generations=config.num_generations,
        temperature=config.temperature,
        max_completion_length=config.max_new_tokens,
        beta=config.beta,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=config.dtype == "bfloat16",
        fp16=config.dtype == "float16",
        logging_steps=5,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=6,
        seed=config.seed,
    )
    if config.max_prompt_length is not None:
        trl_kwargs["max_prompt_length"] = config.max_prompt_length

    # TRL changes quickly; drop unknown kwargs instead of crashing Colab on
    # harmless version skew.
    import inspect
    accepted = set(inspect.signature(TRLGRPOConfig.__init__).parameters)
    trl_kwargs = {k: v for k, v in trl_kwargs.items() if k in accepted}
    trl_args = TRLGRPOConfig(**trl_kwargs)

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


def _parse_args() -> GRPOConfig:
    import argparse

    ap = argparse.ArgumentParser(description="GRPO/LoRA training for Qwen policy")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--sft-adapter", "--init-adapter", dest="sft_adapter", default=None)
    ap.add_argument("--output-dir", default="./ckpts/grpo")
    ap.add_argument("--group-size", "--num-generations", dest="num_generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", "--max-completion-length", dest="max_new_tokens", type=int, default=512)
    ap.add_argument("--max-prompt-length", type=int, default=None)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--per-device-batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # Accepted for notebook compatibility; current reward curriculum is encoded
    # in the fixed TRAIN_SEEDS/prompt generator.
    ap.add_argument("--curriculum", action="store_true")
    ns = ap.parse_args()
    d = vars(ns)
    d.pop("curriculum", None)
    return GRPOConfig(**d)


if __name__ == "__main__":
    train(_parse_args())
