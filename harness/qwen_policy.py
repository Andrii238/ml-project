"""Qwen policy wrapper for baseline eval and GRPO rollouts.

Default model: `Qwen2.5-Coder-1.5B-Instruct` (plan.md §Baseline model).

Loads `transformers` + `torch` lazily so this module imports even in
environments without them (e.g., unit tests). Actual generation happens
only on `generate(...)`.

Supports:
- optional LoRA adapter (peft) for post-SFT / post-GRPO checkpoints.
- 4-bit quantization via bitsandbytes (Colab T4-friendly).
- batched generation.

Usage:
    p = QwenPolicy()  # loads base model
    outs = p.generate([prompt1, prompt2], max_new_tokens=1024, temperature=0.8)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from harness.prompt_builder import SYSTEM_MESSAGE


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@dataclass
class QwenPolicy:
    model_name: str = DEFAULT_MODEL
    adapter_path: str | None = None    # peft LoRA adapter, if any
    load_in_4bit: bool = False         # Colab bitsandbytes often broken; bf16 fits 1.5B on T4
    device: str = "cuda"
    dtype: str = "bfloat16"            # or "float16" / "float32"

    _model: Any = None
    _tokenizer: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                  "float32": torch.float32}[self.dtype]

        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["device_map"] = self.device

        tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                       trust_remote_code=True,
                                                       **kwargs)
        if self.adapter_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._tokenizer = tok
        self._model = model

    def _wrap_chat(self, user_message: str) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]
        return self._tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: Sequence[str], *,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.8,
                 top_p: float = 0.95,
                 batch_size: int = 4,
                 seed: int | None = None) -> list[str]:
        self.load()
        import torch

        if seed is not None:
            torch.manual_seed(seed)

        completions: list[str] = []
        for i in range(0, len(prompts), batch_size):
            batch = list(prompts[i:i + batch_size])
            chats = [self._wrap_chat(p) for p in batch]
            enc = self._tokenizer(chats, return_tensors="pt", padding=True,
                                   truncation=True, max_length=8192).to(
                self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            for j, seq in enumerate(out):
                # Strip the prompt tokens; decode only new tokens.
                prompt_len = enc["input_ids"][j].shape[0]
                new = seq[prompt_len:]
                completions.append(self._tokenizer.decode(
                    new, skip_special_tokens=True))
        return completions

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.generate([prompt], **kwargs)[0]
