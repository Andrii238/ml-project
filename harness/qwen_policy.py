"""Qwen2.5-Coder-1.5B-Instruct policy wrapper.

Exposes `QwenPolicy` — a callable `(prompt: str) -> str` — that fits the
`harness.evaluator.Policy` protocol. Also exposes `qwen_layout_policy` which
takes a Layout and drives it through the full harness prompt/chat pipeline.

Model loading is lazy (constructed on first call) so importing this module is
cheap. `torch_dtype`, `device`, and generation args are configurable but the
defaults target our M2/16GB local setup: fp16 on MPS, temperature 0.7,
top_p 0.9, max_new_tokens 1024 — enough for a JSON edits list without truncating.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mini_factorio.layout import Layout

from .prompt_builder import build_chat_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@dataclass
class QwenPolicy:
    """Lazy-loaded Qwen policy. Reuses model+tokenizer across calls."""

    model_name: str = DEFAULT_MODEL
    device: str | None = None       # "cuda" | "mps" | "cpu"; auto if None.
    torch_dtype: str = "float16"     # "float16" | "bfloat16" | "float32"
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 2048
    do_sample: bool = True

    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[self.torch_dtype]
        # MPS dislikes fp16 for some kernels; fall back to fp32 there.
        if self.device == "mps" and dtype == torch.float16:
            dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype
        ).to(self.device)
        self._model.eval()

    def __call__(self, prompt: str) -> str:
        """Prompt-in, completion-out. Wraps `prompt` as a user message."""
        return self.chat([{"role": "user", "content": prompt}])

    def chat(
        self,
        messages: list[dict[str, str]],
        response_prefix: str = "",
    ) -> str:
        """Chat-format entry point. Returns the assistant's reply string.

        If `response_prefix` is set, that text is appended to the templated
        prompt (as if the assistant already emitted it) and prepended back to
        the returned string. Used to constrain outputs to a specific shape
        (e.g. '{"edits": [' forces the reply to be a JSON edit list).
        """
        import torch

        self._load()
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        text = text + response_prefix
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        completion_ids = outputs[0][inputs["input_ids"].shape[1]:]
        completion = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        return response_prefix + completion

    def propose_edits(self, layout: Layout) -> str:
        """Full harness path: layout -> prompt -> chat -> raw completion.

        Uses assistant-prefill `{"edits": [` so the reply is forced into a
        JSON edit list shape. Removes the "wrote prose first" failure mode.
        """
        return self.chat(build_chat_messages(layout), response_prefix='{"edits": [')
