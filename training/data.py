"""Dataset construction for GRPO training.

Builds a Hugging Face `Dataset` from the deterministic 60/20 train/val split in
`mini_factorio.random_layouts`. Each row has:

- `prompt`: chat-templated string ready for the tokenizer (system + user turns
  from `harness.prompt_builder.build_chat_messages`).
- `layout_json`: the input layout serialized as JSON. TRL's GRPOTrainer forwards
  extra dataset columns to the reward function as keyword args, so this is how
  `reward_wrapper.reward_fn` recovers the pre-edit layout for the delta reward.

The chat template is applied at dataset-build time (not inside the trainer) so
we control the format exactly and can inspect a prompt without loading the
model.
"""
from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset

from harness.prompt_builder import build_chat_messages
from mini_factorio.layout import Layout
from mini_factorio.random_layouts import train_val_split


@dataclass
class SplitSizes:
    train: int = 60
    val: int = 20


def _row(layout: Layout, tokenizer) -> dict:
    messages = build_chat_messages(layout)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "layout_json": layout.to_json()}


def build_datasets(tokenizer, sizes: SplitSizes | None = None) -> tuple[Dataset, Dataset]:
    """Return (train_ds, val_ds). `tokenizer` must have a chat template."""
    sizes = sizes or SplitSizes()
    train_layouts, val_layouts = train_val_split(sizes.train, sizes.val)
    train_rows = [_row(l, tokenizer) for l in train_layouts]
    val_rows = [_row(l, tokenizer) for l in val_layouts]
    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)


def build_val_layouts(sizes: SplitSizes | None = None) -> list[Layout]:
    """Bypass tokenizer — used by evaluate.py which iterates layouts directly."""
    sizes = sizes or SplitSizes()
    _, val = train_val_split(sizes.train, sizes.val)
    return val


def build_train_layouts(sizes: SplitSizes | None = None) -> list[Layout]:
    sizes = sizes or SplitSizes()
    train, _ = train_val_split(sizes.train, sizes.val)
    return train
