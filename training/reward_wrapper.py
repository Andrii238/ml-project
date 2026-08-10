"""TRL-compatible reward function for GRPO training.

Signature matches TRL >= 0.11 `GRPOTrainer` reward function convention:

    def reward_fn(prompts, completions, **kwargs) -> list[float]

Extra dataset columns (here: `layout_json`) are forwarded as list-valued kwargs.

Score semantics — **delta of composite reward** (per user decision, plan §Reward):
    - Parse failure                     -> PARSE_PENALTY  (-1.0)
    - Parse ok, 0 applied edits         ->  0.0           (doing nothing scores 0)
    - Parse ok, applied edits           ->  composite(after) - composite(before)

Rationale for delta: prevents the "output `[]` and win" reward hack. An empty
edit list, or a list where every edit fails validation, scores exactly 0. Any
edit that actually improves the composite reward beats 0, so GRPO will prefer it.
"""
from __future__ import annotations

from harness.edit_applier import apply_edits
from harness.edit_parser import parse_edits
from mini_factorio.layout import Layout
from mini_factorio.reward import compute_reward

PARSE_PENALTY = -1.0


def _score_one(completion: str, layout_json: str) -> float:
    layout = Layout.model_validate_json(layout_json)
    parse = parse_edits(completion)
    if not parse.parse_ok:
        return PARSE_PENALTY
    applied = apply_edits(layout, parse.edits)
    if applied.n_applied == 0:
        return 0.0
    before = compute_reward(layout).composite
    after = compute_reward(applied.layout).composite
    return after - before


def reward_fn(
    prompts: list[str],
    completions: list[str],
    layout_json: list[str] | None = None,
    **_: object,
) -> list[float]:
    if layout_json is None:
        raise ValueError(
            "reward_fn requires `layout_json` column in the dataset (passed by "
            "TRL as a keyword arg). Make sure build_datasets() was used."
        )
    if len(completions) != len(layout_json):
        raise ValueError(
            f"completions ({len(completions)}) and layout_json ({len(layout_json)}) "
            "must have the same length."
        )
    return [_score_one(c, lj) for c, lj in zip(completions, layout_json, strict=True)]
