"""TRL-compatible reward function.

Signature matches TRL's `GRPOTrainer` reward_funcs convention:
    reward_fn(prompts, completions, **kwargs) -> list[float]

Pipeline per (prompt, completion):
1. Recover the Layout the prompt describes (via `layout_from_prompt`).
2. Parse the completion for edits.
3. Apply edits to a copy of the layout.
4. Simulate the resulting layout.
5. Compute the composite reward from `mini_factorio.reward.compute_reward`.

Parse and apply errors don't crash the batch — they yield a penalty reward
so GRPO can learn to avoid them.
"""
from __future__ import annotations

import json
from typing import Any

from harness.edit_applier import apply_edits
from harness.edit_parser import parse_edits
from mini_factorio.layout import Layout
from mini_factorio.reward import DEFAULT_CONFIG, RewardConfig, compute_reward
from mini_factorio.simulator import simulate


# Reward penalties for pathological completions.
PARSE_FAIL_REWARD = -50.0        # completion didn't even yield valid JSON
NO_EDITS_APPLIED_REWARD = -40.0  # JSON parsed but every edit rejected


def layout_from_prompt(prompt: str) -> Layout | None:
    """Extract the layout JSON embedded in the prompt.

    The prompt built by `harness.prompt_builder.build_user_message` includes
    a serialized layout as a JSON envelope. To keep this recoverable, we
    stash the full layout JSON inside a `<<LAYOUT>>...<</LAYOUT>>` sentinel
    that the prompt builder emits alongside the human-readable view.
    """
    start = prompt.find("<<LAYOUT>>")
    end = prompt.find("<</LAYOUT>>")
    if start < 0 or end < 0:
        return None
    payload = prompt[start + len("<<LAYOUT>>"): end]
    try:
        return Layout.from_json(payload)
    except Exception:
        return None


def reward_fn(prompts: list[str], completions: list[str],
              config: RewardConfig = DEFAULT_CONFIG,
              **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    for prompt, completion in zip(prompts, completions):
        lay = layout_from_prompt(prompt)
        if lay is None:
            rewards.append(PARSE_FAIL_REWARD)
            continue
        parse_result = parse_edits(completion)
        if parse_result.parse_error is not None:
            rewards.append(PARSE_FAIL_REWARD)
            continue
        apply_result = apply_edits(lay, parse_result.edits)
        if apply_result.applied == 0 and apply_result.errors:
            rewards.append(NO_EDITS_APPLIED_REWARD)
            continue
        sim = simulate(apply_result.layout)
        br = compute_reward(apply_result.layout, sim=sim, config=config)
        rewards.append(br.total)
    return rewards


def reward_breakdown(prompt: str, completion: str,
                      config: RewardConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Diagnostic version — returns the full breakdown for a single sample."""
    lay = layout_from_prompt(prompt)
    if lay is None:
        return {"error": "prompt has no layout envelope", "total": PARSE_FAIL_REWARD}
    parse_result = parse_edits(completion)
    if parse_result.parse_error is not None:
        return {"error": f"parse: {parse_result.parse_error}",
                "total": PARSE_FAIL_REWARD}
    apply_result = apply_edits(lay, parse_result.edits)
    sim = simulate(apply_result.layout)
    br = compute_reward(apply_result.layout, sim=sim, config=config)
    d = br.as_dict()
    d["edits_parsed"] = len(parse_result.edits)
    d["edits_applied"] = apply_result.applied
    d["edit_errors"] = apply_result.errors
    d["green_science_rate"] = sim.green_science_rate
    d["total_science_produced"] = sim.total_science_produced
    return d
