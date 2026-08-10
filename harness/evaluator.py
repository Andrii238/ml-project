"""End-to-end policy evaluation harness.

`Policy` is any callable taking a prompt string and returning a completion string.
`evaluate_policy` runs the policy on a list of layouts, records per-layout
rewards, valid-edit rate and invalid-JSON rate — the metrics needed for plan.md
§Baseline evaluation notebook.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import mean

from mini_factorio.layout import Layout
from mini_factorio.reward import RewardBreakdown, compute_reward

from .edit_applier import apply_edits
from .edit_parser import parse_edits
from .prompt_builder import build_prompt

Policy = Callable[[str], str]


@dataclass
class EpisodeResult:
    layout_before: Layout
    layout_after: Layout
    completion: str
    parse_ok: bool
    n_edits_attempted: int
    n_edits_applied: int
    edit_errors: list[str]
    reward: RewardBreakdown


@dataclass
class EvalReport:
    episodes: list[EpisodeResult] = field(default_factory=list)

    def mean_reward(self) -> float:
        return mean(e.reward.composite for e in self.episodes) if self.episodes else 0.0

    def mean_green_science(self) -> float:
        return mean(e.reward.green_science_rate for e in self.episodes) \
            if self.episodes else 0.0

    def invalid_json_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if not e.parse_ok) / len(self.episodes)

    def valid_edit_rate(self) -> float:
        total = sum(e.n_edits_attempted for e in self.episodes)
        applied = sum(e.n_edits_applied for e in self.episodes)
        return applied / total if total > 0 else 0.0

    def summary(self) -> dict:
        return {
            "n_episodes": len(self.episodes),
            "mean_reward": self.mean_reward(),
            "mean_green_science": self.mean_green_science(),
            "invalid_json_rate": self.invalid_json_rate(),
            "valid_edit_rate": self.valid_edit_rate(),
        }


def evaluate_policy(
    policy: Policy,
    layouts: Sequence[Layout],
    *,
    samples_per_layout: int = 1,
) -> EvalReport:
    report = EvalReport()
    # Prefer .propose_edits(layout) if the policy exposes it — it uses chat
    # format + assistant prefill. Fall back to prompt-in for plain callables.
    propose = getattr(policy, "propose_edits", None)
    for layout in layouts:
        prompt = build_prompt(layout) if propose is None else None
        for _ in range(samples_per_layout):
            completion = propose(layout) if propose is not None else policy(prompt)
            parse = parse_edits(completion)
            apply = apply_edits(layout, parse.edits)
            reward = compute_reward(apply.layout)
            report.episodes.append(EpisodeResult(
                layout_before=layout,
                layout_after=apply.layout,
                completion=completion,
                parse_ok=parse.parse_ok,
                n_edits_attempted=len(parse.edits.edits),
                n_edits_applied=apply.n_applied,
                edit_errors=[e for e in apply.errors if e],
                reward=reward,
            ))
    return report
