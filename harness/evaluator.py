"""End-to-end policy evaluation.

`evaluate_policy` runs a callable policy on a list of prompts, aggregates
metrics per completion, and returns a summary. Works with any callable
`(prompts, *, seed=None) -> list[completions]` — including `QwenPolicy` or
a mock function for tests.

Metrics per completion:
- reward (composite)
- green_science_rate (packs/sec delivered)
- machine_count / conveyor_count / total_cells
- valid (parsed AND at least one edit applied)
- parse_ok / any_edits_applied

Aggregate:
- mean/std of each numeric metric.
- valid_rate, parse_ok_rate.
- per-metric std across samples for confidence intervals.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from harness.edit_applier import apply_edits
from harness.edit_parser import parse_edits
from mini_factorio.layout import Layout
from mini_factorio.reward import DEFAULT_CONFIG, RewardConfig, compute_reward
from mini_factorio.simulator import simulate

from training.reward_wrapper import layout_from_prompt

Policy = Callable[..., list[str]]


@dataclass
class SampleMetrics:
    seed: int | None
    parse_ok: bool
    edits_parsed: int
    edits_applied: int
    reward: float
    green_science_rate: float
    total_science_produced: float
    machine_count: int
    conveyor_count: int
    total_cells: int
    valid: bool                # parse_ok AND at least one edit applied
    completion: str


@dataclass
class EvalSummary:
    n: int
    n_valid: int
    parse_ok_rate: float
    valid_rate: float
    mean_reward: float
    std_reward: float
    mean_green_science: float
    mean_machine_count: float
    mean_conveyor_count: float
    mean_total_cells: float
    per_sample: list[SampleMetrics] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = vars(self).copy()
        d["per_sample"] = [vars(s) for s in self.per_sample]
        return d


def _score_completion(prompt: str, completion: str, seed: int | None,
                       config: RewardConfig) -> SampleMetrics:
    parse = parse_edits(completion)
    parse_ok = parse.parse_error is None
    lay = layout_from_prompt(prompt)
    if lay is None or not parse_ok:
        return SampleMetrics(
            seed=seed, parse_ok=parse_ok, edits_parsed=len(parse.edits),
            edits_applied=0, reward=-50.0, green_science_rate=0.0,
            total_science_produced=0.0, machine_count=0, conveyor_count=0,
            total_cells=0, valid=False, completion=completion)
    apply_res = apply_edits(lay, parse.edits)
    sim = simulate(apply_res.layout)
    br = compute_reward(apply_res.layout, sim=sim, config=config)
    return SampleMetrics(
        seed=seed, parse_ok=True, edits_parsed=len(parse.edits),
        edits_applied=apply_res.applied, reward=br.total,
        green_science_rate=sim.green_science_rate,
        total_science_produced=sim.total_science_produced,
        machine_count=apply_res.layout.machine_count(),
        conveyor_count=apply_res.layout.conveyor_count(),
        total_cells=apply_res.layout.total_cells_occupied(),
        valid=apply_res.applied > 0, completion=completion)


def evaluate_policy(policy: Policy, prompts: Sequence[str], *,
                     seeds: Sequence[int] | None = None,
                     samples_per_prompt: int = 1,
                     config: RewardConfig = DEFAULT_CONFIG,
                     **policy_kwargs: Any) -> EvalSummary:
    """Run `policy` on each prompt `samples_per_prompt` times, score every
    completion, return aggregate metrics."""
    prompts_list = list(prompts)
    if seeds is None:
        seeds = list(range(len(prompts_list)))
    seed_list = list(seeds)

    per_sample: list[SampleMetrics] = []
    for k in range(samples_per_prompt):
        completions = policy(prompts_list, seed=k, **policy_kwargs) \
            if _accepts_seed(policy) else policy(prompts_list, **policy_kwargs)
        for prompt, comp, sd in zip(prompts_list, completions, seed_list):
            per_sample.append(_score_completion(prompt, comp, sd, config))

    n = len(per_sample)
    n_valid = sum(1 for s in per_sample if s.valid)
    n_parse_ok = sum(1 for s in per_sample if s.parse_ok)
    rewards = [s.reward for s in per_sample]
    return EvalSummary(
        n=n, n_valid=n_valid,
        parse_ok_rate=n_parse_ok / n if n else 0.0,
        valid_rate=n_valid / n if n else 0.0,
        mean_reward=statistics.fmean(rewards) if rewards else 0.0,
        std_reward=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        mean_green_science=statistics.fmean(s.green_science_rate for s in per_sample) if per_sample else 0.0,
        mean_machine_count=statistics.fmean(s.machine_count for s in per_sample) if per_sample else 0.0,
        mean_conveyor_count=statistics.fmean(s.conveyor_count for s in per_sample) if per_sample else 0.0,
        mean_total_cells=statistics.fmean(s.total_cells for s in per_sample) if per_sample else 0.0,
        per_sample=per_sample,
    )


def _accepts_seed(policy: Policy) -> bool:
    import inspect
    try:
        return "seed" in inspect.signature(policy).parameters
    except (TypeError, ValueError):
        return False
