"""Reward function for the simplified green-science env.

Design: every coefficient lives in `RewardConfig`. Every term is computed by
a small named function and contributes to the returned `RewardBreakdown`.

To swap the reward shape:
- change coefficients: pass a modified `RewardConfig` to `compute_reward`.
- disable a term: comment out its line in `compute_reward`.
- add a term: add a function and add a line.

The random exploration bonus is deterministic per layout: seeded by a hash
of the layout JSON so GRPO's group-based advantage sees consistent rewards.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from .entities import ASSEMBLERS, AssemblerTier, ConveyorTier
from .layout import Layout
from .simulator import SimResult, simulate


# ---------------------------------------------------------------- config

@dataclass
class RewardConfig:
    # 1. Do-nothing (no assembler placed at all).
    do_nothing_penalty: float = 30.0

    # 2. Missing required chest.
    chest_missing_penalty: float = 10.0
    required_chest_kinds: tuple[str, ...] = (
        "input-belts", "input-inserters", "output-science",
    )

    # 3. Milestone reward per assembler.
    milestone_has_belts:     float = 0.5
    milestone_has_inserters: float = 0.5
    milestone_is_producing:  float = 1.0

    # 4. Delivered green-science (main term).
    milestone_delivers_any: float = 20.0
    delivered_reward: float = 300.0  # per pack/sec

    # 5. Produced-but-not-delivered partial credit.
    produced_partial_reward: float = 100.0  # per pack/sec

    # 6. Per-machine cost (dollar-based).
    asm_cost: dict[AssemblerTier, float] = field(default_factory=lambda: {
        1: 0.53, 2: 3.22, 3: 8.94,
    })

    # 7. Per-conveyor tile cost (dollar-based).
    conv_cost: dict[ConveyorTier, float] = field(default_factory=lambda: {
        1: 0.03, 2: 0.23, 3: 0.63,
    })

    # 8. One-time tier-unlock penalties.
    asm_tier2_unlock: float = 3.25
    asm_tier3_unlock: float = 9.0
    conv_tier2_unlock: float = 0.45
    conv_tier3_unlock: float = 1.25

    # 9. Random exploration bonus.
    #    Per-i draw: U(0, upper / (1 + decay * i)) for the i-th placed entity.
    random_asm_upper:   float = 2.0
    random_asm_decay:   float = 0.3
    random_conv_upper:  float = 0.3
    random_conv_decay:  float = 0.1

    # Whether to include the random exploration bonus term at all.
    enable_random_bonus: bool = False


DEFAULT_CONFIG = RewardConfig()


# ---------------------------------------------------------------- breakdown

@dataclass
class RewardBreakdown:
    total: float = 0.0
    do_nothing: float = 0.0
    chest_missing: float = 0.0
    milestone_belts: float = 0.0
    milestone_inserters: float = 0.0
    milestone_producing: float = 0.0
    milestone_delivered: float = 0.0
    delivered: float = 0.0
    produced_partial: float = 0.0
    machine_cost: float = 0.0
    conveyor_cost: float = 0.0
    asm_tier_unlock: float = 0.0
    conv_tier_unlock: float = 0.0
    random_bonus: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {k: v for k, v in vars(self).items()}


# ---------------------------------------------------------------- helpers

def _seed_from_layout(layout: Layout) -> int:
    h = hashlib.sha256(layout.to_json().encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def _random_bonus_for_count(rng: random.Random, upper: float, decay: float,
                              count: int) -> float:
    total = 0.0
    for i in range(count):
        bound = upper / (1.0 + decay * i)
        total += rng.uniform(0.0, bound)
    return total


# ---------------------------------------------------------------- main

def compute_reward(layout: Layout, sim: SimResult | None = None,
                    config: RewardConfig = DEFAULT_CONFIG) -> RewardBreakdown:
    """Compute the composite reward for `layout`. If `sim` is None, runs
    `simulate(layout)` internally."""
    if sim is None:
        sim = simulate(layout)

    br = RewardBreakdown()

    # 1. Do-nothing
    n_asm = layout.machine_count()
    if n_asm == 0:
        br.do_nothing = -config.do_nothing_penalty

    # 2. Missing required chests
    present = {c.kind for c in layout.chests}
    missing = [k for k in config.required_chest_kinds if k not in present]
    br.chest_missing = -config.chest_missing_penalty * len(missing)

    # 3. Milestones per assembler
    n_has_belts = sum(1 for mf in sim.machine_flows if mf.belts_in > 0)
    n_has_ins = sum(1 for mf in sim.machine_flows if mf.inserters_in > 0)
    n_producing = sum(1 for mf in sim.machine_flows if mf.science_out > 0)
    br.milestone_belts = config.milestone_has_belts * n_has_belts
    br.milestone_inserters = config.milestone_has_inserters * n_has_ins
    br.milestone_producing = config.milestone_is_producing * n_producing

    # 4. Delivered green science
    if sim.green_science_rate > 0:
        br.milestone_delivered = config.milestone_delivers_any
    br.delivered = config.delivered_reward * sim.green_science_rate

    # 5. Produced but not delivered
    partial = max(0.0, sim.total_science_produced - sim.green_science_rate)
    br.produced_partial = config.produced_partial_reward * partial

    # 6. Per-machine cost
    tier_counts = layout.machine_count_by_tier()
    br.machine_cost = -sum(config.asm_cost[t] * tier_counts[t]
                            for t in (1, 2, 3))

    # 7. Per-conveyor tile cost
    conv_counts = layout.conveyor_count_by_tier()
    br.conveyor_cost = -sum(config.conv_cost[t] * conv_counts[t]
                             for t in (1, 2, 3))

    # 8. Tier unlock penalties
    if tier_counts[2] > 0:
        br.asm_tier_unlock -= config.asm_tier2_unlock
    if tier_counts[3] > 0:
        br.asm_tier_unlock -= config.asm_tier3_unlock
    if conv_counts[2] > 0:
        br.conv_tier_unlock -= config.conv_tier2_unlock
    if conv_counts[3] > 0:
        br.conv_tier_unlock -= config.conv_tier3_unlock

    # 9. Random exploration bonus (deterministic per layout)
    if config.enable_random_bonus:
        rng = random.Random(_seed_from_layout(layout))
        br.random_bonus = (
            _random_bonus_for_count(rng, config.random_asm_upper,
                                     config.random_asm_decay, n_asm)
            + _random_bonus_for_count(rng, config.random_conv_upper,
                                       config.random_conv_decay,
                                       layout.conveyor_count())
        )

    br.total = (
        br.do_nothing + br.chest_missing
        + br.milestone_belts + br.milestone_inserters + br.milestone_producing
        + br.milestone_delivered + br.delivered + br.produced_partial
        + br.machine_cost + br.conveyor_cost
        + br.asm_tier_unlock + br.conv_tier_unlock
        + br.random_bonus
    )
    return br
