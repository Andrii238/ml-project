"""Composite reward for a Mini-Factorio layout.

Reward:
    R = gs_rate
      + shape_alpha * sum_i min(w_i * item_rate_i, 1.0)   # dense chain shaping
      - alpha * materials - beta * cells - gamma * machine_count

`w_i` = green-science produced per unit rate of item i if fully chained
(derived from recipe stoichiometry). Cap of 1.0 per item = saturation at
"enough of item i for 1 GS/sec". `shape_alpha=0.1` keeps the shaping small
enough that a real full chain (gs=1.0 + up to 7*0.1=0.7 bonus = 1.7) always
beats any partial chain (max 7*0.1 = 0.7 with no green science).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .layout import Layout
from .recipes import RECIPES
from .simulator import SimResult, simulate

# Weight tuning: placement penalties were dropped (BETA=GAMMA=0) because the
# original values (tuned assuming baseline achieves GS≈1) totally swamped the
# shape bonus when GS=0, making working sub-chains score worse than empty
# layouts. Materials penalty kept as tiny anti-bloat.
ALPHA = 0.001  # per material item — tiny, anti-bloat
BETA = 0.0     # was 0.01, dropped so shape bonus dominates placement cost
GAMMA = 0.0    # was 0.05, dropped for same reason

# Chain-proportional shaping. Weights = green-science-per-unit-rate of item i.
# Derivation: 1 GS = 5.5 iron-plate + 1.5 copper-plate + 1.5 gear + 3 cable
# + 1 circuit + 1 belt + 1 inserter at the leaf level.
CHAIN_WEIGHT = {
    'iron-plate': 1.0 / 5.5,
    'copper-plate': 1.0 / 1.5,
    'iron-gear-wheel': 1.0 / 1.5,
    'copper-cable': 1.0 / 3.0,
    'electronic-circuit': 1.0,
    'transport-belt': 1.0,
    'inserter': 1.0,
}
SHAPE_ALPHA = 1.0  # multiplier on the shaped bonus sum

# Penalties for structurally invalid layouts (before simulator can run).
INVALID_LAYOUT_REWARD = -1.0


@dataclass
class RewardBreakdown:
    composite: float
    green_science_rate: float
    materials: float
    cells: int
    machine_count: int
    valid: bool
    sim_errors: list[str]
    item_rates: dict[str, float] = field(default_factory=dict)
    shape_bonus: float = 0.0

    def to_dict(self) -> dict:
        return {
            "composite": self.composite,
            "green_science_rate": self.green_science_rate,
            "materials": self.materials,
            "cells": self.cells,
            "machine_count": self.machine_count,
            "valid": self.valid,
            "sim_errors": self.sim_errors,
            "item_rates": self.item_rates,
            "shape_bonus": self.shape_bonus,
        }


def _aggregate_item_rates(layout: Layout, machine_rate: dict[str, float]) -> dict[str, float]:
    """Sum per-machine output rates by product item."""
    rates: dict[str, float] = {}
    for m in layout.machines:
        if m.recipe is None or m.recipe not in RECIPES:
            continue
        rate = machine_rate.get(m.id, 0.0)
        if rate <= 0.0:
            continue
        # Our recipes are single-product; machine_rate is that product's items/sec.
        for prod in RECIPES[m.recipe].products:
            rates[prod] = rates.get(prod, 0.0) + rate
    return rates


def _shape_bonus(item_rates: dict[str, float]) -> float:
    """Sum of min(w_i * rate_i, 1.0) across shaped items. Cap prevents hoarding."""
    total = 0.0
    for item, w in CHAIN_WEIGHT.items():
        r = item_rates.get(item, 0.0)
        total += min(w * r, 1.0)
    return total


def compute_reward(
    layout: Layout,
    *,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    shape_alpha: float = SHAPE_ALPHA,
) -> RewardBreakdown:
    result: SimResult = simulate(layout)
    validation_errs = layout.validate_layout()
    valid = not validation_errs
    if not valid:
        return RewardBreakdown(
            composite=INVALID_LAYOUT_REWARD,
            green_science_rate=0.0,
            materials=0.0,
            cells=0,
            machine_count=0,
            valid=False,
            sim_errors=validation_errs,
        )
    materials = layout.total_materials_used()
    cells = layout.total_cells_occupied()
    m_count = layout.machine_count()
    item_rates = _aggregate_item_rates(layout, result.machine_rate)
    shape_bonus = _shape_bonus(item_rates)
    composite = (
        result.green_science_rate
        + shape_alpha * shape_bonus
        - alpha * materials
        - beta * cells
        - gamma * m_count
    )
    return RewardBreakdown(
        composite=composite,
        green_science_rate=result.green_science_rate,
        materials=materials,
        cells=cells,
        machine_count=m_count,
        valid=True,
        sim_errors=result.errors,
        item_rates=item_rates,
        shape_bonus=shape_bonus,
    )
