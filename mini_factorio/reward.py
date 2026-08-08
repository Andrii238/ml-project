"""Composite reward for a Mini-Factorio layout.

Per plan.md §Reward:
    R = gs_rate - alpha * materials - beta * cells - gamma * machine_count

Weights are small so gs_rate dominates and secondary terms act as tie-breakers.
`compute_reward` returns both the composite scalar (for GRPO) and every raw
component (for the reporting table in plan §Reward reporting).
"""
from __future__ import annotations

from dataclasses import dataclass

from .layout import Layout
from .simulator import SimResult, simulate

# Initial weights (plan.md §Reward). Tuned on training split only per plan §Reward tuning.
ALPHA = 0.001  # per material item
BETA = 0.01    # per occupied grid cell
GAMMA = 0.05   # per placed machine or inserter

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

    def to_dict(self) -> dict:
        return {
            "composite": self.composite,
            "green_science_rate": self.green_science_rate,
            "materials": self.materials,
            "cells": self.cells,
            "machine_count": self.machine_count,
            "valid": self.valid,
            "sim_errors": self.sim_errors,
        }


def compute_reward(
    layout: Layout,
    *,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
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
    composite = result.green_science_rate - alpha * materials - beta * cells - gamma * m_count
    return RewardBreakdown(
        composite=composite,
        green_science_rate=result.green_science_rate,
        materials=materials,
        cells=cells,
        machine_count=m_count,
        valid=True,
        sim_errors=result.errors,
    )
