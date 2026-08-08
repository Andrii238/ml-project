"""Unit tests for Task 1 — Sutton & Barto §4.2-4.3 policy iteration.

The headline test empirically verifies Eq (4.8) — the Policy Improvement
Theorem — by asserting V_new(s) ≥ V_old(s) for every state at every iteration
across 100 random environments of varying sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from task1_gridworld.environment import ACTIONS, DOWN, RIGHT, UP, GridWorld
from task1_gridworld.policy_iteration import (
    evaluate_policy_iterative,
    evaluate_policy_linear,
    improve_policy,
    policy_iteration,
    q_values,
)
from task1_gridworld.random_envs import random_envs_batch, random_gridworld


# ------------------------- Environment invariants -------------------------

def test_env_construction_shapes():
    env = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4))
    assert env.num_states == 25
    assert env.num_actions == 4
    assert env.P.shape == (25, 4, 25)
    assert env.R.shape == (25, 4)


def test_env_transition_is_row_stochastic():
    env = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4),
                    traps=frozenset([(2, 2)]), slip_prob=0.15)
    row_sums = env.P.sum(axis=2)
    assert np.allclose(row_sums, 1.0), "each P[s, a, :] must be a valid distribution"


def test_env_goal_is_absorbing_with_zero_reward():
    env = GridWorld(rows=4, cols=4, start=(0, 0), goal=(3, 3))
    for a in ACTIONS:
        assert env.P[env.goal_state, a, env.goal_state] == 1.0
        assert env.R[env.goal_state, a] == 0.0


def test_env_trap_charges_penalty_on_entry_only():
    env = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4),
                    traps=frozenset([(2, 2)]), trap_penalty=-2.0)
    s_from = env.cell_to_state((1, 2))
    s_trap = env.cell_to_state((2, 2))
    assert env.P[s_from, DOWN, s_trap] == 1.0
    assert env.R[s_from, DOWN] == pytest.approx(-2.0)
    # Leaving the trap cell is a normal step; no ongoing penalty, no stun.
    s_below = env.cell_to_state((3, 2))
    assert env.P[s_trap, DOWN, s_below] == 1.0
    assert env.R[s_trap, DOWN] == pytest.approx(-1.0)


def test_env_walls_block_movement():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2),
                    walls=frozenset([(0, 1)]))
    # From (0,0) going RIGHT should hit the wall and stay put
    s0 = env.cell_to_state((0, 0))
    assert env.P[s0, RIGHT, s0] == 1.0


def test_env_boundary_stays_put():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2))
    s0 = env.cell_to_state((0, 0))
    # UP from (0,0) is out-of-bounds -> stay put
    assert env.P[s0, UP, s0] == 1.0


def test_env_rejects_invalid_config():
    with pytest.raises(AssertionError):
        GridWorld(rows=3, cols=3, start=(0, 0), goal=(0, 0))
    with pytest.raises(AssertionError):
        GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2),
                  walls=frozenset([(0, 0)]))
    with pytest.raises(AssertionError):
        GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2),
                  traps=frozenset([(2, 2)]))


# ---------------------- Policy-evaluation cross-check ----------------------

def test_linear_vs_iterative_evaluation_agree():
    for slip in (0.0, 0.1, 0.3):
        env = GridWorld(rows=6, cols=6, start=(0, 0), goal=(5, 5),
                        traps=frozenset([(2, 3), (4, 1)]),
                        walls=frozenset([(1, 4), (3, 2)]),
                        slip_prob=slip)
        pi = env.uniform_random_policy()
        V_lin = evaluate_policy_linear(pi, env)
        V_it = evaluate_policy_iterative(pi, env, tol=1e-12)
        assert np.max(np.abs(V_lin - V_it)) < 1e-8


# ------------------------- Improvement invariants -------------------------

def test_improve_policy_rows_sum_to_one():
    env = GridWorld(rows=4, cols=4, start=(0, 0), goal=(3, 3),
                    traps=frozenset([(1, 1)]))
    V = evaluate_policy_linear(env.uniform_random_policy(), env)
    pi = improve_policy(V, env)
    assert np.allclose(pi.sum(axis=1), 1.0)
    # Non-tied maximizers should have zero probability
    Q = q_values(V, env)
    max_q = Q.max(axis=1, keepdims=True)
    is_max = Q >= (max_q - 1e-9)
    non_max_prob = np.where(is_max, 0.0, pi).sum()
    assert non_max_prob == 0.0


def test_improve_policy_uniform_on_ties():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2))
    # In a totally symmetric layout, the goal state ties all 4 actions
    V = evaluate_policy_linear(env.uniform_random_policy(), env)
    pi = improve_policy(V, env)
    # Goal state: all actions are self-loops with R=0 → all Q's equal → uniform
    assert np.allclose(pi[env.goal_state], 0.25)


# -------------------- Policy-improvement theorem (Eq 4.8) -----------------

def _assert_value_monotone(history: list[dict], tol: float = 1e-8) -> None:
    """Assert V_new(s) ≥ V_old(s) at every iteration for every state — the
    Sutton-Barto policy improvement theorem (Eq 4.8) applied step-by-step."""
    for i in range(1, len(history)):
        V_prev = history[i - 1]["V"]
        V_curr = history[i]["V"]
        min_delta = float((V_curr - V_prev).min())
        assert min_delta >= -tol, (
            f"Eq (4.8) violated at iter {i}: V_new(s) < V_old(s) by {-min_delta} "
            f"at some state"
        )


def test_eq_4_8_holds_on_deterministic_env():
    env = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4),
                    traps=frozenset([(2, 2)]), slip_prob=0.0)
    _, _, history = policy_iteration(env)
    _assert_value_monotone(history)


def test_eq_4_8_holds_on_stochastic_env():
    env = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4),
                    traps=frozenset([(2, 2)]), slip_prob=0.2)
    _, _, history = policy_iteration(env)
    _assert_value_monotone(history)


# ------------------ Robustness: 100 random envs of varied sizes -----------

@pytest.mark.parametrize("slip_prob", [0.0, 0.1])
def test_eq_4_8_holds_across_random_envs(slip_prob: float):
    """The headline verification: 100 random envs (25 each at sizes
    {5, 10, 15, 20}). Assert V_new(s) ≥ V_old(s) at every iter, every state."""
    envs = random_envs_batch(
        sizes=(5, 10, 15, 20),
        n_per_size=25,
        slip_prob=slip_prob,
        seed=42 if slip_prob == 0.0 else 43,
    )
    assert len(envs) == 100
    for env in envs:
        _, _, history = policy_iteration(env, max_iters=200, tol=1e-8)
        _assert_value_monotone(history, tol=1e-7)


def test_policy_iteration_converges_within_bounds():
    """For finite MDPs policy iteration converges in a finite number of steps.
    Empirical bound: for grids up to 20x20 with our params, well under 50 iters."""
    envs = random_envs_batch(sizes=(5, 10, 20), n_per_size=5, seed=7)
    for env in envs:
        _, _, history = policy_iteration(env, max_iters=200, tol=1e-8)
        # History length includes the initial policy, so improvement steps = len-1.
        assert len(history) - 1 < 100, (
            f"Did not converge quickly: {len(history) - 1} iterations "
            f"on a {env.rows}x{env.cols} env"
        )


# ---------------------- End-to-end optimality sanity check ----------------

def test_optimal_value_improves_over_random():
    env = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4),
                    traps=frozenset([(2, 2)]), gamma=0.9)
    _, _, history = policy_iteration(env)
    V_start_initial = history[0]["V"][env.start_state]
    V_start_final = history[-1]["V"][env.start_state]
    assert V_start_final > V_start_initial + 1e-3, (
        f"Optimal policy no better than random: initial={V_start_initial:.3f}, "
        f"final={V_start_final:.3f}"
    )
