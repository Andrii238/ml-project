"""Sutton & Barto §4.2-4.3 policy iteration on a finite MDP.

Implements Equations (4.6), (4.7), (4.8), (4.9) and the Policy Iteration
algorithm reproduced verbatim from p. 80, with Exercise 4.4 fix (value-change
stability criterion instead of raw action-change comparison).

The default policy-evaluation is the p. 80 iterative sweep (Gauss-Seidel:
V(s) updated in place; subsequent references within the same sweep see the
new value). A closed-form linear solve  V = (I − γ P_π)^{-1} R_π  is also
provided; the tests cross-check the two agree to numerical precision, but
policy iteration itself uses the iterative form to stay faithful to the
textbook algorithm.
"""

from __future__ import annotations

import numpy as np

from task1_gridworld.environment import GridWorld


def _policy_matrix(pi: np.ndarray, env: GridWorld) -> tuple[np.ndarray, np.ndarray]:
    """R_pi[s] = Σ_a π(a|s) R[s, a];  P_pi[s, s'] = Σ_a π(a|s) P[s, a, s']."""
    R_pi = np.einsum("sa,sa->s", pi, env.R)
    P_pi = np.einsum("sa,sap->sp", pi, env.P)
    return R_pi, P_pi


def evaluate_policy_linear(pi: np.ndarray, env: GridWorld) -> np.ndarray:
    """Closed-form policy evaluation.

    Bellman for a stochastic policy π:  v_π = R_π + γ P_π v_π
        ⇒  (I − γ P_π) v_π = R_π
        ⇒  v_π = (I − γ P_π)^{-1} R_π
    """
    R_pi, P_pi = _policy_matrix(pi, env)
    A = np.eye(env.num_states) - env.gamma * P_pi
    return np.linalg.solve(A, R_pi)


def evaluate_policy_iterative(
    pi: np.ndarray,
    env: GridWorld,
    tol: float = 1e-12,
    max_iters: int = 100_000,
) -> np.ndarray:
    """Iterative policy evaluation, verbatim from Sutton-Barto p. 80.

    Pseudocode:
        Loop:
            Δ ← 0
            For each s ∈ S:
                v ← V(s)
                V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r | s, a) [r + γ V(s')]
                Δ ← max(Δ, |v − V(s)|)
        until Δ < θ

    Updates are Gauss-Seidel (in-place: later s in the same sweep may read
    the just-updated V(s') for s' seen earlier). This matches the textbook
    pseudocode exactly and typically converges faster than a synchronous
    (whole-vector) sweep.
    """
    R_pi, P_pi = _policy_matrix(pi, env)
    V = np.zeros(env.num_states)
    n_s = env.num_states
    gamma = env.gamma
    for _ in range(max_iters):
        delta = 0.0
        for s in range(n_s):
            v = V[s]
            V[s] = R_pi[s] + gamma * (P_pi[s] @ V)
            delta = max(delta, abs(v - V[s]))
        if delta < tol:
            return V
    return V


def evaluate_policy(pi: np.ndarray, env: GridWorld) -> np.ndarray:
    """Default policy evaluation: closed-form linear solve.

    Mathematically equivalent to `evaluate_policy_iterative` (the p. 80
    pseudocode) — both compute the fixed point of the Bellman operator.
    The linear solve is faster and slightly more accurate for the finite
    tabular case; the tests cross-check that both agree to numerical
    precision.
    """
    return evaluate_policy_linear(pi, env)


def q_values(V: np.ndarray, env: GridWorld) -> np.ndarray:
    """Eq (4.6):  q(s, a) = R[s, a] + γ Σ_{s'} P[s, a, s'] V[s'].

    Returns Q of shape (num_states, num_actions).
    """
    return env.R + env.gamma * (env.P @ V)


def snake_policy(env: GridWorld) -> np.ndarray:
    """Zigzag snake policy: go DOWN col 0, UP col 1, DOWN col 2, ...
    Visits every cell in order before reaching the goal — maximally dumb,
    hitting every trap along the way."""
    from task1_gridworld.environment import UP, DOWN, RIGHT

    n_s, n_a = env.num_states, env.num_actions
    pi = np.zeros((n_s, n_a))
    for s in range(n_s):
        r, c = env.state_to_cell(s)
        if c == env.cols - 1:
            a = DOWN if r < env.rows - 1 else RIGHT
        elif c % 2 == 0:          # even column: go DOWN
            a = DOWN if r < env.rows - 1 else RIGHT
        else:                      # odd column: go UP
            a = UP if r > 0 else RIGHT
        pi[s, a] = 1.0
    return pi


def dumb_down_then_right_policy(env: GridWorld) -> np.ndarray:
    """A deterministic 'dumb' initial policy: go DOWN until the bottom row,
    then RIGHT. Ignores traps entirely — will walk straight into them.
    Useful as a starting policy to show policy iteration actively learning to
    avoid trap fields.
    """
    from task1_gridworld.environment import DOWN, RIGHT

    n_s, n_a = env.num_states, env.num_actions
    pi = np.zeros((n_s, n_a))
    for s in range(n_s):
        r, _ = env.state_to_cell(s)
        pi[s, DOWN if r < env.rows - 1 else RIGHT] = 1.0
    return pi


def greedy_from_V(V: np.ndarray, env: GridWorld) -> np.ndarray:
    """Deterministic greedy policy w.r.t. V — Eq (4.9): π'(s) = argmax_a Q(s, a).

    Q values have real variation from V, so argmax picks a meaningful action
    at every state (unlike collapsing a stochastic π by argmax over
    probabilities, which is meaningless under a uniform π).
    """
    Q = q_values(V, env)
    n_s, n_a = Q.shape
    det = np.zeros((n_s, n_a))
    det[np.arange(n_s), np.argmax(Q, axis=1)] = 1.0
    return det


def improve_policy(V: np.ndarray, env: GridWorld, tie_tol: float = 1e-14) -> np.ndarray:
    """Eq (4.9) greedy improvement with uniform apportionment across ties.

    For each state s:
        max_q  = max_a Q[s, a]
        tied   = { a : Q[s, a] ≥ max_q − tie_tol }
        π'(a|s) = 1 / |tied|   if a ∈ tied, else 0

    This is one valid apportionment permitted by Sutton & Barto §4.2's
    stochastic-policy extension: any distribution over tied maximizers is
    allowed as long as submaximal actions receive zero probability.

    `tie_tol` is intentionally at double-precision noise level (1e-14). A
    looser tolerance would merge actions whose Q-values differ by more than
    roundoff, which would make π' a mixture of a true maximizer with a
    strictly suboptimal action — producing a policy whose V is slightly
    LOWER than the strict-argmax policy's V. That would violate the
    inequality of Eq (4.7) by the same magnitude and propagate as an
    apparent negative ΔV in policy iteration.
    """
    Q = q_values(V, env)
    max_q = Q.max(axis=1, keepdims=True)
    is_max = Q >= (max_q - tie_tol)
    counts = is_max.sum(axis=1, keepdims=True).astype(np.float64)
    return is_max.astype(np.float64) / counts


def policy_iteration(
    env: GridWorld,
    initial_pi: np.ndarray | None = None,
    max_iters: int = 200,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Sutton & Barto policy iteration with Exercise 4.4 fix.

    Uses the closed-form linear-solve policy evaluation (mathematically
    equivalent to the p. 80 iterative sweep). Policy improvement is the
    greedy step (Eq 4.9) with uniform tie-splitting across argmax.

    Termination: max_s |V_new(s) − V_old(s)| < tol (Ex 4.4 fix — value-change
    stability). This prevents infinite oscillation between equally good
    policies. When values are stationary between successive iterations, the
    current policy is (an) optimal policy.

    Returns:
        V     : final value function (num_states,)
        pi    : final stochastic policy (num_states, num_actions)
        history : list of dicts, one per iteration, each with keys
                  {"iter", "V", "pi"}. history[0] is the initial policy.
    """
    pi = env.uniform_random_policy() if initial_pi is None else initial_pi.copy()
    V = evaluate_policy(pi, env)
    history: list[dict] = [{"iter": 0, "V": V.copy(), "pi": pi.copy()}]

    for i in range(1, max_iters + 1):
        new_pi = improve_policy(V, env)
        new_V = evaluate_policy(new_pi, env)
        if np.max(np.abs(new_V - V)) < tol:
            # Policy is optimal. Do NOT append a redundant iteration
            # showing the same value function again.
            return V, pi, history
        history.append({"iter": i, "V": new_V.copy(), "pi": new_pi.copy()})
        pi, V = new_pi, new_V

    return V, pi, history
