"""Grid-agnostic finite MDP for Sutton & Barto §4.2-4.3 policy iteration demo.

Traps are ordinary traversable cells that cost `trap_penalty` on entry
(default -2 vs -1 for a normal step). So stepping through a trap is a real
tradeoff — sometimes worth it if it shortens the path enough. Optional
`slip_prob` makes transitions stochastic. Grid sizes up to 20x20 fit
comfortably in a linear-solve policy evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Cell = tuple[int, int]

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)
NUM_ACTIONS = 4
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
ACTION_NAMES: dict[int, str] = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}
ACTION_ARROWS: dict[int, str] = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}


@dataclass
class GridWorld:
    """Finite gridworld MDP with traps, walls, and optional slip transitions.

    Reward convention:
    - Every non-terminal transition costs `step_reward` (typically -1).
    - Stepping into a trap cell yields `trap_penalty` (default -2, replacing
      the step reward for that transition). Trap cells are otherwise ordinary.
    - Reaching the goal yields `goal_reward`. Goal is absorbing terminal.

    Slip convention: with probability `slip_prob`, the executed action is
    replaced by a uniform random action from all four actions.
    """

    rows: int
    cols: int
    start: Cell
    goal: Cell
    traps: frozenset[Cell] = frozenset()
    walls: frozenset[Cell] = frozenset()
    slip_prob: float = 0.0
    trap_penalty: float = -2.0
    step_reward: float = -1.0
    goal_reward: float = 0.0
    gamma: float = 0.9

    num_actions: int = field(init=False, default=NUM_ACTIONS)
    num_states: int = field(init=False)
    P: np.ndarray = field(init=False, repr=False)
    R: np.ndarray = field(init=False, repr=False)
    goal_state: int = field(init=False)
    start_state: int = field(init=False)

    def __post_init__(self) -> None:
        self.traps = frozenset(self.traps)
        self.walls = frozenset(self.walls)
        self._validate()

        self.num_states = self.rows * self.cols
        self.goal_state = self.cell_to_state(self.goal)
        self.start_state = self.cell_to_state(self.start)

        self.P, self.R = self._build_dynamics()

    def _validate(self) -> None:
        assert self.rows > 0 and self.cols > 0
        assert self._in_bounds(self.start) and self._in_bounds(self.goal)
        assert self.start != self.goal
        assert self.start not in self.walls and self.goal not in self.walls
        assert self.start not in self.traps and self.goal not in self.traps
        assert self.walls.isdisjoint(self.traps)
        assert 0.0 <= self.slip_prob <= 1.0
        assert 0.0 < self.gamma < 1.0
        for cell in self.walls | self.traps:
            assert self._in_bounds(cell), f"out of bounds: {cell}"

    def _in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def cell_to_state(self, cell: Cell) -> int:
        r, c = cell
        return r * self.cols + c

    def state_to_cell(self, s: int) -> Cell:
        return (s // self.cols, s % self.cols)

    def is_terminal(self, s: int) -> bool:
        return self.state_to_cell(s) == self.goal

    def _step_deterministic(self, cell: Cell, action: int) -> Cell:
        """Result of `action` from `cell` ignoring noise. Walls/boundaries make you stay put."""
        dr, dc = ACTION_DELTAS[action]
        r, c = cell
        candidate = (r + dr, c + dc)
        if not self._in_bounds(candidate) or candidate in self.walls:
            return cell
        return candidate

    def _build_dynamics(self) -> tuple[np.ndarray, np.ndarray]:
        n_s, n_a = self.num_states, self.num_actions
        P = np.zeros((n_s, n_a, n_s), dtype=np.float64)
        R = np.zeros((n_s, n_a), dtype=np.float64)

        for s in range(n_s):
            cell = self.state_to_cell(s)

            if cell in self.walls:
                for a in ACTIONS:
                    P[s, a, s] = 1.0
                continue

            if cell == self.goal:
                for a in ACTIONS:
                    P[s, a, s] = 1.0
                    R[s, a] = 0.0
                continue

            for a in ACTIONS:
                p_exec = np.full(n_a, self.slip_prob / n_a)
                p_exec[a] += 1.0 - self.slip_prob

                exp_r = 0.0
                for a_exec, p_a in enumerate(p_exec):
                    if p_a == 0.0:
                        continue
                    dest = self._step_deterministic(cell, a_exec)
                    next_s = self.cell_to_state(dest)
                    if dest == self.goal:
                        r = self.goal_reward
                    elif dest in self.traps:
                        r = self.trap_penalty
                    else:
                        r = self.step_reward
                    P[s, a, next_s] += p_a
                    exp_r += p_a * r
                R[s, a] = exp_r

        return P, R

    def uniform_random_policy(self) -> np.ndarray:
        return np.full((self.num_states, self.num_actions), 1.0 / self.num_actions)

    def _reward_of_transition(self, s: int, s_next: int) -> float:
        if self.state_to_cell(s) == self.goal:
            return 0.0
        cell_next = self.state_to_cell(s_next)
        if cell_next == self.goal:
            return self.goal_reward
        if cell_next in self.traps:
            return self.trap_penalty
        return self.step_reward

    def sample_transition(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, float, bool]:
        s_next = int(rng.choice(self.num_states, p=self.P[s, a, :]))
        return s_next, self._reward_of_transition(s, s_next), self.is_terminal(s_next)

    def rollout(
        self,
        pi: np.ndarray,
        rng: np.random.Generator,
        max_steps: int = 200,
    ) -> dict:
        """Sample a trajectory from `start_state` following `pi`."""
        s = self.start_state
        states = [s]
        actions: list[int] = []
        rewards: list[float] = []
        trap_hits = 0
        for _ in range(max_steps):
            a = int(rng.choice(self.num_actions, p=pi[s]))
            s_next, r, done = self.sample_transition(s, a, rng)
            actions.append(a)
            rewards.append(r)
            states.append(s_next)
            if self.state_to_cell(s_next) in self.traps:
                trap_hits += 1
            s = s_next
            if done:
                break
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "reached_goal": self.is_terminal(states[-1]),
            "steps": len(actions),
            "undiscounted_return": float(sum(rewards)),
            "trap_hits": trap_hits,
        }
