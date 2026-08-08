"""Random `GridWorld` generator for the Sutton-Barto robustness tests.

Ensures that goal is reachable from start (BFS check ignoring traps but
respecting walls). Retries wall placement until the reachability constraint
is satisfied.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from task1_gridworld.environment import Cell, GridWorld


def _bfs_reachable(rows: int, cols: int, start: Cell, goal: Cell, walls: set[Cell]) -> bool:
    """True iff `goal` is reachable from `start` in a grid where `walls`
    are impassable. Traps are not obstacles — they are traversable at cost.
    """
    if start == goal:
        return True
    visited = {start}
    queue: deque[Cell] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (r + dr, c + dc)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols):
                continue
            if nb in walls or nb in visited:
                continue
            if nb == goal:
                return True
            visited.add(nb)
            queue.append(nb)
    return False


def random_gridworld(
    rows: int,
    cols: int,
    n_traps: int = 0,
    n_walls: int = 0,
    slip_prob: float = 0.0,
    gamma: float = 0.9,
    step_reward: float = -1.0,
    trap_penalty: float = -5.0,
    goal_reward: float = 0.0,
    rng: np.random.Generator | None = None,
    max_wall_retries: int = 200,
) -> GridWorld:
    """Generate a random `GridWorld`.

    Start and goal are two distinct random cells. Traps and walls are sampled
    from the remaining cells with no overlap. Wall placement retries until
    goal is reachable from start via BFS; raises if it cannot be satisfied.
    """
    rng = rng if rng is not None else np.random.default_rng()
    all_cells: list[Cell] = [(r, c) for r in range(rows) for c in range(cols)]

    # Draw start and goal (distinct)
    start_idx, goal_idx = rng.choice(len(all_cells), size=2, replace=False)
    start = all_cells[int(start_idx)]
    goal = all_cells[int(goal_idx)]

    remaining = [cell for cell in all_cells if cell not in (start, goal)]
    remaining_arr = np.array(remaining, dtype=object)

    # Sample traps
    n_traps = min(n_traps, len(remaining_arr))
    trap_ids = rng.choice(len(remaining_arr), size=n_traps, replace=False)
    traps = frozenset(tuple(remaining_arr[i]) for i in trap_ids)

    # Sample walls, retrying if reachability breaks
    remaining_after_traps = [cell for cell in remaining if cell not in traps]

    walls: frozenset[Cell] = frozenset()
    n_walls = min(n_walls, len(remaining_after_traps))
    for _ in range(max_wall_retries):
        if n_walls == 0:
            walls = frozenset()
            break
        wall_ids = rng.choice(len(remaining_after_traps), size=n_walls, replace=False)
        candidate = {tuple(remaining_after_traps[int(i)]) for i in wall_ids}
        if _bfs_reachable(rows, cols, start, goal, candidate):
            walls = frozenset(candidate)
            break
    else:
        raise RuntimeError(
            f"Failed to place {n_walls} walls with goal reachable after "
            f"{max_wall_retries} retries; try fewer walls."
        )

    return GridWorld(
        rows=rows,
        cols=cols,
        start=start,
        goal=goal,
        traps=traps,
        walls=walls,
        slip_prob=slip_prob,
        gamma=gamma,
        step_reward=step_reward,
        trap_penalty=trap_penalty,
        goal_reward=goal_reward,
    )


def random_envs_batch(
    sizes: tuple[int, ...] = (5, 10, 15, 20),
    n_per_size: int = 25,
    trap_frac: float = 0.05,
    wall_frac: float = 0.05,
    slip_prob: float = 0.0,
    seed: int = 42,
) -> list[GridWorld]:
    """Batch-generate `len(sizes) * n_per_size` random envs.

    trap_frac and wall_frac are fractions of grid cells to convert to traps
    and walls respectively.
    """
    rng = np.random.default_rng(seed)
    envs: list[GridWorld] = []
    for size in sizes:
        n_cells = size * size
        n_traps = max(1, int(trap_frac * n_cells))
        n_walls = max(0, int(wall_frac * n_cells))
        for _ in range(n_per_size):
            envs.append(
                random_gridworld(
                    rows=size,
                    cols=size,
                    n_traps=n_traps,
                    n_walls=n_walls,
                    slip_prob=slip_prob,
                    rng=rng,
                )
            )
    return envs
