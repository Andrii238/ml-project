"""Random episode generator for the simplified green-science env.

Per-episode chest emission rate: drawn from
    0.9 * Uniform(0, 3) + 0.1 * Uniform(3, 5)
independently for the belts chest and the inserters chest.

Two starting-layout modes:
- `empty_episode(seed)`: only the three chests are placed (random positions).
  Model builds everything else from scratch. Used for the main training set.
- `partial_episode(seed)`: chests + a few random valid conveyor/assembler
  placements pre-existing. Used to expose the model to edit-existing-layout
  scenarios.

Both modes return a `Layout` that passes `validate_layout()`.
"""
from __future__ import annotations

import random

from .entities import ASSEMBLERS, CHEST_KINDS
from .layout import Assembler, Chest, ChestRates, Conveyor, Layout, DEFAULT_GRID


# --------------------------------------------------------------- rates

# Per-episode chest emission rate distribution.
CHEST_RATE_LOW_UPPER = 3.0
CHEST_RATE_HIGH_UPPER = 5.0
CHEST_RATE_HIGH_PROB = 0.1


def sample_chest_rate(rng: random.Random) -> float:
    """One draw from `0.9*U(0,3) + 0.1*U(3,5)` items/sec."""
    if rng.random() < CHEST_RATE_HIGH_PROB:
        return rng.uniform(CHEST_RATE_LOW_UPPER, CHEST_RATE_HIGH_UPPER)
    return rng.uniform(0.0, CHEST_RATE_LOW_UPPER)


def sample_chest_rates(rng: random.Random) -> ChestRates:
    return ChestRates(
        belts=sample_chest_rate(rng),
        inserters=sample_chest_rate(rng),
    )


# --------------------------------------------------------------- placement helpers

def _random_free_tile(rng: random.Random, grid_size: tuple[int, int],
                        occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
    """Return a random tile not in `occupied`, or None if none exist."""
    w, h = grid_size
    all_tiles = [(x, y) for x in range(w) for y in range(h)
                 if (x, y) not in occupied]
    if not all_tiles:
        return None
    return rng.choice(all_tiles)


def _random_free_3x3_anchor(rng: random.Random, grid_size: tuple[int, int],
                              occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
    """Return a top-left (x, y) such that the 3x3 footprint is inside the
    grid and doesn't overlap `occupied`."""
    w, h = grid_size
    candidates = []
    for x in range(w - 2):
        for y in range(h - 2):
            fp = {(x + dx, y + dy) for dx in range(3) for dy in range(3)}
            if not (fp & occupied):
                candidates.append((x, y))
    if not candidates:
        return None
    return rng.choice(candidates)


def _place_random_chests(lay: Layout, rng: random.Random) -> None:
    """Place one chest of each kind at random free tiles."""
    occupied: set[tuple[int, int]] = set()
    for kind in CHEST_KINDS:
        tile = _random_free_tile(rng, lay.grid_size, occupied)
        if tile is None:
            return
        cid = f"chest_{kind}"
        lay.chests.append(Chest(id=cid, kind=kind, x=tile[0], y=tile[1]))
        occupied.add(tile)


# --------------------------------------------------------------- episodes

def empty_episode(seed: int, grid_size: tuple[int, int] = DEFAULT_GRID) -> Layout:
    """Layout with three chests at random positions and no other entities.

    Chest rates are sampled per this episode."""
    rng = random.Random(seed)
    lay = Layout(grid_size=grid_size, chest_rates=sample_chest_rates(rng))
    _place_random_chests(lay, rng)
    return lay


def partial_episode(seed: int, grid_size: tuple[int, int] = DEFAULT_GRID,
                     max_asm: int = 3, max_conv: int = 15) -> Layout:
    """Layout with chests + a random number of asm-1s + a random number of T1
    conveyors placed at valid non-overlapping positions.

    Not an attempt at a good starting layout — just a partial state the model
    can be asked to improve."""
    rng = random.Random(seed)
    lay = empty_episode(seed, grid_size=grid_size)
    occupied = set(lay.occupied_tiles().keys())

    # Random number of assemblers
    n_asm = rng.randint(0, max_asm)
    for i in range(n_asm):
        anchor = _random_free_3x3_anchor(rng, lay.grid_size, occupied)
        if anchor is None:
            break
        aid = f"asm_seed_{i}"
        a = Assembler(id=aid, tier=1, x=anchor[0], y=anchor[1])
        lay.assemblers.append(a)
        occupied.update(a.footprint)

    # Random number of conveyors
    n_conv = rng.randint(0, max_conv)
    directions = ("north", "east", "south", "west")
    for i in range(n_conv):
        tile = _random_free_tile(rng, lay.grid_size, occupied)
        if tile is None:
            break
        cid = f"conv_seed_{i}"
        d = rng.choice(directions)
        lay.conveyors.append(Conveyor(id=cid, tier=1, x=tile[0], y=tile[1], direction=d))
        occupied.add(tile)

    return lay


def sample_episodes(n: int, mode: str = "empty", start_seed: int = 0,
                    grid_size: tuple[int, int] = DEFAULT_GRID) -> list[Layout]:
    """Batch-generate `n` episodes with deterministic seeds start_seed..start_seed+n-1."""
    fn = empty_episode if mode == "empty" else partial_episode
    return [fn(start_seed + i, grid_size=grid_size) for i in range(n)]
