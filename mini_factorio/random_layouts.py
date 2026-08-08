"""Random layout generator for training/evaluation splits.

Two modes:
- `empty_layout`: empty grid with placed resource patches and a starting budget.
  This is the "empty grid with budget" case from plan.md — the model must add
  every machine.
- `random_partial_layout`: same map + budget as empty, plus a random handful of
  machines/inserters/belts already placed. Some will be non-optimal on purpose,
  giving the model something to fix.

All numbers derive from real Factorio (via entities.py / recipes.py). The random
seed is exposed so train and val splits are reproducible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .entities import MACHINES, RESOURCE_TYPES
from .layout import Belt, BeltTile, Inserter, Layout, Machine, Resource

DEFAULT_GRID = (16, 16)
DEFAULT_BUDGET = {
    "iron-plate": 400,
    "copper-plate": 100,
    "stone": 80,
    "iron-gear-wheel": 60,
    "electronic-circuit": 60,
}


@dataclass
class ResourceLayoutConfig:
    grid_size: tuple[int, int] = DEFAULT_GRID
    patch_size: int = 3
    types: tuple[str, ...] = RESOURCE_TYPES  # all four (coal needed for furnace fuel)


def _random_resource_positions(
    rng: random.Random, cfg: ResourceLayoutConfig
) -> list[Resource]:
    """Place one square patch per resource type near the edges, non-overlapping."""
    w, h = cfg.grid_size
    ps = cfg.patch_size
    placed: list[Resource] = []
    tried = 0
    for rtype in cfg.types:
        while tried < 200:
            tried += 1
            # Bias placement toward the left / top edges (source-y).
            edge = rng.choice(["left", "top", "right", "bottom"])
            if edge == "left":
                x, y = 0, rng.randint(0, h - ps)
            elif edge == "top":
                x, y = rng.randint(0, w - ps), 0
            elif edge == "right":
                x, y = w - ps, rng.randint(0, h - ps)
            else:
                x, y = rng.randint(0, w - ps), h - ps
            # Overlap check
            if any(_squares_overlap((x, y, ps), (r.x, r.y, r.size)) for r in placed):
                continue
            placed.append(Resource(type=rtype, x=x, y=y, size=ps))
            break
    return placed


def _squares_overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    ax, ay, asz = a
    bx, by, bsz = b
    return not (ax + asz <= bx or bx + bsz <= ax or ay + asz <= by or by + bsz <= ay)


def empty_layout(seed: int, cfg: ResourceLayoutConfig | None = None) -> Layout:
    cfg = cfg or ResourceLayoutConfig()
    rng = random.Random(seed)
    resources = _random_resource_positions(rng, cfg)
    return Layout(
        grid_size=cfg.grid_size,
        resources=resources,
        budget=dict(DEFAULT_BUDGET),
        machines=[],
        inserters=[],
        belts=[],
    )


def _random_free_tile(
    rng: random.Random, layout: Layout, size: tuple[int, int]
) -> tuple[int, int] | None:
    w, h = layout.grid_size
    fw, fh = size
    occupied = set(layout.occupied_tiles().keys())
    # Also exclude ore patches so the tile isn't stuck on a resource for non-miners.
    for _ in range(500):
        x, y = rng.randint(0, w - fw), rng.randint(0, h - fh)
        tiles = [(x + dx, y + dy) for dx in range(fw) for dy in range(fh)]
        if any(t in occupied for t in tiles):
            continue
        return (x, y)
    return None


def random_partial_layout(seed: int, n_extras: int = 3) -> Layout:
    """Empty layout + N random machines/inserters. Purposely not optimal."""
    rng = random.Random(seed)
    layout = empty_layout(seed)
    machine_types = list(MACHINES.keys())
    counter = 0
    for _ in range(n_extras):
        mtype = rng.choice(machine_types)
        spec = MACHINES[mtype]
        pos = _random_free_tile(rng, layout, spec.size)
        if pos is None:
            continue
        counter += 1
        machine_kwargs = dict(id=f"rand_m{counter}", type=mtype, x=pos[0], y=pos[1])
        if mtype == "electric-mining-drill":
            # Pick a target resource that overlaps the miner footprint, else skip.
            hits = [
                r.type for r in layout.resources
                if any(
                    r.x <= pos[0] + dx < r.x + r.size and r.y <= pos[1] + dy < r.y + r.size
                    for dx in range(spec.size[0]) for dy in range(spec.size[1])
                )
            ]
            if not hits:
                counter -= 1
                continue
            machine_kwargs["target_resource"] = hits[0]
        elif mtype == "stone-furnace":
            machine_kwargs["recipe"] = rng.choice(["iron-plate", "copper-plate"])
        else:  # assembling-machine-1
            machine_kwargs["recipe"] = rng.choice([
                "iron-gear-wheel", "copper-cable", "electronic-circuit",
                "transport-belt", "inserter", "logistic-science-pack",
            ])
        layout.machines.append(Machine(**machine_kwargs))
    # A few random inserters
    for _ in range(rng.randint(0, n_extras)):
        pos = _random_free_tile(rng, layout, (1, 1))
        if pos is None:
            continue
        counter += 1
        layout.inserters.append(
            Inserter(
                id=f"rand_i{counter}",
                x=pos[0],
                y=pos[1],
                direction=rng.choice(["north", "east", "south", "west"]),
            )
        )
    return layout


def train_val_split(n_train: int = 60, n_val: int = 20) -> tuple[list[Layout], list[Layout]]:
    """Deterministic disjoint seeds: train [0..n_train), val [1000..1000+n_val)."""
    train = [empty_layout(seed=s) if s % 2 == 0 else random_partial_layout(seed=s)
             for s in range(n_train)]
    val = [empty_layout(seed=1000 + s) if s % 2 == 0 else random_partial_layout(seed=1000 + s)
           for s in range(n_val)]
    return train, val
