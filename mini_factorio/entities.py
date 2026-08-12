"""Entity definitions for the simplified green-science env.

Sourced from factoriolab data.json (Factorio 2.0.77) for the numeric values.
Everything else here is our simplification of real Factorio for the RL env.

Contents:
- ChestKind             (input-belts, input-inserters, output-science)
- AssemblerTier         (1, 2, 3) with crafting rate for the green-science recipe
- Conveyor              (2 lanes, one item type per lane, per-tile primitive)
- Splitter              (1x2, 50/50, direction-oriented)
- Direction and helpers

Design decisions locked in plan.md §"Locked decisions (2026-08-12)":
- No inserter entity in the sim (adjacency = auto-transfer).
- No electricity. Machines always powered.
- Only one recipe: green-science pack (logistic-science-pack).
  Real Factorio recipe: 1 transport-belt + 1 inserter -> 1 pack, time = 6s.
- Chest positions decided by the model each episode.
- Perpendicular same-tile conveyor crossings allowed (translate to distance-2
  underground pair in real Factorio).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Direction = Literal["north", "east", "south", "west"]

DIRECTIONS: tuple[Direction, ...] = ("north", "east", "south", "west")

DIR_DELTA: dict[Direction, tuple[int, int]] = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}

OPPOSITE: dict[Direction, Direction] = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}


def is_perpendicular(a: Direction, b: Direction) -> bool:
    return {a, b} in ({"north", "east"}, {"north", "west"},
                      {"south", "east"}, {"south", "west"})


ChestKind = Literal["input-belts", "input-inserters", "output-science"]

CHEST_KINDS: tuple[ChestKind, ...] = ("input-belts", "input-inserters",
                                       "output-science")

# Green-science pack (logistic-science-pack) inputs.
# Which chest emits which item.
CHEST_ITEM: dict[ChestKind, str | None] = {
    "input-belts": "transport-belt",
    "input-inserters": "inserter",
    "output-science": None,  # consumes green science, does not emit
}


AssemblerTier = Literal[1, 2, 3]

ASSEMBLER_TIERS: tuple[AssemblerTier, ...] = (1, 2, 3)


@dataclass(frozen=True)
class AssemblerSpec:
    tier: AssemblerTier
    size: tuple[int, int]                # (w, h)
    crafts_per_sec_green_science: float  # crafting_speed / recipe_time
    factorio_id: str                     # for translator


# Green science recipe time = 6 s (factoriolab).
# crafting_speed values from factoriolab: 0.5 / 0.75 / 1.25 for asm-1/2/3.
ASSEMBLERS: dict[AssemblerTier, AssemblerSpec] = {
    1: AssemblerSpec(tier=1, size=(3, 3),
                     crafts_per_sec_green_science=0.5 / 6.0,
                     factorio_id="assembling-machine-1"),
    2: AssemblerSpec(tier=2, size=(3, 3),
                     crafts_per_sec_green_science=0.75 / 6.0,
                     factorio_id="assembling-machine-2"),
    3: AssemblerSpec(tier=3, size=(3, 3),
                     crafts_per_sec_green_science=1.25 / 6.0,
                     factorio_id="assembling-machine-3"),
}


# Green-science recipe: 1 belt + 1 inserter -> 1 pack.
GREEN_SCIENCE_ITEM = "logistic-science-pack"
GREEN_SCIENCE_INPUTS: dict[str, int] = {
    "transport-belt": 1,
    "inserter": 1,
}


# Conveyor: 1x1 tile, has a direction (item flow-out direction), 2 lanes,
# and a tier (1/2/3 = yellow/red/blue in real Factorio).
# Each lane carries at most one item type. Two conveyors may share a tile
# provided their directions are perpendicular ("crossing"); translates 1:1
# to a distance-2 perpendicular underground in real Factorio.
CONVEYOR_LANES = 2

ConveyorTier = Literal[1, 2, 3]

CONVEYOR_TIERS: tuple[ConveyorTier, ...] = (1, 2, 3)

# Per-lane throughput cap, items/sec, per real Factorio.
CONVEYOR_LANE_CAPACITY: dict[ConveyorTier, float] = {
    1: 15.0,   # yellow (transport-belt)
    2: 30.0,   # red (fast-transport-belt)
    3: 45.0,   # blue (express-transport-belt)
}

# Factorio ids for translator.
CONVEYOR_FACTORIO_ID: dict[ConveyorTier, str] = {
    1: "transport-belt",
    2: "fast-transport-belt",
    3: "express-transport-belt",
}


@dataclass(frozen=True)
class ChestSpec:
    kind: ChestKind
    factorio_id: str          # for translator (infinity chest either mode)
    emits: str | None         # item name emitted (None if it consumes)


CHESTS: dict[ChestKind, ChestSpec] = {
    "input-belts": ChestSpec(kind="input-belts",
                              factorio_id="infinity-chest",
                              emits="transport-belt"),
    "input-inserters": ChestSpec(kind="input-inserters",
                                  factorio_id="infinity-chest",
                                  emits="inserter"),
    "output-science": ChestSpec(kind="output-science",
                                 factorio_id="infinity-chest",
                                 emits=None),
}
