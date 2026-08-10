"""Mini-Factorio layout schema, validation, and JSON I/O.

Coordinate system: (x, y) with x increasing east, y increasing south (top-left origin).
Machines occupy a size = (w, h) square with top-left anchor (x, y). Belt tiles and
inserters are 1x1. Directions: 'north', 'east', 'south', 'west'. A belt tile's
direction is the item-flow direction on that tile; belt tiles must chain in that
direction. An inserter's direction is the drop direction — pickup is opposite.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from .entities import DIRECTIONS, MACHINES, PLACEABLES, RESOURCE_TYPES
from .recipes import RECIPES

Direction = Literal["north", "east", "south", "west"]

# Unit direction deltas: (dx, dy) added to a tile in that direction.
DIR_DELTA: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}
OPPOSITE: dict[str, str] = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}


class Resource(BaseModel):
    type: str
    x: int
    y: int
    size: int = Field(gt=0)


class Machine(BaseModel):
    id: str
    type: str  # miner: 'electric-mining-drill'; furnace: 'stone-furnace' |
               # 'steel-furnace' | 'electric-furnace'; assembler:
               # 'assembling-machine-1' | '...-2' | '...-3'
    x: int
    y: int
    # For miners, `direction` is the drop direction (facing) — determines
    # drop_position. For furnaces/assemblers it's currently cosmetic (Factorio
    # inserters can access any side of the footprint).
    direction: Direction = "north"
    recipe: str | None = None
    target_resource: str | None = None
    kind: Literal["machine"] = "machine"

    def drop_position(self) -> tuple[int, int]:
        """Tile where a mining drill deposits ore, per its facing direction.

        Meaningful only for miners; for other machine types the returned tile
        happens to be one tile outside the footprint but isn't used by the sim.
        Layout convention: top-left anchor, x east, y south.
        """
        w, h = MACHINES[self.type].size
        cx = self.x + w // 2
        cy = self.y + h // 2
        if self.direction == "north":
            return (cx, self.y - 1)
        if self.direction == "south":
            return (cx, self.y + h)
        if self.direction == "east":
            return (self.x + w, cy)
        return (self.x - 1, cy)  # west


class Inserter(BaseModel):
    id: str
    x: int
    y: int
    direction: Direction
    type: Literal["inserter", "fast-inserter", "stack-inserter"] = "inserter"
    kind: Literal["inserter"] = "inserter"


class BeltTile(BaseModel):
    x: int
    y: int
    direction: Direction


class Belt(BaseModel):
    id: str
    item: str
    tiles: list[BeltTile] = Field(min_length=1)
    # Belt tier controls per-tile throughput cap in the sim. Default yellow.
    type: Literal["transport-belt", "fast-transport-belt",
                  "express-transport-belt"] = "transport-belt"


class Layout(BaseModel):
    grid_size: tuple[int, int]  # (width, height)
    resources: list[Resource] = []
    budget: dict[str, int] = {}
    machines: list[Machine] = []
    inserters: list[Inserter] = []
    belts: list[Belt] = []

    # ------------------------- I/O -------------------------

    def to_json(self, *, indent: int | None = None) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "Layout":
        return cls.model_validate_json(s)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        return cls.model_validate(d)

    # ------------------------- Geometry helpers -------------------------

    def in_bounds(self, x: int, y: int) -> bool:
        w, h = self.grid_size
        return 0 <= x < w and 0 <= y < h

    def machine_footprint(self, m: Machine) -> list[tuple[int, int]]:
        w, h = MACHINES[m.type].size
        return [(m.x + dx, m.y + dy) for dx in range(w) for dy in range(h)]

    def resource_footprint(self, r: Resource) -> list[tuple[int, int]]:
        return [(r.x + dx, r.y + dy) for dx in range(r.size) for dy in range(r.size)]

    def occupied_tiles(self) -> dict[tuple[int, int], str]:
        """Map tile -> owning entity id. Excludes ore patches (mineable through)."""
        occ: dict[tuple[int, int], str] = {}
        for m in self.machines:
            for t in self.machine_footprint(m):
                occ[t] = m.id
        for i in self.inserters:
            occ[(i.x, i.y)] = i.id
        for b in self.belts:
            for bt in b.tiles:
                occ[(bt.x, bt.y)] = b.id
        return occ

    def resource_at(self, x: int, y: int) -> str | None:
        for r in self.resources:
            if (r.x <= x < r.x + r.size) and (r.y <= y < r.y + r.size):
                return r.type
        return None

    # ------------------------- Validation -------------------------

    def validate_layout(self) -> list[str]:
        """Return list of human-readable error strings. Empty = valid."""
        errs: list[str] = []
        w, h = self.grid_size
        if w <= 0 or h <= 0:
            errs.append(f"grid_size must be positive, got ({w}, {h})")
            return errs

        # Resources
        for r in self.resources:
            if r.type not in RESOURCE_TYPES:
                errs.append(f"resource {r.type!r}: unknown resource type")
            for t in self.resource_footprint(r):
                if not self.in_bounds(*t):
                    errs.append(f"resource at {(r.x, r.y)} size {r.size} out of bounds")
                    break

        # Machines: valid type, valid recipe/target, in-bounds, on-ore for miners
        seen_ids: set[str] = set()
        occ: dict[tuple[int, int], str] = {}

        for m in self.machines:
            if m.id in seen_ids:
                errs.append(f"duplicate entity id {m.id!r}")
                continue
            seen_ids.add(m.id)
            if m.type not in MACHINES:
                errs.append(f"machine {m.id}: unknown type {m.type!r}")
                continue
            spec = MACHINES[m.type]
            # Bounds
            for t in self.machine_footprint(m):
                if not self.in_bounds(*t):
                    errs.append(f"machine {m.id}: footprint out of bounds at {t}")
                    break
            # Collisions
            for t in self.machine_footprint(m):
                if t in occ:
                    errs.append(f"machine {m.id}: overlaps entity {occ[t]!r} at {t}")
                else:
                    occ[t] = m.id
            # Recipe / target checks
            kind = _machine_kind(m.type)
            if kind == "miner":
                if m.target_resource is None:
                    errs.append(f"miner {m.id}: missing target_resource")
                elif m.target_resource not in RESOURCE_TYPES:
                    errs.append(f"miner {m.id}: unknown target_resource {m.target_resource!r}")
                else:
                    # At least one footprint tile must sit on the target resource patch
                    hits = [t for t in self.machine_footprint(m)
                            if self.resource_at(*t) == m.target_resource]
                    if not hits:
                        errs.append(
                            f"miner {m.id}: footprint does not overlap any "
                            f"{m.target_resource!r} patch"
                        )
                if m.recipe is not None:
                    errs.append(f"miner {m.id}: recipe must be None (mining is implicit)")
            else:  # furnace or assembler
                if m.recipe is None:
                    errs.append(f"{kind} {m.id}: missing recipe")
                elif m.recipe not in RECIPES:
                    errs.append(f"{kind} {m.id}: unknown recipe {m.recipe!r}")
                else:
                    r = RECIPES[m.recipe]
                    if r.kind != kind:
                        errs.append(
                            f"{kind} {m.id}: recipe {m.recipe!r} runs on {r.kind!r}, "
                            f"not {kind!r}"
                        )
                if m.target_resource is not None:
                    errs.append(f"{kind} {m.id}: target_resource only valid for miners")
                if spec.fuel_category is not None and kind == "furnace":
                    # Fuel path is checked at simulator time (needs graph info)
                    pass

        # Inserters
        for i in self.inserters:
            if i.id in seen_ids:
                errs.append(f"duplicate entity id {i.id!r}")
                continue
            seen_ids.add(i.id)
            if i.direction not in DIRECTIONS:
                errs.append(f"inserter {i.id}: bad direction {i.direction!r}")
            t = (i.x, i.y)
            if not self.in_bounds(*t):
                errs.append(f"inserter {i.id}: out of bounds at {t}")
            elif t in occ:
                errs.append(f"inserter {i.id}: overlaps {occ[t]!r} at {t}")
            else:
                occ[t] = i.id

        # Belts
        for b in self.belts:
            if b.id in seen_ids:
                errs.append(f"duplicate entity id {b.id!r}")
                continue
            seen_ids.add(b.id)
            # Contiguity: for each tile, the next tile must be at (x,y)+DIR_DELTA[direction]
            prev = None
            for idx, bt in enumerate(b.tiles):
                if bt.direction not in DIRECTIONS:
                    errs.append(f"belt {b.id}[{idx}]: bad direction {bt.direction!r}")
                    continue
                t = (bt.x, bt.y)
                if not self.in_bounds(*t):
                    errs.append(f"belt {b.id}[{idx}]: out of bounds at {t}")
                if t in occ:
                    errs.append(f"belt {b.id}[{idx}]: overlaps {occ[t]!r} at {t}")
                else:
                    occ[t] = b.id
                if prev is not None:
                    pdx, pdy = DIR_DELTA[prev.direction]
                    expected = (prev.x + pdx, prev.y + pdy)
                    if t != expected:
                        errs.append(
                            f"belt {b.id}[{idx}]: not contiguous; expected {expected} "
                            f"after tile flowing {prev.direction!r}, got {t}"
                        )
                prev = bt

        return errs

    # ------------------------- Costs -------------------------

    def construction_cost(self) -> dict[str, float]:
        """Total materials to build every entity (using real factoriolab recipes)."""
        total: dict[str, float] = {}
        for m in self.machines:
            for item, amt in MACHINES[m.type].cost.items():
                total[item] = total.get(item, 0.0) + amt
        for i in self.inserters:
            for item, amt in PLACEABLES["inserter"].cost_per_place.items():
                total[item] = total.get(item, 0.0) + amt
        belt_tiles = sum(len(b.tiles) for b in self.belts)
        for item, amt in PLACEABLES["transport-belt"].cost_per_place.items():
            total[item] = total.get(item, 0.0) + amt * belt_tiles
        return total

    def total_materials_used(self) -> float:
        """Sum of all items consumed for construction — for reward composite term."""
        return sum(self.construction_cost().values())

    def total_cells_occupied(self) -> int:
        return len(self.occupied_tiles())

    def machine_count(self) -> int:
        """Machines only (miners + furnaces + assemblers).

        Inserters are placeables, not machines; they're already captured by
        the cell-count term of the reward. Historically this method returned
        machines+inserters, which caused γ·(machines+inserters) to over-penalize
        realistic layouts (~15 inserters × 0.05 = 0.75 extra penalty).
        """
        return len(self.machines)

    def entity_count(self) -> int:
        """All placed entities: machines + inserters + belt tiles.

        Not currently used by the reward, but exposed for diagnostics.
        """
        return len(self.machines) + len(self.inserters) + sum(len(b.tiles) for b in self.belts)


def _machine_kind(machine_type: str) -> str:
    """Map machine entity id → simulator kind (miner / furnace / assembler)."""
    if machine_type == "electric-mining-drill":
        return "miner"
    if machine_type in ("stone-furnace", "steel-furnace", "electric-furnace"):
        return "furnace"
    if machine_type in ("assembling-machine-1", "assembling-machine-2",
                       "assembling-machine-3"):
        return "assembler"
    raise ValueError(f"unknown machine type: {machine_type}")
