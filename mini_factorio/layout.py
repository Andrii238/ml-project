"""Layout schema, validation, and JSON I/O for the simplified env.

Coordinate system: (x, y). x increases east, y increases south (top-left origin).
Grid is 20x20 by default (see plan.md).

Entities:
- Chest         : 1x1, one of {input-belts, input-inserters, output-science}
- Assembler     : 3x3, tier in {1, 2, 3}
- Conveyor      : 1x1 tile, direction (item flow direction), 2 lanes

Same-tile perpendicular conveyor crossings are permitted. Any other same-tile
overlap is invalid.

Chest count rule: exactly one input-belts, one input-inserters, one
output-science chest per layout for well-formed episodes; validator flags
mismatch but does not raise (sim can still run on partial/invalid layouts,
reward reflects the issue).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .entities import (
    ASSEMBLERS,
    CHEST_KINDS,
    CONVEYOR_LANES,
    CONVEYOR_TIERS,
    DIR_DELTA,
    DIRECTIONS,
    Direction,
    ChestKind,
    AssemblerTier,
    ConveyorTier,
    is_perpendicular,
)


DEFAULT_GRID = (20, 20)


class Chest(BaseModel):
    id: str
    kind: ChestKind
    x: int
    y: int

    @property
    def footprint(self) -> list[tuple[int, int]]:
        return [(self.x, self.y)]


class Assembler(BaseModel):
    id: str
    tier: AssemblerTier
    x: int  # top-left anchor
    y: int

    @property
    def footprint(self) -> list[tuple[int, int]]:
        w, h = ASSEMBLERS[self.tier].size
        return [(self.x + dx, self.y + dy)
                for dx in range(w) for dy in range(h)]

    def border_tiles(self) -> list[tuple[int, int]]:
        """Tiles orthogonally adjacent to the footprint (12 tiles for 3x3)."""
        w, h = ASSEMBLERS[self.tier].size
        tiles: list[tuple[int, int]] = []
        for dx in range(w):
            tiles.append((self.x + dx, self.y - 1))
            tiles.append((self.x + dx, self.y + h))
        for dy in range(h):
            tiles.append((self.x - 1, self.y + dy))
            tiles.append((self.x + w, self.y + dy))
        return tiles


class Conveyor(BaseModel):
    id: str
    tier: ConveyorTier = 1
    x: int
    y: int
    direction: Direction

    @property
    def footprint(self) -> list[tuple[int, int]]:
        return [(self.x, self.y)]

    def downstream_tile(self) -> tuple[int, int]:
        dx, dy = DIR_DELTA[self.direction]
        return (self.x + dx, self.y + dy)

    def upstream_tile(self) -> tuple[int, int]:
        dx, dy = DIR_DELTA[self.direction]
        return (self.x - dx, self.y - dy)


class ChestRates(BaseModel):
    """Per-episode emission rates for the two input chests (items/sec)."""
    belts: float = 0.0
    inserters: float = 0.0


class Layout(BaseModel):
    grid_size: tuple[int, int] = DEFAULT_GRID
    chests: list[Chest] = []
    assemblers: list[Assembler] = []
    conveyors: list[Conveyor] = []
    chest_rates: ChestRates = Field(default_factory=ChestRates)

    # -------------------- I/O --------------------

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

    # -------------------- Geometry --------------------

    def in_bounds(self, x: int, y: int) -> bool:
        w, h = self.grid_size
        return 0 <= x < w and 0 <= y < h

    def all_entities(self):
        """Iterator over (id, footprint, category) tuples for all placed entities."""
        for c in self.chests:
            yield c.id, c.footprint, "chest"
        for a in self.assemblers:
            yield a.id, a.footprint, "assembler"
        for cv in self.conveyors:
            yield cv.id, cv.footprint, "conveyor"

    def occupied_tiles(self) -> dict[tuple[int, int], list[tuple[str, str]]]:
        """Map tile -> list of (entity_id, category) that occupy it.

        A tile normally has exactly one occupant. It may hold exactly two
        conveyors iff they are perpendicular (a same-tile crossing).
        """
        occ: dict[tuple[int, int], list[tuple[str, str]]] = {}
        for eid, tiles, cat in self.all_entities():
            for t in tiles:
                occ.setdefault(t, []).append((eid, cat))
        return occ

    # -------------------- Validation --------------------

    def validate_layout(self) -> list[str]:
        """Return list of human-readable error strings. Empty means valid."""
        errs: list[str] = []
        w, h = self.grid_size
        if w <= 0 or h <= 0:
            errs.append(f"grid_size must be positive, got ({w}, {h})")
            return errs

        # Chest kinds should each appear at most once, and each should appear
        # (well-formed episode has 1 of each). Report as errors.
        seen_kinds: dict[ChestKind, int] = {k: 0 for k in CHEST_KINDS}
        for c in self.chests:
            if c.kind not in CHEST_KINDS:
                errs.append(f"chest {c.id}: unknown kind {c.kind!r}")
                continue
            seen_kinds[c.kind] += 1
        for k, n in seen_kinds.items():
            if n == 0:
                errs.append(f"missing chest of kind {k!r}")
            elif n > 1:
                errs.append(f"more than one chest of kind {k!r} (found {n})")

        # ID uniqueness across all entities.
        seen_ids: set[str] = set()
        for eid, _tiles, _cat in self.all_entities():
            if eid in seen_ids:
                errs.append(f"duplicate entity id {eid!r}")
            seen_ids.add(eid)

        # Bounds per entity.
        for eid, tiles, cat in self.all_entities():
            for t in tiles:
                if not self.in_bounds(*t):
                    errs.append(f"{cat} {eid}: tile {t} out of bounds")
                    break

        # Direction sanity for conveyors.
        for cv in self.conveyors:
            if cv.direction not in DIRECTIONS:
                errs.append(f"conveyor {cv.id}: bad direction {cv.direction!r}")

        # Overlap check with the perpendicular-crossing exception.
        occ = self.occupied_tiles()
        for tile, occupants in occ.items():
            if len(occupants) == 1:
                continue
            if len(occupants) == 2:
                cats = {o[1] for o in occupants}
                if cats != {"conveyor"}:
                    errs.append(f"tile {tile}: two entities overlap ({occupants}); "
                                f"only perpendicular conveyors may share a tile")
                    continue
                dirs = [next(cv.direction for cv in self.conveyors
                             if cv.id == oid) for oid, _ in occupants]
                if not is_perpendicular(dirs[0], dirs[1]):
                    errs.append(f"tile {tile}: two conveyors overlap but are not "
                                f"perpendicular (directions={dirs})")
            else:
                errs.append(f"tile {tile}: {len(occupants)} entities overlap "
                            f"({occupants}); at most 2 (crossing) allowed")

        return errs

    # -------------------- Counters (used by the reward) --------------------

    def machine_count(self) -> int:
        return len(self.assemblers)

    def machine_count_by_tier(self) -> dict[AssemblerTier, int]:
        counts: dict[AssemblerTier, int] = {1: 0, 2: 0, 3: 0}
        for a in self.assemblers:
            counts[a.tier] += 1
        return counts

    def conveyor_count(self) -> int:
        return len(self.conveyors)

    def conveyor_count_by_tier(self) -> dict[ConveyorTier, int]:
        counts: dict[ConveyorTier, int] = {1: 0, 2: 0, 3: 0}
        for cv in self.conveyors:
            counts[cv.tier] += 1
        return counts

    def total_cells_occupied(self) -> int:
        # A same-tile crossing (2 conveyors sharing a tile) still counts as
        # one physical cell of area used. Callers that want cost per placeable
        # should use conveyor_count / splitter_count / machine_count instead.
        return len(self.occupied_tiles())
