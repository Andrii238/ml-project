"""Apply a list of typed edits to a Layout.

Each edit is applied in isolation:
- If the edit would produce an invalid layout (out-of-bounds, ID collision,
  overlap not covered by the perpendicular-crossing rule, etc.), the edit is
  rejected and the layout is left unchanged for that step.
- Rejections are collected in `errors`; successful edits are counted.

Rules enforced per edit:
- `place_chest`: id unique, tile in bounds and unoccupied.
- `place_assembler`: id unique, 3x3 footprint in bounds and non-overlapping.
- `place_conveyor`: id unique, tile in bounds. Overlap allowed only if the
  existing occupant is exactly one conveyor with a perpendicular direction.
- `remove_entity`: id must reference an existing chest / assembler / conveyor.

The applier does NOT enforce layout-level rules like "exactly one chest per
kind" — those are surfaced by `layout.validate_layout()` at reward time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mini_factorio.entities import ASSEMBLERS, is_perpendicular
from mini_factorio.layout import (
    Assembler,
    Chest,
    Conveyor,
    Layout,
)

from .edit_schema import (
    Edit,
    PlaceAssembler,
    PlaceChest,
    PlaceConveyor,
    RemoveEntity,
)


@dataclass
class ApplyResult:
    layout: Layout                                       # final state (mutated copy)
    applied: int = 0                                     # number of successful edits
    errors: list[str] = field(default_factory=list)     # per-edit error strings


# --------------------------------------------------------------- helpers

def _all_ids(lay: Layout) -> set[str]:
    ids: set[str] = set()
    for c in lay.chests:
        ids.add(c.id)
    for a in lay.assemblers:
        ids.add(a.id)
    for cv in lay.conveyors:
        ids.add(cv.id)
    return ids


def _in_bounds(lay: Layout, x: int, y: int) -> bool:
    return lay.in_bounds(x, y)


def _occupants_at(lay: Layout, tile: tuple[int, int]) -> list[tuple[str, str]]:
    """List of (entity_id, category) currently on `tile`."""
    out: list[tuple[str, str]] = []
    for c in lay.chests:
        if (c.x, c.y) == tile:
            out.append((c.id, "chest"))
    for a in lay.assemblers:
        if tile in a.footprint:
            out.append((a.id, "assembler"))
    for cv in lay.conveyors:
        if (cv.x, cv.y) == tile:
            out.append((cv.id, "conveyor"))
    return out


# --------------------------------------------------------------- per-edit

def _apply_place_chest(lay: Layout, e: PlaceChest) -> str | None:
    if e.id in _all_ids(lay):
        return f"place_chest id {e.id!r}: duplicate"
    if not _in_bounds(lay, e.x, e.y):
        return f"place_chest {e.id!r}: tile ({e.x},{e.y}) out of bounds"
    if _occupants_at(lay, (e.x, e.y)):
        return f"place_chest {e.id!r}: tile ({e.x},{e.y}) occupied"
    lay.chests.append(Chest(id=e.id, kind=e.kind, x=e.x, y=e.y))
    return None


def _apply_place_assembler(lay: Layout, e: PlaceAssembler) -> str | None:
    if e.id in _all_ids(lay):
        return f"place_assembler {e.id!r}: duplicate id"
    w, h = ASSEMBLERS[e.tier].size
    footprint = [(e.x + dx, e.y + dy) for dx in range(w) for dy in range(h)]
    for t in footprint:
        if not _in_bounds(lay, *t):
            return f"place_assembler {e.id!r}: footprint tile {t} out of bounds"
    for t in footprint:
        if _occupants_at(lay, t):
            return f"place_assembler {e.id!r}: tile {t} occupied"
    lay.assemblers.append(Assembler(id=e.id, tier=e.tier, x=e.x, y=e.y))
    return None


def _apply_place_conveyor(lay: Layout, e: PlaceConveyor) -> str | None:
    if e.id in _all_ids(lay):
        return f"place_conveyor {e.id!r}: duplicate id"
    if not _in_bounds(lay, e.x, e.y):
        return f"place_conveyor {e.id!r}: tile ({e.x},{e.y}) out of bounds"
    occ = _occupants_at(lay, (e.x, e.y))
    if occ:
        # Only allowed: exactly one existing conveyor with a perpendicular direction.
        if len(occ) != 1 or occ[0][1] != "conveyor":
            return (f"place_conveyor {e.id!r}: tile ({e.x},{e.y}) occupied by "
                    f"{occ}; only a perpendicular conveyor may share a tile")
        other = next(c for c in lay.conveyors if c.id == occ[0][0])
        if not is_perpendicular(other.direction, e.direction):
            return (f"place_conveyor {e.id!r}: existing conveyor {other.id!r} "
                    f"at ({e.x},{e.y}) is direction {other.direction!r}; "
                    f"crossing requires perpendicular, got {e.direction!r}")
    lay.conveyors.append(Conveyor(id=e.id, tier=e.tier, x=e.x, y=e.y,
                                    direction=e.direction))
    return None


def _apply_remove_entity(lay: Layout, e: RemoveEntity) -> str | None:
    before = len(lay.chests) + len(lay.assemblers) + len(lay.conveyors)
    lay.chests = [c for c in lay.chests if c.id != e.id]
    lay.assemblers = [a for a in lay.assemblers if a.id != e.id]
    lay.conveyors = [cv for cv in lay.conveyors if cv.id != e.id]
    after = len(lay.chests) + len(lay.assemblers) + len(lay.conveyors)
    if after == before:
        return f"remove_entity {e.id!r}: no such entity"
    return None


# --------------------------------------------------------------- driver

_DISPATCH = {
    PlaceChest:     _apply_place_chest,
    PlaceAssembler: _apply_place_assembler,
    PlaceConveyor:  _apply_place_conveyor,
    RemoveEntity:   _apply_remove_entity,
}


def apply_edits(layout: Layout, edits: list[Edit]) -> ApplyResult:
    lay = Layout.from_dict(layout.to_dict())  # deep copy via serialization
    result = ApplyResult(layout=lay)
    for i, edit in enumerate(edits):
        fn = _DISPATCH.get(type(edit))
        if fn is None:
            result.errors.append(f"edit[{i}]: unhandled edit type {type(edit).__name__}")
            continue
        err = fn(lay, edit)
        if err is None:
            result.applied += 1
        else:
            result.errors.append(f"edit[{i}]: {err}")
    return result
