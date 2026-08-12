"""Translate a simplified-env Layout into a Factorio-compatible entity list.

Handles three jobs:

1. Entity emission:
   - Chest -> `infinity-chest` (input chests carry filters with their emitted
     item; output chest empty).
   - Assembler -> `assembling-machine-{tier}` with recipe `logistic-science-pack`.
   - Conveyor -> `transport-belt` / `fast-transport-belt` /
     `express-transport-belt` per tier.

2. Same-tile perpendicular crossings -> one straight belt plus a
   `underground-belt` (entry + exit) pair on the perpendicular axis. The
   entry sits one tile upstream of the crossing, the exit one tile
   downstream. If the neighboring tiles already contain conveyors of the
   same direction, those are removed to make room for entry / exit.

3. Inserter injection between every assembler and its adjacent conveyors.
   Real Factorio requires a 1-tile gap between machine and belt for the
   inserter. Our sim allows them touching, so the translator shifts the
   interface conveyor 1 tile away from the machine and cascades the shift
   along the connected conveyor chain until it hits an obstacle or grid
   edge (grid may grow beyond 20x20 as needed).

Direction encoding used (Factorio 2.0): 0=N, 4=E, 8=S, 12=W.

The result is a plain dict; wrap into a real Factorio blueprint format at
call sites that need it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from mini_factorio.entities import (
    ASSEMBLERS,
    CHESTS,
    CONVEYOR_FACTORIO_ID,
    DIR_DELTA,
    GREEN_SCIENCE_ITEM,
    OPPOSITE,
    is_perpendicular,
)
from mini_factorio.layout import Assembler, Chest, Conveyor, Layout


# -------------- direction encoding --------------

FACTORIO_DIRECTION = {
    "north": 0,
    "east":  4,
    "south": 8,
    "west":  12,
}


# -------------- output structures --------------

@dataclass
class TranslatedEntity:
    entity_number: int
    name: str
    position: dict[str, float]           # {"x": float, "y": float} — tile center
    direction: int | None = None
    recipe: str | None = None
    type: str | None = None              # underground-belt: "input" | "output"
    infinity_settings: dict | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "entity_number": self.entity_number,
            "name": self.name,
            "position": self.position,
        }
        if self.direction is not None:
            d["direction"] = self.direction
        if self.recipe is not None:
            d["recipe"] = self.recipe
        if self.type is not None:
            d["type"] = self.type
        if self.infinity_settings is not None:
            d["infinity_settings"] = self.infinity_settings
        return d


@dataclass
class TranslationResult:
    entities: list[TranslatedEntity] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    grid_size: tuple[int, int] = (0, 0)   # final grid, may exceed layout.grid_size

    def as_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "warnings": list(self.warnings),
            "grid_size": list(self.grid_size),
        }


# -------------- helpers --------------

def _tile_center(x: int, y: int, w: int = 1, h: int = 1) -> dict[str, float]:
    return {"x": x + w / 2.0, "y": y + h / 2.0}


def _cascade_shift(conveyors: list[Conveyor], seed_id: str,
                    shift_delta: tuple[int, int],
                    id_to_cv: dict[str, Conveyor],
                    tile_to_cvs: dict[tuple[int, int], list[Conveyor]]) -> set[str]:
    """Shift `seed_id` by `shift_delta` and cascade the shift to any conveyor
    that was directly upstream, until no further cascade is needed.
    Mutates `conveyors` in place (positions updated). Returns set of shifted ids."""
    shifted: set[str] = set()
    queue = [seed_id]
    while queue:
        cid = queue.pop(0)
        if cid in shifted:
            continue
        cv = id_to_cv[cid]
        old_tile = (cv.x, cv.y)
        cv.x += shift_delta[0]
        cv.y += shift_delta[1]
        shifted.add(cid)
        # Anything at cv's OLD position that fed cv previously must also shift.
        # That was: conveyor at old_tile - DIR_DELTA[cv.direction] with downstream = old_tile.
        # Since positions changed, find upstream by original layout:
        upstream_tile = (old_tile[0] - DIR_DELTA[cv.direction][0],
                          old_tile[1] - DIR_DELTA[cv.direction][1])
        for cv_up in tile_to_cvs.get(upstream_tile, []):
            if cv_up.id in shifted:
                continue
            if (cv_up.x + DIR_DELTA[cv_up.direction][0],
                cv_up.y + DIR_DELTA[cv_up.direction][1]) == old_tile:
                # cv_up used to feed cv. Cascade.
                queue.append(cv_up.id)
    return shifted


# -------------- entity builders --------------

def _translate_chest(chest: Chest, entity_number: int) -> TranslatedEntity:
    """Chest -> infinity-chest. All chests start with no filter (empty). The
    driver inserts items into input chests each game-second to enforce the
    sim's per-second emission rate. The output chest simply accumulates
    green-science delivered to it."""
    infinity_settings: dict = {"remove_unfiltered_items": False, "filters": []}
    return TranslatedEntity(
        entity_number=entity_number,
        name=CHESTS[chest.kind].factorio_id,
        position=_tile_center(chest.x, chest.y, 1, 1),
        infinity_settings=infinity_settings,
    )


def _translate_assembler(a: Assembler, entity_number: int) -> TranslatedEntity:
    w, h = ASSEMBLERS[a.tier].size
    return TranslatedEntity(
        entity_number=entity_number,
        name=ASSEMBLERS[a.tier].factorio_id,
        position=_tile_center(a.x, a.y, w, h),
        recipe=GREEN_SCIENCE_ITEM,
    )


def _translate_belt(cv: Conveyor, entity_number: int) -> TranslatedEntity:
    return TranslatedEntity(
        entity_number=entity_number,
        name=CONVEYOR_FACTORIO_ID[cv.tier],
        position=_tile_center(cv.x, cv.y, 1, 1),
        direction=FACTORIO_DIRECTION[cv.direction],
    )


def _translate_underground(cv: Conveyor, entity_number: int,
                             role: str, tile: tuple[int, int]) -> TranslatedEntity:
    """`role` in {"input", "output"}."""
    name = CONVEYOR_FACTORIO_ID[cv.tier].replace("transport-belt", "underground-belt")
    # yellow -> "underground-belt"; red -> "fast-underground-belt";
    # blue -> "express-underground-belt". The replace above handles that.
    return TranslatedEntity(
        entity_number=entity_number,
        name=name,
        position=_tile_center(tile[0], tile[1], 1, 1),
        direction=FACTORIO_DIRECTION[cv.direction],
        type=role,
    )


def _make_inserter(pos: tuple[int, int], pickup_direction: str,
                     entity_number: int) -> TranslatedEntity:
    """Emit a basic inserter at `pos`. `pickup_direction` is the direction from
    which the inserter picks up items. Real Factorio uses direction = pickup
    direction (per FLE_NOTES)."""
    return TranslatedEntity(
        entity_number=entity_number,
        name="inserter",
        position=_tile_center(pos[0], pos[1], 1, 1),
        direction=FACTORIO_DIRECTION[pickup_direction],
    )


# -------------- main --------------

def translate(layout: Layout) -> TranslationResult:
    result = TranslationResult()
    ids = itertools.count(1)

    # Deep-copy the layout — we mutate conveyor positions during grid expansion.
    lay = Layout.from_dict(layout.to_dict())

    # ---- Job 2: interface conveyors -> inserters ----
    # Rule: any conveyor whose downstream OR upstream is on a machine footprint
    # is an interface conveyor. In real Factorio the transfer is done by an
    # inserter, not a belt. We REPLACE the conveyor with an inserter at the
    # same tile (no shift, no grid expansion, no chest overlap).
    # The inserter's pickup direction points at whatever the sim treated as
    # the source (chest, upstream belt, or the machine itself for output).
    interface_events: list[dict] = []
    interface_ids: set[str] = set()

    for asm in lay.assemblers:
        fp = set(asm.footprint)
        for cv in lay.conveyors:
            if cv.id in interface_ids:
                continue
            if cv.downstream_tile() in fp:
                # cv delivers to machine. Inserter picks from where cv used to
                # get its input (its upstream_tile), drops on machine.
                # Pickup direction = OPPOSITE of cv.direction (looking back
                # from cv toward its upstream).
                interface_events.append({
                    "tile": (cv.x, cv.y),
                    "pickup": OPPOSITE[cv.direction],
                })
                interface_ids.add(cv.id)
            elif cv.upstream_tile() in fp:
                # cv receives from machine. Inserter picks from machine
                # (upstream_tile of cv, which lies on the footprint), drops
                # on cv's downstream tile.
                # Pickup direction = OPPOSITE of cv.direction (toward machine).
                interface_events.append({
                    "tile": (cv.x, cv.y),
                    "pickup": OPPOSITE[cv.direction],
                })
                interface_ids.add(cv.id)

    # Drop the interface conveyors from the layout so we don't emit them as belts.
    lay.conveyors = [cv for cv in lay.conveyors if cv.id not in interface_ids]

    # ---- Compute expanded grid size ----
    w, h = lay.grid_size
    max_x = max((cv.x for cv in lay.conveyors), default=w - 1)
    max_y = max((cv.y for cv in lay.conveyors), default=h - 1)
    min_x = min((cv.x for cv in lay.conveyors), default=0)
    min_y = min((cv.y for cv in lay.conveyors), default=0)
    if min_x < 0 or min_y < 0:
        # Shift everything into non-negative coords.
        dx = max(0, -min_x)
        dy = max(0, -min_y)
        for cv in lay.conveyors:
            cv.x += dx
            cv.y += dy
        for a in lay.assemblers:
            a.x += dx
            a.y += dy
        for c in lay.chests:
            c.x += dx
            c.y += dy
        max_x += dx
        max_y += dy
    grid_w = max(w, max_x + 1)
    grid_h = max(h, max_y + 1)
    result.grid_size = (grid_w, grid_h)

    # ---- Job 1: emit chests + assemblers ----
    for chest in lay.chests:
        result.entities.append(_translate_chest(chest, next(ids)))
    for asm in lay.assemblers:
        result.entities.append(_translate_assembler(asm, next(ids)))

    # ---- Job 3: crossings -> undergrounds ----
    tile_conveyors: dict[tuple[int, int], list[Conveyor]] = {}
    for cv in lay.conveyors:
        tile_conveyors.setdefault((cv.x, cv.y), []).append(cv)

    processed: set[str] = set()
    for tile, cvs in tile_conveyors.items():
        if len(cvs) == 1:
            result.entities.append(_translate_belt(cvs[0], next(ids)))
            processed.add(cvs[0].id)
        elif len(cvs) == 2:
            cv_a, cv_b = cvs
            if not is_perpendicular(cv_a.direction, cv_b.direction):
                result.warnings.append(
                    f"tile {tile}: two parallel conveyors; layout invalid, "
                    f"emitting both as overlapping belts"
                )
                result.entities.append(_translate_belt(cv_a, next(ids)))
                result.entities.append(_translate_belt(cv_b, next(ids)))
                processed.update({cv_a.id, cv_b.id})
                continue
            # Choose the surface belt vs the undergrounded belt. Convention:
            # keep the horizontal one as belt if one is horizontal, else keep cv_a.
            if cv_a.direction in ("east", "west"):
                cv_belt, cv_under = cv_a, cv_b
            else:
                cv_belt, cv_under = cv_b, cv_a
            # Belt at the crossing tile.
            result.entities.append(_translate_belt(cv_belt, next(ids)))
            processed.add(cv_belt.id)
            # Underground entry: one tile upstream of the crossing along cv_under.
            entry_tile = (tile[0] - DIR_DELTA[cv_under.direction][0],
                          tile[1] - DIR_DELTA[cv_under.direction][1])
            exit_tile  = (tile[0] + DIR_DELTA[cv_under.direction][0],
                          tile[1] + DIR_DELTA[cv_under.direction][1])
            # Emit entry + exit. If a conveyor exists at entry/exit tile with
            # the same direction, mark it processed so we don't double-emit.
            for other in tile_conveyors.get(entry_tile, []):
                if other.direction == cv_under.direction:
                    processed.add(other.id)
            for other in tile_conveyors.get(exit_tile, []):
                if other.direction == cv_under.direction:
                    processed.add(other.id)
            result.entities.append(_translate_underground(
                cv_under, next(ids), role="input", tile=entry_tile))
            result.entities.append(_translate_underground(
                cv_under, next(ids), role="output", tile=exit_tile))
            processed.add(cv_under.id)
            # Underground extends the effective grid — update grid_size.
            for t in (entry_tile, exit_tile):
                if t[0] + 1 > result.grid_size[0]:
                    result.grid_size = (t[0] + 1, result.grid_size[1])
                if t[1] + 1 > result.grid_size[1]:
                    result.grid_size = (result.grid_size[0], t[1] + 1)
        else:
            result.warnings.append(f"tile {tile}: {len(cvs)} conveyors; skipped")

    # Emit any conveyors that weren't handled above (shouldn't happen normally).
    for cv in lay.conveyors:
        if cv.id not in processed:
            result.entities.append(_translate_belt(cv, next(ids)))

    # ---- Job 2 continued: emit inserters at each recorded interface tile ----
    for ev in interface_events:
        result.entities.append(_make_inserter(ev["tile"], ev["pickup"], next(ids)))

    return result
