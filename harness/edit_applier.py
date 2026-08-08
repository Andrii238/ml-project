"""Apply an EditList to a Layout, one edit at a time.

Each edit is validated in isolation before being applied. A failing edit is
skipped (with an error string recorded) but subsequent edits still run against
the layout as it stands. This matches plan.md §Edit schema:

    "Each edit is validated independently. On failure, that specific edit is
     rejected with a clear error string; other edits in the list still apply."

Validation strategy: build a candidate layout, run `Layout.validate_layout()`,
and roll back to the previous state if the result is invalid. This piggybacks on
the same rules the simulator uses.
"""
from __future__ import annotations

from dataclasses import dataclass

from mini_factorio.layout import Belt, BeltTile, Inserter, Layout, Machine

from .edit_schema import (
    AddBelt,
    AddEntity,
    AddInserter,
    Edit,
    EditList,
    ExtendBelt,
    RemoveBelt,
    RemoveEntity,
)


@dataclass
class ApplyResult:
    layout: Layout
    errors: list[str]  # per-edit error strings; empty on the ones that applied cleanly
    n_applied: int


def _clone(layout: Layout) -> Layout:
    return Layout.model_validate(layout.model_dump())


def _validate_diff(before: Layout, after: Layout) -> str | None:
    """Return the first validation error introduced by the change, or None."""
    before_errs = set(before.validate_layout())
    after_errs = [e for e in after.validate_layout() if e not in before_errs]
    return after_errs[0] if after_errs else None


def _apply_one(layout: Layout, edit: Edit) -> tuple[Layout, str | None]:
    candidate = _clone(layout)
    if isinstance(edit, AddEntity):
        if any(m.id == edit.id for m in candidate.machines):
            return layout, f"add_entity {edit.id}: id already exists"
        candidate.machines.append(Machine(
            id=edit.id, type=edit.type, x=edit.x, y=edit.y,
            recipe=edit.recipe, target_resource=edit.target_resource,
        ))
    elif isinstance(edit, AddInserter):
        if any(i.id == edit.id for i in candidate.inserters):
            return layout, f"add_inserter {edit.id}: id already exists"
        candidate.inserters.append(Inserter(
            id=edit.id, x=edit.x, y=edit.y, direction=edit.direction,
        ))
    elif isinstance(edit, AddBelt):
        if any(b.id == edit.id for b in candidate.belts):
            return layout, f"add_belt {edit.id}: id already exists"
        candidate.belts.append(Belt(
            id=edit.id, item=edit.item,
            tiles=[BeltTile(x=t[0], y=t[1], direction=t[2]) for t in edit.tiles],
        ))
    elif isinstance(edit, ExtendBelt):
        belt = next((b for b in candidate.belts if b.id == edit.id), None)
        if belt is None:
            return layout, f"extend_belt {edit.id}: belt not found"
        belt.tiles.extend(
            BeltTile(x=t[0], y=t[1], direction=t[2]) for t in edit.tiles
        )
    elif isinstance(edit, RemoveEntity):
        n0 = len(candidate.machines) + len(candidate.inserters)
        candidate.machines = [m for m in candidate.machines if m.id != edit.id]
        candidate.inserters = [i for i in candidate.inserters if i.id != edit.id]
        if len(candidate.machines) + len(candidate.inserters) == n0:
            return layout, f"remove_entity {edit.id}: not found"
    elif isinstance(edit, RemoveBelt):
        n0 = len(candidate.belts)
        candidate.belts = [b for b in candidate.belts if b.id != edit.id]
        if len(candidate.belts) == n0:
            return layout, f"remove_belt {edit.id}: not found"
    else:  # unreachable given the discriminated union
        return layout, f"unknown edit op: {type(edit).__name__}"

    err = _validate_diff(layout, candidate)
    if err is not None:
        return layout, f"{edit.op}: rejected — {err}"
    return candidate, None


def apply_edits(layout: Layout, edits: EditList) -> ApplyResult:
    current = _clone(layout)
    errors: list[str] = []
    applied = 0
    for edit in edits.edits:
        current, err = _apply_one(current, edit)
        if err is None:
            applied += 1
            errors.append("")
        else:
            errors.append(err)
    return ApplyResult(layout=current, errors=errors, n_applied=applied)
