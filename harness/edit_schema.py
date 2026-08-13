"""Pydantic schema for LLM edit outputs (simplified env).

Five edit types:
- place_chest      {op, kind, x, y, id}
- place_assembler  {op, tier, x, y, id}
- place_conveyor   {op, tier, x, y, direction, id}
- place_conveyor_line {op, tier, from_x, from_y, to_x, to_y, id}
- remove_entity    {op, id}

`parse_edit(dict)` normalizes a raw dict into a typed model. `edits_from_json`
in `edit_parser.py` chains parsing over an entire LLM response.
"""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ValidationError

from mini_factorio.entities import AssemblerTier, ChestKind, ConveyorTier, Direction


class PlaceChest(BaseModel):
    op: Literal["place_chest"] = "place_chest"
    id: str
    kind: ChestKind
    x: int
    y: int


class PlaceAssembler(BaseModel):
    op: Literal["place_assembler"] = "place_assembler"
    id: str
    tier: AssemblerTier
    x: int
    y: int


class PlaceConveyor(BaseModel):
    op: Literal["place_conveyor"] = "place_conveyor"
    id: str
    tier: ConveyorTier
    x: int
    y: int
    direction: Direction


class PlaceConveyorLine(BaseModel):
    op: Literal["place_conveyor_line"] = "place_conveyor_line"
    id: str
    tier: ConveyorTier
    from_x: int
    from_y: int
    to_x: int
    to_y: int


class RemoveEntity(BaseModel):
    op: Literal["remove_entity"] = "remove_entity"
    id: str


Edit = Union[PlaceChest, PlaceAssembler, PlaceConveyor, PlaceConveyorLine, RemoveEntity]

_EDIT_MODELS: dict[str, type[BaseModel]] = {
    "place_chest":     PlaceChest,
    "place_assembler": PlaceAssembler,
    "place_conveyor":  PlaceConveyor,
    "place_conveyor_line": PlaceConveyorLine,
    "remove_entity":   RemoveEntity,
}


def parse_edit(d: dict) -> tuple[Edit | None, str | None]:
    """Return (typed edit, None) on success, or (None, error_str) on failure."""
    if not isinstance(d, dict):
        return None, f"edit is not a dict: {type(d).__name__}"
    op = d.get("op")
    if op not in _EDIT_MODELS:
        return None, f"unknown op {op!r}; expected one of {sorted(_EDIT_MODELS)}"
    model = _EDIT_MODELS[op]
    try:
        return model.model_validate(d), None
    except ValidationError as e:
        return None, f"validation error for {op!r}: {e.errors()[0].get('msg', str(e))}"
