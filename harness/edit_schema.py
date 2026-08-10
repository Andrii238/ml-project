"""Edit schema for LLM outputs (plan.md §Edit schema).

The LLM proposes a JSON list of edits. Each edit is one of six discriminated
variants below. `EditList` is the top-level container the parser targets.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

Direction = Literal["north", "east", "south", "west"]


class AddEntity(BaseModel):
    op: Literal["add_entity"] = "add_entity"
    id: str
    type: str  # 'electric-mining-drill' | 'stone-furnace' | 'assembling-machine-1'
    x: int
    y: int
    # Required for miners (drop_position depends on facing); cosmetic for
    # furnaces/assemblers. Optional so old completions still parse.
    direction: Direction | None = None
    recipe: str | None = None
    target_resource: str | None = None


class RemoveEntity(BaseModel):
    op: Literal["remove_entity"] = "remove_entity"
    id: str


class AddInserter(BaseModel):
    op: Literal["add_inserter"] = "add_inserter"
    id: str
    x: int
    y: int
    direction: Direction


class AddBelt(BaseModel):
    op: Literal["add_belt"] = "add_belt"
    id: str
    item: str
    tiles: list[tuple[int, int, Direction]] = Field(min_length=1)


class RemoveBelt(BaseModel):
    op: Literal["remove_belt"] = "remove_belt"
    id: str


class ExtendBelt(BaseModel):
    op: Literal["extend_belt"] = "extend_belt"
    id: str
    tiles: list[tuple[int, int, Direction]] = Field(min_length=1)


Edit = Annotated[
    Union[AddEntity, RemoveEntity, AddInserter, AddBelt, RemoveBelt, ExtendBelt],
    Field(discriminator="op"),
]


class EditList(BaseModel):
    edits: list[Edit] = []
