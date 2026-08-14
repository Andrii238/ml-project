"""Translator tests for current simplified Layout -> FLE entity translation."""
from __future__ import annotations

import pytest

from mini_factorio.layout import Assembler, Chest, Conveyor, Layout
from translator.to_fle import FACTORIO_DIRECTION, translate


def _base_layout() -> Layout:
    return Layout(
        grid_size=(25, 25),
        chests=[
            Chest(id="out", kind="output-science", x=0, y=0),
            Chest(id="belts", kind="input-belts", x=2, y=0),
            Chest(id="ins", kind="input-inserters", x=3, y=0),
        ],
    )


def _entity_names(layout: Layout) -> list[str]:
    return [e.name for e in translate(layout).entities]


def test_directions_use_factorio_2_encoding():
    assert FACTORIO_DIRECTION == {"north": 0, "east": 4, "south": 8, "west": 12}


def test_chests_translate_to_infinity_chests():
    tr = translate(_base_layout())
    chest_entities = [e for e in tr.entities if e.name == "infinity-chest"]
    assert len(chest_entities) == 3
    assert all(e.infinity_settings is not None for e in chest_entities)


def test_assembler_translates_to_tier_machine_with_green_science_recipe():
    lay = _base_layout()
    lay.assemblers.append(Assembler(id="a", tier=2, x=5, y=6))
    tr = translate(lay)
    asm = next(e for e in tr.entities if e.name == "assembling-machine-2")
    assert asm.position == {"x": 6.5, "y": 7.5}
    assert asm.recipe == "logistic-science-pack"


def test_conveyor_direction_translates():
    lay = _base_layout()
    lay.conveyors.append(Conveyor(id="c", tier=1, x=5, y=5, direction="north"))
    tr = translate(lay)
    belt = next(e for e in tr.entities if e.name == "transport-belt")
    assert belt.direction == 0
    assert belt.position == {"x": 5.5, "y": 5.5}


def test_tiered_conveyors_translate_to_correct_factorio_ids():
    lay = _base_layout()
    lay.conveyors.extend([
        Conveyor(id="c1", tier=1, x=5, y=5, direction="east"),
        Conveyor(id="c2", tier=2, x=6, y=5, direction="east"),
        Conveyor(id="c3", tier=3, x=7, y=5, direction="east"),
    ])
    names = _entity_names(lay)
    assert "transport-belt" in names
    assert "fast-transport-belt" in names
    assert "express-transport-belt" in names


def test_machine_interface_conveyor_becomes_inserter():
    lay = _base_layout()
    lay.assemblers.append(Assembler(id="a", tier=1, x=5, y=5))
    # Downstream tile enters assembler footprint, so translator replaces this
    # interface conveyor with an inserter for real Factorio compatibility.
    lay.conveyors.append(Conveyor(id="feed", tier=1, x=4, y=5, direction="east"))
    tr = translate(lay)
    names = [e.name for e in tr.entities]
    assert "inserter" in names
    assert "transport-belt" not in names


def test_perpendicular_crossing_uses_underground_pair():
    lay = _base_layout()
    lay.conveyors.extend([
        Conveyor(id="h", tier=1, x=10, y=10, direction="east"),
        Conveyor(id="v", tier=1, x=10, y=10, direction="north"),
    ])
    tr = translate(lay)
    names = [e.name for e in tr.entities]
    assert names.count("transport-belt") == 1
    assert names.count("underground-belt") == 2
    roles = sorted(e.type for e in tr.entities if e.name == "underground-belt")
    assert roles == ["input", "output"]


def test_parallel_crossing_warns_and_emits_both_belts():
    lay = _base_layout()
    lay.conveyors.extend([
        Conveyor(id="a", tier=1, x=10, y=10, direction="east"),
        Conveyor(id="b", tier=1, x=10, y=10, direction="east"),
    ])
    tr = translate(lay)
    assert tr.warnings
    assert [e.name for e in tr.entities].count("transport-belt") == 2


def test_translation_entity_numbers_are_unique_and_1_indexed():
    lay = _base_layout()
    lay.assemblers.append(Assembler(id="a", tier=1, x=5, y=5))
    lay.conveyors.append(Conveyor(id="c", tier=1, x=10, y=10, direction="east"))
    tr = translate(lay)
    nums = [e.entity_number for e in tr.entities]
    assert nums == list(range(1, len(nums) + 1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
