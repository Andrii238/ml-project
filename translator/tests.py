"""Tests for translator/to_fle.py.

Verify translation correctness (direction encoding, position centering, entity
naming, recipe attachment, power-grid insertion). Cannot verify FLE actually
accepts the commands without a Docker + Factorio server — plan.md §FLE
integration marks that as the follow-up validation gate.
"""
from __future__ import annotations

import json

import pytest

from mini_factorio.layout import Belt, BeltTile, Inserter, Layout, Machine, Resource
from translator.to_fle import (
    FACTORIO_DIRECTION,
    layout_to_blueprint_dict,
    layout_to_lua_commands,
)


# ---------- Direction encoding ----------


def test_directions_use_factorio_2_encoding():
    assert FACTORIO_DIRECTION == {"north": 0, "east": 4, "south": 8, "west": 12}


# ---------- Position centering (top-left + size/2) ----------


def test_miner_3x3_position_center():
    lay = Layout(
        grid_size=(16, 16),
        resources=[Resource(type="iron-ore", x=0, y=0, size=3)],
        machines=[Machine(id="m", type="electric-mining-drill", x=0, y=0,
                          target_resource="iron-ore")],
    )
    cmds = layout_to_lua_commands(lay, add_power=False)
    miner = next(c for c in cmds if "electric-mining-drill" in c)
    assert "position={1.5, 1.5}" in miner


def test_furnace_2x2_position_center():
    lay = Layout(
        grid_size=(16, 16),
        machines=[Machine(id="f", type="stone-furnace", x=4, y=6, recipe="iron-plate")],
    )
    cmds = layout_to_lua_commands(lay, add_power=False)
    f = next(c for c in cmds if "stone-furnace" in c)
    # 2x2 at (4,6) → center (5, 7)
    assert "position={5, 7}" in f


def test_inserter_1x1_position_and_direction():
    """Our schema: direction = drop direction. Factorio: direction = pickup.
    So our 'east' (drop east, pickup west) → Factorio 'west' = 12.
    Verified live via LuaInserter.pickup_position read-back."""
    lay = Layout(
        grid_size=(16, 16),
        inserters=[Inserter(id="i", x=3, y=4, direction="east")],
    )
    cmds = layout_to_lua_commands(lay, add_power=False)
    ins = next(c for c in cmds if "inserter" in c and "position=" in c)
    assert "position={3.5, 4.5}" in ins
    assert "direction=12" in ins  # opposite of our 'east'


def test_belt_direction_north_maps_to_zero():
    lay = Layout(
        grid_size=(16, 16),
        belts=[Belt(id="b", item="iron-ore",
                    tiles=[BeltTile(x=5, y=5, direction="north")])],
    )
    cmds = layout_to_lua_commands(lay, add_power=False)
    belt = next(c for c in cmds if "transport-belt" in c)
    assert "direction=0" in belt


# ---------- One command per tile ----------


def test_belt_produces_one_command_per_tile():
    lay = Layout(
        grid_size=(16, 16),
        belts=[Belt(id="b", item="iron-ore", tiles=[
            BeltTile(x=0, y=0, direction="east"),
            BeltTile(x=1, y=0, direction="east"),
            BeltTile(x=2, y=0, direction="east"),
        ])],
    )
    cmds = layout_to_lua_commands(lay, add_power=False)
    belt_cmds = [c for c in cmds if "transport-belt" in c]
    assert len(belt_cmds) == 3


def test_resource_patch_produces_one_command_per_tile():
    lay = Layout(
        grid_size=(16, 16),
        resources=[Resource(type="iron-ore", x=0, y=0, size=3)],
    )
    cmds = layout_to_lua_commands(lay, add_power=False)
    ore_cmds = [c for c in cmds if "name='iron-ore'" in c]
    assert len(ore_cmds) == 9  # 3×3 = 9 tiles
    # Ore tiles must be neutral force (not player) so miners can mine them.
    assert all("force='neutral'" in c for c in ore_cmds)
    assert all(f"amount={100_000}" in c for c in ore_cmds)


# ---------- Recipe attachment ----------


def test_recipe_applied_to_assembler():
    lay = Layout(
        grid_size=(16, 16),
        machines=[Machine(id="a", type="assembling-machine-1", x=0, y=0,
                          recipe="iron-gear-wheel")],
    )
    cmds = layout_to_lua_commands(lay, add_power=False)
    a = next(c for c in cmds if "assembling-machine-1" in c)
    assert "set_recipe('iron-gear-wheel')" in a


def test_miner_has_no_recipe_set():
    lay = Layout(
        grid_size=(16, 16),
        resources=[Resource(type="iron-ore", x=0, y=0, size=3)],
        machines=[Machine(id="m", type="electric-mining-drill", x=0, y=0,
                          target_resource="iron-ore")],
    )
    cmds = layout_to_lua_commands(lay, add_power=False)
    miner = next(c for c in cmds if "electric-mining-drill" in c)
    assert "set_recipe" not in miner


def test_furnace_has_no_recipe_set():
    """Factorio furnaces smelt whatever ore is inserted; set_recipe raises
    'Entity is not assembling-machine' if called on them (real FLE error)."""
    lay = Layout(
        grid_size=(16, 16),
        machines=[Machine(id="f", type="stone-furnace", x=0, y=0, recipe="iron-plate")],
    )
    cmds = layout_to_lua_commands(lay, add_power=False)
    f = next(c for c in cmds if "stone-furnace" in c)
    assert "set_recipe" not in f


# ---------- Command ordering ----------


def test_clear_surface_is_first_command():
    lay = Layout(grid_size=(16, 16))
    cmds = layout_to_lua_commands(lay)
    assert "destroy" in cmds[0]


def test_command_order_power_then_resources_then_machines():
    lay = Layout(
        grid_size=(16, 16),
        resources=[Resource(type="iron-ore", x=8, y=8, size=2)],
        machines=[Machine(id="m", type="electric-mining-drill", x=8, y=8,
                          target_resource="iron-ore")],
    )
    cmds = layout_to_lua_commands(lay, add_power=True)
    joined = "|".join(cmds)
    sub_idx = joined.find("substation")
    ore_idx = joined.find("name='iron-ore'")
    miner_idx = joined.find("electric-mining-drill")
    assert sub_idx < ore_idx < miner_idx


# ---------- Power grid ----------


def test_power_grid_placed_when_requested():
    lay = Layout(grid_size=(16, 16))
    cmds = layout_to_lua_commands(lay, add_power=True)
    assert any("substation" in c for c in cmds)
    assert any("electric-energy-interface" in c for c in cmds)


def test_power_grid_skipped_when_disabled():
    lay = Layout(grid_size=(16, 16))
    cmds = layout_to_lua_commands(lay, add_power=False)
    assert not any("substation" in c for c in cmds)


def test_power_grid_avoids_occupied_tiles():
    # Fill center with a machine so substation must fall back to a corner.
    lay = Layout(
        grid_size=(16, 16),
        machines=[Machine(id="a", type="assembling-machine-1", x=7, y=7,
                          recipe="iron-gear-wheel")],
    )
    cmds = layout_to_lua_commands(lay, add_power=True)
    sub_cmd = next(c for c in cmds if "substation" in c and "position" in c)
    # Substation position must not sit on the machine (which occupies 7-9 × 7-9)
    assert "position={8, 8}" not in sub_cmd


# ---------- Blueprint dict ----------


def test_blueprint_dict_is_json_serializable():
    lay = Layout(
        grid_size=(16, 16),
        machines=[Machine(id="a", type="assembling-machine-1", x=0, y=0,
                          recipe="iron-gear-wheel")],
    )
    bp = layout_to_blueprint_dict(lay)
    s = json.dumps(bp)
    assert "iron-gear-wheel" in s
    assert "logistic-science-pack" in s  # blueprint icon


def test_blueprint_inserter_direction_uses_pickup_convention():
    """Blueprint dict for inserters follows Factorio's pickup-direction convention,
    same as the Lua path (verified live)."""
    lay = Layout(
        grid_size=(16, 16),
        inserters=[Inserter(id="i", x=3, y=4, direction="east")],
    )
    bp = layout_to_blueprint_dict(lay)
    ins = next(e for e in bp["blueprint"]["entities"] if e["name"] == "inserter")
    assert ins["direction"] == 12  # our 'east' → Factorio 'west' = 12


def test_blueprint_entity_count_matches_layout():
    lay = Layout(
        grid_size=(16, 16),
        resources=[Resource(type="iron-ore", x=0, y=0, size=3)],  # not in blueprint
        machines=[
            Machine(id="m", type="electric-mining-drill", x=0, y=0,
                    target_resource="iron-ore"),
            Machine(id="a", type="assembling-machine-1", x=5, y=5,
                    recipe="iron-gear-wheel"),
        ],
        inserters=[Inserter(id="i", x=4, y=4, direction="east")],
        belts=[Belt(id="b", item="iron-ore", tiles=[
            BeltTile(x=3, y=3, direction="east"),
            BeltTile(x=4, y=3, direction="east"),
        ])],
    )
    bp = layout_to_blueprint_dict(lay)
    ents = bp["blueprint"]["entities"]
    # 2 machines + 1 inserter + 2 belt tiles = 5 entities
    assert len(ents) == 5
    # Entity numbers are 1-indexed and unique
    nums = [e["entity_number"] for e in ents]
    assert nums == sorted(nums) == list(range(1, len(ents) + 1))


def test_blueprint_recipe_attached_to_assembler():
    lay = Layout(
        grid_size=(16, 16),
        machines=[Machine(id="a", type="assembling-machine-1", x=0, y=0,
                          recipe="logistic-science-pack")],
    )
    bp = layout_to_blueprint_dict(lay)
    asm = next(e for e in bp["blueprint"]["entities"] if e["name"] == "assembling-machine-1")
    assert asm["recipe"] == "logistic-science-pack"


# ---------- End-to-end sanity ----------


# ---------- FLE driver: import safety + batch math ----------


def test_fle_driver_imports_without_fle_installed():
    """fle_driver.py must import cleanly even when factorio-learning-environment
    is not installed — it defers the fle import until a function is called."""
    import translator.fle_driver as d  # noqa: F401
    # Sanity: the API exists
    assert hasattr(d, "smoke_test")
    assert hasattr(d, "validate_and_measure")
    assert hasattr(d, "cross_check")
    assert hasattr(d, "summarize")


def test_fle_driver_raises_helpful_error_without_rcon():
    """Calling a live function without factorio-rcon-py installed →
    ImportError with next-step message."""
    import translator.fle_driver as d
    try:
        import factorio_rcon  # noqa: F401
    except ImportError:
        # factorio-rcon-py is not installed — this is the case we want to test
        with pytest.raises(ImportError, match="factorio-rcon-py"):
            d.smoke_test()
    # If factorio-rcon-py IS installed we skip (would try to connect).


def test_pearson_r_perfect_correlation():
    from translator.fle_driver import _pearson_r
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0, 4.0, 6.0, 8.0]
    assert _pearson_r(xs, ys) == pytest.approx(1.0)


def test_pearson_r_zero_variance_returns_none():
    from translator.fle_driver import _pearson_r
    assert _pearson_r([1, 1, 1], [1, 2, 3]) is None
    assert _pearson_r([], []) is None


def test_pearson_r_negative_correlation():
    from translator.fle_driver import _pearson_r
    r = _pearson_r([1, 2, 3], [3, 2, 1])
    assert r == pytest.approx(-1.0)


def test_mape_zero_error():
    from translator.fle_driver import _mape
    assert _mape([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)


def test_mape_ignores_zero_truth():
    from translator.fle_driver import _mape
    # y=0 is skipped from MAPE (undefined division)
    assert _mape([1.0], [0.0]) is None
    assert _mape([1.0, 2.0], [0.0, 2.0]) == pytest.approx(0.0)


def test_summarize_ship_gates_all_pass():
    from translator.fle_driver import FLEValidation, summarize
    results = [
        FLEValidation("a", True, [], fle_rate=1.0, sim_rate=1.0),
        FLEValidation("b", True, [], fle_rate=2.0, sim_rate=2.0),
        FLEValidation("c", True, [], fle_rate=3.0, sim_rate=3.0),
    ]
    report = summarize(results)
    assert report.build_success_rate == 1.0
    assert report.pearson_r == pytest.approx(1.0)
    assert report.mape == pytest.approx(0.0)
    assert report.ship_gates == {
        "build_success_100pct": True,
        "pearson_r_ge_0_9": True,
        "mape_le_0_20": True,
    }


def test_summarize_ship_gates_build_failure():
    from translator.fle_driver import FLEValidation, summarize
    results = [
        FLEValidation("a", True, [], fle_rate=1.0, sim_rate=1.0),
        FLEValidation("b", False, ["cmd[3]: some error"], sim_rate=2.0),
    ]
    report = summarize(results)
    assert report.build_success_rate == 0.5
    assert report.ship_gates["build_success_100pct"] is False


def test_summarize_ship_gates_mape_over_threshold():
    from translator.fle_driver import FLEValidation, summarize
    # sim off by 30% consistently: perfect Pearson but MAPE 0.3 > 0.2
    results = [
        FLEValidation(f"L{i}", True, [], fle_rate=v, sim_rate=v * 1.3)
        for i, v in enumerate([1.0, 2.0, 3.0, 4.0])
    ]
    report = summarize(results)
    assert report.ship_gates["pearson_r_ge_0_9"] is True
    assert report.ship_gates["mape_le_0_20"] is False


# ---------- End-to-end sanity ----------


def test_full_handcrafted_layout_translates_cleanly():
    lay = Layout(
        grid_size=(16, 16),
        resources=[
            Resource(type="iron-ore", x=5, y=1, size=3),
            Resource(type="coal", x=9, y=4, size=3),
        ],
        machines=[
            Machine(id="mi", type="electric-mining-drill", x=5, y=1,
                    target_resource="iron-ore"),
            Machine(id="mc", type="electric-mining-drill", x=9, y=4,
                    target_resource="coal"),
            Machine(id="f", type="stone-furnace", x=9, y=1, recipe="iron-plate"),
        ],
        inserters=[
            Inserter(id="i1", x=8, y=1, direction="east"),
            Inserter(id="ic", x=9, y=3, direction="north"),
        ],
    )
    cmds = layout_to_lua_commands(lay)
    assert all(c.strip() for c in cmds), "no empty commands"
    # Blueprint round-trip through JSON
    bp = layout_to_blueprint_dict(lay)
    json.loads(json.dumps(bp))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
