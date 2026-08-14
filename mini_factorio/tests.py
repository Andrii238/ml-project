"""Tests for the simplified env — Chunk 1 (data types).

Covers entities.py + layout.py: construction, JSON round-trip, validation
rules (bounds, ID uniqueness, overlap, perpendicular-crossing exception,
chest kinds).

Run: python -m mini_factorio.tests
"""
from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable

from .entities import (
    ASSEMBLERS,
    CHEST_ITEM,
    CONVEYOR_LANE_CAPACITY,
    GREEN_SCIENCE_INPUTS,
    GREEN_SCIENCE_ITEM,
    is_perpendicular,
)
from .layout import (
    Assembler,
    Chest,
    ChestRates,
    Conveyor,
    DEFAULT_GRID,
    Layout,
)
from .simulator import simulate
from .reward import DEFAULT_CONFIG, RewardConfig, compute_reward
from harness.prompt_builder import (
    build_chat_messages,
    build_user_message,
    render_entity_list,
    render_grid,
)
from harness.edit_applier import apply_edits
from harness.edit_parser import parse_edits
from harness.edit_schema import (
    PlaceAssembler,
    PlaceChest,
    PlaceConveyor,
    PlaceConveyorLine,
    RemoveEntity,
    parse_edit,
)
from translator.to_fle import FACTORIO_DIRECTION, translate
from .random_layouts import (
    CHEST_RATE_BUCKETS,
    CHEST_RATE_JITTER,
    empty_episode,
    partial_episode,
    sample_chest_rate,
    sample_chest_rates,
    sample_episodes,
)
import random as _random


# ---------------------------------------------------------------- helpers

def _empty_layout() -> Layout:
    return Layout()


def _minimal_layout() -> Layout:
    """One chest of each kind + one asm-1 + no conveyors. Not simulable but
    passes structural validation (all chest kinds present, no overlaps)."""
    return Layout(
        chests=[
            Chest(id="c_in_b", kind="input-belts", x=0, y=0),
            Chest(id="c_in_i", kind="input-inserters", x=0, y=1),
            Chest(id="c_out", kind="output-science", x=0, y=2),
        ],
        assemblers=[Assembler(id="a1", tier=1, x=10, y=10)],
    )


# ---------------------------------------------------------------- test suite

def test_entities_green_science_rates_match_plan():
    got = {t: ASSEMBLERS[t].crafts_per_sec_green_science for t in (1, 2, 3)}
    assert abs(got[1] - 0.5 / 6) < 1e-9
    assert abs(got[2] - 0.75 / 6) < 1e-9
    assert abs(got[3] - 1.25 / 6) < 1e-9
    assert abs(got[1] - 0.0833) < 0.001
    assert abs(got[2] - 0.125) < 0.001
    assert abs(got[3] - 0.208) < 0.001


def test_green_science_recipe_shape():
    assert GREEN_SCIENCE_ITEM == "logistic-science-pack"
    assert GREEN_SCIENCE_INPUTS == {"transport-belt": 1, "inserter": 1}


def test_chest_items_map_correctly():
    assert CHEST_ITEM["input-belts"] == "transport-belt"
    assert CHEST_ITEM["input-inserters"] == "inserter"
    assert CHEST_ITEM["output-science"] is None


def test_perpendicular_helper():
    assert is_perpendicular("north", "east")
    assert is_perpendicular("north", "west")
    assert is_perpendicular("south", "east")
    assert is_perpendicular("south", "west")
    assert not is_perpendicular("north", "south")
    assert not is_perpendicular("east", "west")
    assert not is_perpendicular("north", "north")


def test_lane_capacity_by_tier():
    assert CONVEYOR_LANE_CAPACITY[1] == 15.0
    assert CONVEYOR_LANE_CAPACITY[2] == 30.0
    assert CONVEYOR_LANE_CAPACITY[3] == 45.0


def test_default_grid_25x25():
    assert _empty_layout().grid_size == DEFAULT_GRID == (25, 25)


def test_assembler_footprint_3x3():
    a = Assembler(id="a", tier=2, x=5, y=6)
    fp = set(a.footprint)
    expected = {(x, y) for x in range(5, 8) for y in range(6, 9)}
    assert fp == expected


def test_assembler_border_has_12_tiles():
    a = Assembler(id="a", tier=1, x=5, y=6)
    border = a.border_tiles()
    assert len(border) == 12
    fp = set(a.footprint)
    assert not (set(border) & fp)


def test_conveyor_downstream_and_upstream():
    cv = Conveyor(id="c", x=5, y=5, direction="east")
    assert cv.downstream_tile() == (6, 5)
    assert cv.upstream_tile() == (4, 5)


def test_minimal_layout_validates():
    errs = _minimal_layout().validate_layout()
    assert errs == [], errs


def test_empty_layout_flags_missing_chests():
    errs = _empty_layout().validate_layout()
    assert any("missing chest of kind 'input-belts'" in e for e in errs)
    assert any("missing chest of kind 'input-inserters'" in e for e in errs)
    assert any("missing chest of kind 'output-science'" in e for e in errs)


def test_duplicate_chest_kind_flagged():
    lay = _minimal_layout()
    lay.chests.append(Chest(id="c_extra", kind="input-belts", x=5, y=5))
    errs = lay.validate_layout()
    assert any("more than one chest of kind 'input-belts'" in e for e in errs)


def test_out_of_bounds_flagged():
    lay = _minimal_layout()
    lay.conveyors.append(Conveyor(id="cv_bad", x=25, y=5, direction="east"))
    errs = lay.validate_layout()
    assert any("cv_bad" in e and "out of bounds" in e for e in errs)


def test_duplicate_id_flagged():
    lay = _minimal_layout()
    lay.conveyors.append(Conveyor(id="a1", x=5, y=5, direction="east"))
    errs = lay.validate_layout()
    assert any("duplicate entity id 'a1'" in e for e in errs)


def test_overlap_non_conveyor_flagged():
    lay = _minimal_layout()
    lay.assemblers.append(Assembler(id="a_bad", tier=1, x=0, y=0))
    errs = lay.validate_layout()
    assert any("overlap" in e.lower() for e in errs)


def test_perpendicular_conveyor_crossing_allowed():
    lay = _minimal_layout()
    lay.conveyors.extend([
        Conveyor(id="cvE", x=8, y=8, direction="east"),
        Conveyor(id="cvN", x=8, y=8, direction="north"),
    ])
    errs = lay.validate_layout()
    assert errs == [], errs


def test_parallel_conveyors_same_tile_rejected():
    lay = _minimal_layout()
    lay.conveyors.extend([
        Conveyor(id="cvE1", x=8, y=8, direction="east"),
        Conveyor(id="cvE2", x=8, y=8, direction="east"),
    ])
    errs = lay.validate_layout()
    assert any("perpendicular" in e for e in errs), errs


def test_three_conveyors_same_tile_rejected():
    lay = _minimal_layout()
    lay.conveyors.extend([
        Conveyor(id="cv1", x=8, y=8, direction="east"),
        Conveyor(id="cv2", x=8, y=8, direction="north"),
        Conveyor(id="cv3", x=8, y=8, direction="south"),
    ])
    errs = lay.validate_layout()
    assert any("at most 2" in e for e in errs), errs


def test_chest_and_conveyor_overlap_rejected():
    lay = _minimal_layout()
    lay.conveyors.append(Conveyor(id="cv", x=0, y=0, direction="east"))
    errs = lay.validate_layout()
    assert any("overlap" in e.lower() for e in errs)


def test_json_round_trip_preserves_layout():
    lay = _minimal_layout()
    lay.conveyors.append(Conveyor(id="cv", x=5, y=5, direction="east"))
    lay.chest_rates = ChestRates(belts=1.5, inserters=2.5)
    s = lay.to_json()
    back = Layout.from_json(s)
    assert back.to_dict() == lay.to_dict()


def test_json_is_valid_json():
    lay = _minimal_layout()
    json.loads(lay.to_json())


def test_counters_by_tier():
    lay = _minimal_layout()
    lay.assemblers.extend([
        Assembler(id="a2", tier=2, x=5, y=5),
        Assembler(id="a2b", tier=2, x=5, y=10),
        Assembler(id="a3", tier=3, x=10, y=5),
    ])
    counts = lay.machine_count_by_tier()
    assert counts == {1: 1, 2: 2, 3: 1}
    assert lay.machine_count() == 4


def test_crossing_counts_as_one_cell():
    lay = _minimal_layout()
    lay.conveyors.extend([
        Conveyor(id="cvE", x=8, y=8, direction="east"),
        Conveyor(id="cvN", x=8, y=8, direction="north"),
    ])
    assert lay.total_cells_occupied() == 3 + 9 + 1
    assert lay.conveyor_count() == 2


# ================================================================
# Chunk 2 — simulator tests
# ================================================================


def _simple_chain_layout(tier: int = 1, belts_rate: float = 5.0,
                          inserters_rate: float = 5.0) -> Layout:
    """Assembler at (7,4), footprint (7..9, 4..6). Chests hug the assembler.

    - Belts chest at (5,4). Conveyor at (6,4) east → downstream (7,4) on
      footprint. Feeds belts to assembler.
    - Inserters chest at (5,6). Conveyor at (6,6) east → downstream (7,6) on
      footprint. Feeds inserters to assembler.
    - Output conveyor at (10,5) east — upstream (9,5) on footprint, downstream
      (11,5). Output-science chest at (11,5).
    """
    lay = Layout(chest_rates=ChestRates(belts=belts_rate, inserters=inserters_rate))
    lay.chests = [
        Chest(id="cb", kind="input-belts",     x=5, y=4),
        Chest(id="ci", kind="input-inserters", x=5, y=6),
        Chest(id="co", kind="output-science",  x=11, y=5),
    ]
    lay.assemblers = [Assembler(id="a", tier=tier, x=7, y=4)]
    lay.conveyors = [
        Conveyor(id="cv_b", x=6, y=4, direction="east"),
        Conveyor(id="cv_i", x=6, y=6, direction="east"),
        Conveyor(id="cv_o", x=10, y=5, direction="east"),
    ]
    return lay


def test_simulator_empty_layout_zero_science():
    lay = Layout()
    res = simulate(lay)
    assert res.green_science_rate == 0.0


def test_simulator_single_asm_saturated_supply_asm1():
    # asm-1 crafts 0.0833/s. Supply 5/s of each — assembler bottlenecked by crafting.
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    errs = lay.validate_layout()
    assert errs == [], errs
    res = simulate(lay)
    # Expected: asm-1 produces at crafting rate ≈ 0.0833. All delivered to output.
    assert abs(res.green_science_rate - 0.5 / 6) < 1e-6, res.green_science_rate


def test_simulator_single_asm_saturated_supply_asm3():
    lay = _simple_chain_layout(tier=3, belts_rate=5.0, inserters_rate=5.0)
    res = simulate(lay)
    assert abs(res.green_science_rate - 1.25 / 6) < 1e-6, res.green_science_rate


def test_simulator_assembler_consumes_from_adjacent_side_conveyors():
    lay = Layout(chest_rates=ChestRates(belts=10.0, inserters=10.0))
    lay.chests = [
        Chest(id="cb", kind="input-belts", x=0, y=10),
        Chest(id="ci", kind="input-inserters", x=0, y=12),
        Chest(id="co", kind="output-science", x=8, y=11),
    ]
    lay.assemblers = [Assembler(id="a", tier=1, x=3, y=10)]
    lay.conveyors = [
        Conveyor(id="b1", x=1, y=10, direction="east"),
        Conveyor(id="b2", x=2, y=10, direction="east"),
        Conveyor(id="i1", x=1, y=12, direction="east"),
        Conveyor(id="i2", x=2, y=12, direction="east"),
        Conveyor(id="o1", x=6, y=11, direction="east"),
        Conveyor(id="o2", x=7, y=11, direction="east"),
    ]
    res = simulate(lay)
    assert res.warnings == []
    assert abs(res.green_science_rate - 0.5 / 6) < 1e-6, res.green_science_rate


def test_simulator_assembler_outputs_to_adjacent_science_conveyor():
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    lay.chests = [c for c in lay.chests if c.kind != "output-science"]
    lay.chests.append(Chest(id="co", kind="output-science", x=6, y=7))
    lay.conveyors = [c for c in lay.conveyors if c.id != "cv_o"]
    # This conveyor is adjacent to the assembler but does not point away from
    # its footprint. New rule: adjacent empty/science conveyors can receive output.
    lay.conveyors.append(Conveyor(id="cv_o", x=7, y=7, direction="west"))
    res = simulate(lay)
    assert abs(res.green_science_rate - 0.5 / 6) < 1e-6, res.green_science_rate


def test_simulator_single_asm_bottlenecked_by_belts():
    # Belts supply 0.05/s, inserters 5/s. Output = 0.05/s (bottleneck).
    lay = _simple_chain_layout(tier=3, belts_rate=0.05, inserters_rate=5.0)
    res = simulate(lay)
    assert abs(res.green_science_rate - 0.05) < 1e-6, res.green_science_rate


def test_simulator_single_asm_bottlenecked_by_inserters():
    lay = _simple_chain_layout(tier=3, belts_rate=5.0, inserters_rate=0.02)
    res = simulate(lay)
    assert abs(res.green_science_rate - 0.02) < 1e-6, res.green_science_rate


def test_simulator_missing_output_chest_zero():
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    lay.chests = [c for c in lay.chests if c.kind != "output-science"]
    res = simulate(lay)
    assert res.green_science_rate == 0.0


def test_simulator_no_output_path_no_delivery():
    # Remove the output conveyor: assembler produces but nothing flows to output chest.
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    lay.conveyors = [c for c in lay.conveyors if c.id != "cv_o"]
    res = simulate(lay)
    assert res.green_science_rate == 0.0
    # But the assembler internally produced.
    mf = [m for m in res.machine_flows if m.id == "a"][0]
    assert mf.science_out > 0


def test_simulator_zero_belts_rate_zero_output():
    lay = _simple_chain_layout(tier=1, belts_rate=0.0, inserters_rate=5.0)
    res = simulate(lay)
    assert res.green_science_rate == 0.0


def test_simulator_perpendicular_crossing_passes_flow():
    """A crossing tile (two perpendicular conveyors sharing a tile) doesn't
    interfere with either flow. Belts pass east across a crossing while a
    separate conveyor going south sits on the same tile.
    """
    # Assembler at (7,4), footprint (7..9, 4..6).
    # Belts chest at (0,4), belts chain east along y=4.
    # Inserters chest at (5,8), conveyor (5,7) north → (5,6) north → belts row.
    # Then we want to route inserters east to reach the assembler.
    # Route: (5,7) north, (5,6) north (stops at (5,5) but that's not on
    # assembler footprint since it's x=5). Turn east: (5,6) east... conflicts.
    # Actually route through a crossing: (5,7) north → (5,6) east → (6,6) east →
    # downstream (7,6) on assembler footprint ✓. But (5,6) needs to face east
    # AND be receiving from (5,7) which is south of it.
    # For (5,6) east: upstream is (4,6), not (5,7). So (5,7) north doesn't feed
    # (5,6) east — the chain is broken.
    # Simpler: chest at (7,7), conveyor (7,6) north → upstream (7,7)=chest ✓,
    # downstream (7,5) on footprint ✓. Direct delivery.
    # Assembler at (7,4), footprint = {7..9} x {4..6}. Border tile (8,7) is
    # outside footprint. Place inserters chest at (8,8), conveyor at (8,7)
    # north → downstream (8,6) on footprint ✓.
    lay = Layout(chest_rates=ChestRates(belts=5.0, inserters=5.0))
    lay.chests = [
        Chest(id="cb", kind="input-belts",     x=0, y=4),
        Chest(id="ci", kind="input-inserters", x=8, y=8),
        Chest(id="co", kind="output-science",  x=11, y=5),
    ]
    lay.assemblers = [Assembler(id="a", tier=1, x=7, y=4)]
    lay.conveyors = [
        # Belts chain
        Conveyor(id="cb1", x=1, y=4, direction="east"),
        Conveyor(id="cb2", x=2, y=4, direction="east"),
        Conveyor(id="cb3", x=3, y=4, direction="east"),
        Conveyor(id="cb4", x=4, y=4, direction="east"),
        Conveyor(id="cb5", x=5, y=4, direction="east"),
        Conveyor(id="cb6", x=6, y=4, direction="east"),
        # Inserters
        Conveyor(id="ci1", x=8, y=7, direction="north"),
        # Crossing on (4,4): a north-going conveyor sharing the tile with cb4
        # (east). Carries nothing (no upstream source feeds it) but must not
        # interfere with the east flow.
        Conveyor(id="cx", x=4, y=4, direction="north"),
        # Output
        Conveyor(id="cv_o", x=10, y=5, direction="east"),
    ]
    errs = lay.validate_layout()
    assert errs == [], errs
    res = simulate(lay)
    assert res.green_science_rate > 0, res.green_science_rate
    # cb6 should carry the full 5.0/s of belts (the crossing did not steal it).
    assert abs(res.conveyor_items.get((6, 4), {}).get("transport-belt", 0.0) - 5.0) < 1e-6


# ================================================================
# Chunk 3 — reward tests
# ================================================================


def _no_random_config() -> RewardConfig:
    cfg = RewardConfig()
    cfg.enable_random_bonus = False
    return cfg


def test_reward_empty_layout_hits_do_nothing_and_missing_chests():
    lay = Layout()
    br = compute_reward(lay, config=_no_random_config())
    # do-nothing = -30. Missing 3 chests * -10 = -30. Total = -60.
    assert br.do_nothing == -30.0
    assert br.chest_missing == -30.0
    assert br.total == -60.0


def test_reward_missing_only_output_chest():
    lay = Layout()
    lay.chests = [
        Chest(id="cb", kind="input-belts",     x=0, y=0),
        Chest(id="ci", kind="input-inserters", x=0, y=1),
    ]
    lay.assemblers = [Assembler(id="a", tier=1, x=10, y=10)]
    br = compute_reward(lay, config=_no_random_config())
    # do-nothing = 0 (asm present). chest_missing = -10 (only output missing).
    assert br.do_nothing == 0.0
    assert br.chest_missing == -10.0


def test_reward_full_chain_asm1_produces_and_delivers():
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    cfg = _no_random_config()
    br = compute_reward(lay, config=cfg)
    # asm-1: crafts_per_sec = 0.5/6 ≈ 0.0833.
    expected = (
        cfg.milestone_delivers_any
        + cfg.delivered_reward * (0.5 / 6.0)
        + cfg.milestone_has_belts
        + cfg.milestone_has_inserters
        + cfg.milestone_is_producing
        - cfg.asm_cost[1]
        - cfg.conv_cost[1] * 3
    )
    assert abs(br.total - expected) < 1e-3, (br.total, expected)


def test_reward_asm3_triggers_tier_unlock():
    lay = _simple_chain_layout(tier=3, belts_rate=5.0, inserters_rate=5.0)
    cfg = _no_random_config()
    br = compute_reward(lay, config=cfg)
    expected = (
        cfg.milestone_delivers_any
        + cfg.delivered_reward * (1.25 / 6.0)
        + cfg.milestone_has_belts
        + cfg.milestone_has_inserters
        + cfg.milestone_is_producing
        - cfg.asm_cost[3]
        - cfg.conv_cost[1] * 3
        - cfg.asm_tier3_unlock
    )
    assert abs(br.total - expected) < 1e-3, (br.total, expected)


def test_reward_conveyor_tier_unlock_fires():
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    # Bump one conveyor to T2 (red).
    lay.conveyors[0].tier = 2
    cfg = _no_random_config()
    br = compute_reward(lay, config=cfg)
    baseline = compute_reward(_simple_chain_layout(tier=1, belts_rate=5.0,
                                                     inserters_rate=5.0),
                                config=cfg).total
    net_penalty = cfg.conv_tier2_unlock + (cfg.conv_cost[2] - cfg.conv_cost[1])
    assert abs(br.total - (baseline - net_penalty)) < 1e-3


def test_reward_random_bonus_is_deterministic():
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    a = compute_reward(lay).random_bonus
    b = compute_reward(lay).random_bonus
    assert a == b


def test_reward_random_bonus_zero_when_disabled():
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    br = compute_reward(lay, config=_no_random_config())
    assert br.random_bonus == 0.0


def test_reward_random_bonus_positive_when_enabled():
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    cfg = RewardConfig(enable_random_bonus=True)
    br = compute_reward(lay, config=cfg)
    assert br.random_bonus > 0.0


def test_reward_produced_but_not_delivered_partial():
    # Remove the output conveyor; asm produces but nothing reaches the chest.
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    lay.conveyors = [c for c in lay.conveyors if c.id != "cv_o"]
    br = compute_reward(lay, config=_no_random_config())
    # delivered = 0, but produced_partial = 5 * 0.0833.
    assert br.milestone_delivered == 0.0
    assert br.delivered == 0.0
    assert br.produced_partial > 0


def test_reward_one_time_delivery_bonus():
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    br = compute_reward(lay, config=_no_random_config())
    assert br.milestone_delivered == 20.0


def test_reward_breakdown_sums_to_total():
    lay = _simple_chain_layout(tier=2, belts_rate=5.0, inserters_rate=5.0)
    br = compute_reward(lay)
    parts = (br.do_nothing + br.chest_missing + br.milestone_belts
             + br.milestone_inserters + br.milestone_producing + br.delivered
             + br.milestone_delivered + br.produced_partial + br.machine_cost
             + br.conveyor_cost + br.asm_tier_unlock + br.conv_tier_unlock
             + br.random_bonus)
    assert abs(parts - br.total) < 1e-9


def test_reward_config_can_swap_coefficients():
    lay = Layout()  # empty
    cfg = RewardConfig(do_nothing_penalty=100.0, chest_missing_penalty=0.0,
                        enable_random_bonus=False)
    br = compute_reward(lay, config=cfg)
    assert br.total == -100.0


# ================================================================
# Chunk 4 — random layouts / episode setup tests
# ================================================================


def test_chest_rate_distribution_within_bounds():
    rng = _random.Random(0)
    samples = [sample_chest_rate(rng) for _ in range(1000)]
    lo = min(CHEST_RATE_BUCKETS) * (1.0 - CHEST_RATE_JITTER)
    hi = max(CHEST_RATE_BUCKETS) * (1.0 + CHEST_RATE_JITTER)
    for s in samples:
        assert lo <= s <= hi, s


def test_chest_rate_distribution_uses_all_buckets():
    """Bucketed curriculum should cover low, medium, and high flow regimes."""
    rng = _random.Random(0)
    N = 10_000
    samples = [sample_chest_rate(rng) for _ in range(N)]
    assert any(s < 1.0 for s in samples)
    assert any(3.5 < s < 4.5 for s in samples)
    assert any(s > 10.0 for s in samples)


def test_empty_episode_has_three_chests_no_other_entities():
    lay = empty_episode(seed=42)
    assert len(lay.chests) == 3
    assert len(lay.assemblers) == 0
    assert len(lay.conveyors) == 0
    kinds = {c.kind for c in lay.chests}
    assert kinds == {"input-belts", "input-inserters", "output-science"}
    positions = {c.kind: (c.x, c.y) for c in lay.chests}
    assert positions == {
        "output-science": (0, 0),
        "input-belts": (2, 0),
        "input-inserters": (3, 0),
    }


def test_empty_episode_validates():
    for s in range(20):
        lay = empty_episode(seed=s)
        errs = lay.validate_layout()
        assert errs == [], (s, errs)


def test_partial_episode_validates():
    for s in range(20):
        lay = partial_episode(seed=s)
        errs = lay.validate_layout()
        assert errs == [], (s, errs)


def test_100_random_episodes_all_valid():
    for lay in sample_episodes(100, mode="empty"):
        assert lay.validate_layout() == []
    for lay in sample_episodes(100, mode="partial"):
        assert lay.validate_layout() == []


def test_same_seed_same_layout():
    a = empty_episode(seed=7)
    b = empty_episode(seed=7)
    assert a.to_json() == b.to_json()


def test_different_seeds_differ():
    a = empty_episode(seed=1)
    b = empty_episode(seed=2)
    assert a.to_json() != b.to_json()


def test_chest_rates_sampled_per_episode():
    lay = empty_episode(seed=0)
    # Rates should be within bounds.
    hi = max(CHEST_RATE_BUCKETS) * (1.0 + CHEST_RATE_JITTER)
    assert 0.0 <= lay.chest_rates.belts <= hi
    assert 0.0 <= lay.chest_rates.inserters <= hi
    assert lay.chest_rates.belts == lay.chest_rates.inserters


# ================================================================
# Chunk 5 — prompt builder tests
# ================================================================


def test_render_grid_dimensions_match_layout():
    lay = _minimal_layout()
    grid = render_grid(lay)
    lines = grid.split("\n")
    w, h = lay.grid_size
    assert len(lines) == h
    for line in lines:
        assert len(line) == w


def test_render_grid_shows_chests_by_char():
    lay = _minimal_layout()  # cb=(0,0) input-belts, ci=(0,1), co=(0,2)
    grid = render_grid(lay)
    lines = grid.split("\n")
    assert lines[0][0] == "B"
    assert lines[1][0] == "I"
    assert lines[2][0] == "O"


def test_render_grid_shows_assembler_tier():
    lay = _minimal_layout()  # asm at (10, 10), tier 1
    grid = render_grid(lay)
    lines = grid.split("\n")
    for y in range(10, 13):
        for x in range(10, 13):
            assert lines[y][x] == "1"


def test_render_grid_shows_conveyor_direction():
    lay = _minimal_layout()
    lay.conveyors.append(Conveyor(id="cv", x=5, y=5, direction="east"))
    grid = render_grid(lay)
    lines = grid.split("\n")
    assert lines[5][5] == ">"


def test_render_grid_crossing_shows_plus():
    lay = _minimal_layout()
    lay.conveyors.append(Conveyor(id="cvE", x=5, y=5, direction="east"))
    lay.conveyors.append(Conveyor(id="cvN", x=5, y=5, direction="north"))
    grid = render_grid(lay)
    lines = grid.split("\n")
    assert lines[5][5] == "+"


def test_render_entity_list_includes_all_entities():
    lay = _minimal_layout()
    lay.conveyors.append(Conveyor(id="cv", x=5, y=5, direction="east", tier=2))
    s = render_entity_list(lay)
    assert '"kind":"input-belts"' in s
    assert '"kind":"input-inserters"' in s
    assert '"kind":"output-science"' in s
    assert '"tier":1' in s  # from the asm
    assert '"tier":2' in s  # from the conveyor
    assert '"direction":"east"' in s


def test_user_message_contains_expected_sections():
    lay = empty_episode(seed=0)
    msg = build_user_message(lay)
    assert "Grid:" in msg
    assert "Green science recipe" in msg
    assert "Chest emission rates" in msg
    assert "Grid legend:" in msg
    assert "Current layout" in msg
    assert "Edit vocabulary" in msg
    assert "Goal:" in msg
    assert "Reply with the JSON array" in msg


def test_chat_messages_two_role_shape():
    lay = empty_episode(seed=0)
    ms = build_chat_messages(lay)
    assert len(ms) == 2
    assert ms[0]["role"] == "system"
    assert ms[1]["role"] == "user"
    assert "green science" in ms[0]["content"].lower()


def test_user_message_rates_precision():
    lay = Layout()
    lay.chests = [
        Chest(id="cb", kind="input-belts", x=0, y=0),
        Chest(id="ci", kind="input-inserters", x=0, y=1),
        Chest(id="co", kind="output-science", x=0, y=2),
    ]
    lay.chest_rates = ChestRates(belts=1.234, inserters=0.056)
    msg = build_user_message(lay)
    assert "1.234" in msg
    assert "0.056" in msg


# ================================================================
# Chunk 6 — edit schema + parser + applier tests
# ================================================================


# ---- Schema

def test_parse_edit_place_chest():
    e, err = parse_edit({"op": "place_chest", "id": "c1",
                          "kind": "input-belts", "x": 3, "y": 5})
    assert err is None
    assert isinstance(e, PlaceChest)
    assert e.kind == "input-belts"


def test_parse_edit_bad_op():
    e, err = parse_edit({"op": "eat_lunch", "id": "x"})
    assert e is None
    assert "unknown op" in err


def test_parse_edit_bad_tier():
    e, err = parse_edit({"op": "place_assembler", "id": "a", "tier": 5,
                          "x": 0, "y": 0})
    assert e is None
    assert "validation error" in err


def test_parse_edit_bad_direction():
    e, err = parse_edit({"op": "place_conveyor", "id": "c", "tier": 1,
                          "x": 0, "y": 0, "direction": "up"})
    assert e is None


def test_parse_edit_conveyor_line():
    e, err = parse_edit({"op": "place_conveyor_line", "id": "l", "tier": 1,
                          "from_x": 0, "from_y": 0, "to_x": 0, "to_y": 4})
    assert err is None
    assert isinstance(e, PlaceConveyorLine)


# ---- Parser

def test_parse_edits_clean_json_array():
    text = '[{"op":"place_chest","id":"c","kind":"input-belts","x":0,"y":0}]'
    r = parse_edits(text)
    assert r.ok
    assert len(r.edits) == 1


def test_parse_edits_strips_prose_before_and_after():
    text = 'Sure! Here you go:\n[{"op":"remove_entity","id":"x"}]\nDone.'
    r = parse_edits(text)
    assert r.ok
    assert len(r.edits) == 1


def test_parse_edits_strips_markdown_fence():
    text = "```json\n[{\"op\":\"remove_entity\",\"id\":\"a\"}]\n```"
    r = parse_edits(text)
    assert r.ok
    assert len(r.edits) == 1


def test_parse_edits_truncated_array_repaired_when_possible():
    text = '[{"op":"remove_entity","id":"a"},'   # unclosed after one complete edit
    r = parse_edits(text)
    assert r.parse_error is None
    assert len(r.edits) == 1


def test_parse_edits_partial_valid_edits_survive():
    text = ('[{"op":"place_chest","id":"c","kind":"input-belts","x":0,"y":0},'
            '{"op":"nope"}]')
    r = parse_edits(text)
    assert len(r.edits) == 1
    assert len(r.edit_errors) == 1


def test_parse_edits_no_bracket():
    r = parse_edits("hello world")
    assert r.parse_error is not None


# ---- Applier

def test_apply_place_chest_adds_to_layout():
    lay = Layout()
    e = PlaceChest(id="c", kind="input-belts", x=3, y=3)
    r = apply_edits(lay, [e])
    assert r.applied == 1
    assert r.errors == []
    assert len(r.layout.chests) == 1


def test_apply_duplicate_id_rejected():
    lay = Layout()
    e1 = PlaceChest(id="dup", kind="input-belts", x=0, y=0)
    e2 = PlaceChest(id="dup", kind="input-inserters", x=1, y=0)
    r = apply_edits(lay, [e1, e2])
    assert r.applied == 1
    assert len(r.errors) == 1
    assert "duplicate" in r.errors[0]


def test_apply_out_of_bounds_rejected():
    lay = Layout()
    e = PlaceConveyor(id="c", tier=1, x=25, y=5, direction="east")
    r = apply_edits(lay, [e])
    assert r.applied == 0
    assert len(r.errors) == 1
    assert "out of bounds" in r.errors[0]


def test_apply_assembler_footprint_overlap_rejected():
    lay = Layout()
    lay.chests.append(Chest(id="c", kind="input-belts", x=5, y=5))
    e = PlaceAssembler(id="a", tier=1, x=4, y=4)  # footprint includes (5,5)
    r = apply_edits(lay, [e])
    assert r.applied == 0
    assert "occupied" in r.errors[0]


def test_apply_perpendicular_crossing_allowed():
    lay = Layout()
    r = apply_edits(lay, [
        PlaceConveyor(id="e", tier=1, x=5, y=5, direction="east"),
        PlaceConveyor(id="n", tier=1, x=5, y=5, direction="north"),
    ])
    assert r.applied == 2
    assert r.errors == []


def test_apply_parallel_conveyor_stacking_rejected():
    lay = Layout()
    r = apply_edits(lay, [
        PlaceConveyor(id="e1", tier=1, x=5, y=5, direction="east"),
        PlaceConveyor(id="e2", tier=1, x=5, y=5, direction="east"),
    ])
    assert r.applied == 1
    assert len(r.errors) == 1
    assert "perpendicular" in r.errors[0]


def test_apply_conveyor_line_expands_direction_and_excludes_endpoints():
    lay = Layout()
    r = apply_edits(lay, [
        PlaceConveyorLine(id="l", tier=1, from_x=0, from_y=0, to_x=0, to_y=4),
    ])
    assert r.applied == 1
    assert r.errors == []
    assert [(c.id, c.x, c.y, c.direction) for c in r.layout.conveyors] == [
        ("l_1", 0, 1, "south"),
        ("l_2", 0, 2, "south"),
        ("l_3", 0, 3, "south"),
    ]


def test_apply_conveyor_line_rejects_diagonal():
    lay = Layout()
    r = apply_edits(lay, [
        PlaceConveyorLine(id="l", tier=1, from_x=0, from_y=0, to_x=2, to_y=2),
    ])
    assert r.applied == 0
    assert "straight" in r.errors[0]


def test_apply_remove_entity_removes():
    lay = Layout()
    r = apply_edits(lay, [
        PlaceConveyor(id="c", tier=1, x=5, y=5, direction="east"),
        RemoveEntity(id="c"),
    ])
    assert r.applied == 2
    assert len(r.layout.conveyors) == 0


def test_apply_remove_missing_entity_rejected():
    lay = Layout()
    r = apply_edits(lay, [RemoveEntity(id="ghost")])
    assert r.applied == 0
    assert "no such entity" in r.errors[0]


def test_apply_does_not_mutate_input_layout():
    lay = Layout()
    lay.chests.append(Chest(id="c", kind="input-belts", x=0, y=0))
    original_json = lay.to_json()
    r = apply_edits(lay, [PlaceConveyor(id="cv", tier=1, x=5, y=5,
                                          direction="east")])
    assert lay.to_json() == original_json  # input unchanged
    assert len(r.layout.conveyors) == 1     # copy modified


def test_end_to_end_parse_then_apply():
    lay = Layout()
    text = (
        "Here's the plan:\n```json\n"
        "[\n"
        '  {"op":"place_chest","id":"cb","kind":"input-belts","x":0,"y":0},\n'
        '  {"op":"place_chest","id":"ci","kind":"input-inserters","x":0,"y":1},\n'
        '  {"op":"place_chest","id":"co","kind":"output-science","x":19,"y":19},\n'
        '  {"op":"place_assembler","id":"a1","tier":1,"x":5,"y":5},\n'
        '  {"op":"place_conveyor","id":"cv1","tier":1,"x":4,"y":5,"direction":"east"}\n'
        "]\n```"
    )
    p = parse_edits(text)
    assert p.ok
    r = apply_edits(lay, p.edits)
    assert r.applied == 5
    assert r.errors == []
    assert r.layout.validate_layout() == []


# ================================================================
# Chunk 7 — translator to real Factorio
# ================================================================


def test_translator_emits_infinity_chests():
    # Chests are emitted as infinity-chest entities with no filter — the
    # driver inserts items each game-second to enforce sim's chest rates.
    lay = _minimal_layout()
    res = translate(lay)
    chests = [e for e in res.entities if e.name == "infinity-chest"]
    assert len(chests) == 3


def test_translator_emits_assembler_with_recipe():
    lay = _minimal_layout()  # asm-1 at (10, 10)
    res = translate(lay)
    asm_entities = [e for e in res.entities if e.name.startswith("assembling-machine-")]
    assert len(asm_entities) == 1
    e = asm_entities[0]
    assert e.name == "assembling-machine-1"
    assert e.recipe == "logistic-science-pack"
    # Position is footprint center = (10 + 1.5, 10 + 1.5) = (11.5, 11.5)
    assert e.position == {"x": 11.5, "y": 11.5}


def test_translator_belt_tier_names():
    lay = _minimal_layout()
    lay.conveyors = [
        Conveyor(id="c1", tier=1, x=15, y=15, direction="east"),
        Conveyor(id="c2", tier=2, x=16, y=15, direction="east"),
        Conveyor(id="c3", tier=3, x=17, y=15, direction="east"),
    ]
    res = translate(lay)
    names = [e.name for e in res.entities if "belt" in e.name and "underground" not in e.name]
    assert "transport-belt" in names
    assert "fast-transport-belt" in names
    assert "express-transport-belt" in names


def test_translator_direction_encoding():
    lay = _minimal_layout()
    lay.conveyors = [
        Conveyor(id="cn", tier=1, x=5, y=5, direction="north"),
        Conveyor(id="ce", tier=1, x=6, y=5, direction="east"),
        Conveyor(id="cs", tier=1, x=7, y=5, direction="south"),
        Conveyor(id="cw", tier=1, x=8, y=5, direction="west"),
    ]
    res = translate(lay)
    dirs = {e.direction for e in res.entities if e.name == "transport-belt"}
    assert dirs == {0, 4, 8, 12}


def test_translator_crossing_produces_underground_pair():
    lay = _minimal_layout()
    lay.conveyors = [
        Conveyor(id="ce", tier=1, x=5, y=5, direction="east"),
        Conveyor(id="cn_prev", tier=1, x=5, y=6, direction="north"),
        Conveyor(id="cn_cross", tier=1, x=5, y=5, direction="north"),  # crossing
        Conveyor(id="cn_next", tier=1, x=5, y=4, direction="north"),
    ]
    res = translate(lay)
    ug = [e for e in res.entities if "underground-belt" in e.name]
    # Expect entry + exit for the north-going crossing pair.
    assert len(ug) == 2
    types = {e.type for e in ug}
    assert types == {"input", "output"}
    # Belt at the crossing tile is the east one.
    belt_at_crossing = [e for e in res.entities
                         if e.name == "transport-belt"
                         and e.position == {"x": 5.5, "y": 5.5}]
    assert len(belt_at_crossing) == 1
    assert belt_at_crossing[0].direction == FACTORIO_DIRECTION["east"]


def test_translator_injects_inserter_for_interface_conveyor():
    # Belts chest at (5,4). Cv (6,4) east delivers to asm at (7,4).
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    res = translate(lay)
    inserters = [e for e in res.entities if e.name == "inserter"]
    # Three interface conveyors in _simple_chain_layout: cv_b (belts in),
    # cv_i (inserters in), cv_o (science out). All must yield an inserter.
    assert len(inserters) == 3


def test_translator_interface_conveyor_replaced_by_inserter():
    # _simple_chain_layout has cv_b at (6,4) east delivering to asm at (7,4).
    # Under the interface rule, cv_b is dropped; an inserter is emitted at (6,4).
    # No transport-belt should appear at (6,4).
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    res = translate(lay)
    belts_at_6_4 = [e for e in res.entities
                     if e.name == "transport-belt"
                     and abs(e.position["x"] - 6.5) < 1e-9
                     and abs(e.position["y"] - 4.5) < 1e-9]
    assert belts_at_6_4 == []
    inserters_at_6_4 = [e for e in res.entities
                         if e.name == "inserter"
                         and abs(e.position["x"] - 6.5) < 1e-9
                         and abs(e.position["y"] - 4.5) < 1e-9]
    assert len(inserters_at_6_4) == 1


def test_translator_grid_size_expands_if_needed():
    # Put an assembler near the grid edge and a conveyor just outside; the
    # cascade shift will push the conveyor into an expanded grid area.
    lay = Layout()
    lay.chests = [
        Chest(id="cb", kind="input-belts",     x=15, y=15),
        Chest(id="ci", kind="input-inserters", x=15, y=17),
        Chest(id="co", kind="output-science",  x=15, y=19),
    ]
    lay.assemblers = [Assembler(id="a", tier=1, x=17, y=15)]  # footprint (17..19, 15..17)
    lay.conveyors = [
        Conveyor(id="cv_out", tier=1, x=17, y=18, direction="south"),  # upstream (17,17) on asm
    ]
    res = translate(lay)
    # Cv_out will shift south (away from machine) to (17, 19). Grid should be
    # at least 20 wide/tall.
    assert res.grid_size[1] >= 20


def test_translator_no_entities_gives_empty_result():
    lay = Layout()
    res = translate(lay)
    assert res.entities == []
    assert res.grid_size == (25, 25)


def test_translator_output_dict_serializable():
    lay = _simple_chain_layout(tier=1, belts_rate=5.0, inserters_rate=5.0)
    res = translate(lay)
    d = res.as_dict()
    import json as _json
    _json.dumps(d)  # must not raise


# ---------------------------------------------------------------- runner

def _all_tests() -> list[Callable[[], None]]:
    g = globals()
    return [g[n] for n in sorted(g) if n.startswith("test_")]


def main() -> int:
    tests = _all_tests()
    fails: list[tuple[str, str]] = []
    for t in tests:
        try:
            t()
        except Exception:
            fails.append((t.__name__, traceback.format_exc()))
    print(f"{len(tests) - len(fails)} / {len(tests)} passed")
    for name, tb in fails:
        print(f"\n---- FAIL: {name} ----\n{tb}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
