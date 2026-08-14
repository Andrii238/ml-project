"""Harness tests for the current simplified green-science schema."""
from __future__ import annotations

import json

import pytest

from harness.edit_applier import apply_edits
from harness.edit_parser import MAX_EDITS, parse_edits
from harness.edit_schema import RemoveEntity
from harness.evaluator import evaluate_policy
from harness.prompt_builder import build_user_message
from mini_factorio.layout import Assembler, Conveyor
from mini_factorio.random_layouts import empty_episode


def test_parse_clean_bare_json_array():
    r = parse_edits('[{"op":"remove_entity","id":"m1"}]')
    assert r.ok
    assert len(r.edits) == 1
    assert isinstance(r.edits[0], RemoveEntity)


def test_parse_fenced_json_array():
    text = 'Sure\n```json\n[{"op":"remove_entity","id":"m1"}]\n```\nDone.'
    r = parse_edits(text)
    assert r.ok
    assert isinstance(r.edits[0], RemoveEntity)


def test_parse_trailing_junk_after_complete_array():
    r = parse_edits('[{"op":"remove_entity","id":"m1"}]\nextra junk')
    assert r.ok
    assert len(r.edits) == 1


def test_parse_truncated_array_repairs_complete_objects_and_caps():
    text = "[" + ",".join(
        json.dumps({"op": "remove_entity", "id": f"x{i}"})
        for i in range(MAX_EDITS + 5)
    )
    r = parse_edits(text)
    assert r.parse_error is None
    assert len(r.edits) == MAX_EDITS


def test_parse_malformed_returns_error():
    r = parse_edits("clearly not json")
    assert r.parse_error is not None
    assert not r.ok


def test_parse_wrong_schema_collects_edit_error():
    r = parse_edits('[{"op":"remove_entity"}]')
    assert r.parse_error is None
    assert r.edit_errors
    assert not r.ok


def test_apply_place_assembler_success():
    lay = empty_episode(seed=0)
    r = parse_edits('[{"op":"place_assembler","id":"a1","tier":1,"x":5,"y":5}]')
    res = apply_edits(lay, r.edits)
    assert res.applied == 1
    assert any(a.id == "a1" for a in res.layout.assemblers)


def test_apply_rejects_out_of_bounds_assembler():
    lay = empty_episode(seed=0)
    r = parse_edits('[{"op":"place_assembler","id":"a1","tier":1,"x":24,"y":24}]')
    res = apply_edits(lay, r.edits)
    assert res.applied == 0
    assert "out of bounds" in res.errors[0]


def test_apply_rejects_duplicate_id():
    lay = empty_episode(seed=0)
    lay.assemblers.append(Assembler(id="a1", tier=1, x=5, y=5))
    r = parse_edits('[{"op":"place_assembler","id":"a1","tier":1,"x":10,"y":10}]')
    res = apply_edits(lay, r.edits)
    assert res.applied == 0
    assert "duplicate" in res.errors[0]


def test_apply_conveyor_line_excludes_endpoints_and_infers_direction():
    lay = empty_episode(seed=0)
    r = parse_edits(
        '[{"op":"place_conveyor_line","id":"l1","tier":1,'
        '"from_x":0,"from_y":5,"to_x":4,"to_y":5}]'
    )
    res = apply_edits(lay, r.edits)
    assert res.applied == 1
    assert [(c.x, c.y, c.direction) for c in res.layout.conveyors] == [
        (1, 5, "east"),
        (2, 5, "east"),
        (3, 5, "east"),
    ]


def test_apply_rejects_diagonal_conveyor_line():
    lay = empty_episode(seed=0)
    r = parse_edits(
        '[{"op":"place_conveyor_line","id":"l1","tier":1,'
        '"from_x":0,"from_y":0,"to_x":4,"to_y":4}]'
    )
    res = apply_edits(lay, r.edits)
    assert res.applied == 0
    assert "straight" in res.errors[0]


def test_apply_allows_perpendicular_conveyor_crossing():
    lay = empty_episode(seed=0)
    lay.conveyors.append(Conveyor(id="c1", tier=1, x=5, y=5, direction="east"))
    r = parse_edits('[{"op":"place_conveyor","id":"c2","tier":1,"x":5,"y":5,"direction":"north"}]')
    res = apply_edits(lay, r.edits)
    assert res.applied == 1
    assert res.layout.validate_layout() == []


def test_apply_remove_success():
    lay = empty_episode(seed=0)
    lay.assemblers.append(Assembler(id="a1", tier=1, x=5, y=5))
    r = parse_edits('[{"op":"remove_entity","id":"a1"}]')
    res = apply_edits(lay, r.edits)
    assert res.applied == 1
    assert not res.layout.assemblers


def test_evaluator_scores_prompt_completion():
    lay = empty_episode(seed=0)
    prompt = build_user_message(lay)

    def policy(prompts, **_kwargs):
        return ['[{"op":"place_assembler","id":"a1","tier":1,"x":5,"y":5}]'
                for _ in prompts]

    report = evaluate_policy(policy, [prompt])
    assert report.n == 1
    assert report.parse_ok_rate == 1.0
    assert report.valid_rate == 1.0
    assert report.mean_machine_count == 1.0


def test_evaluator_malformed_policy_penalized():
    lay = empty_episode(seed=0)
    prompt = build_user_message(lay)

    def policy(prompts, **_kwargs):
        return ["I refuse to output JSON." for _ in prompts]

    report = evaluate_policy(policy, [prompt])
    assert report.parse_ok_rate == 0.0
    assert report.valid_rate == 0.0
    assert report.mean_reward == -50.0


def test_prompt_contains_layout_rules_and_goal():
    lay = empty_episode(seed=0)
    prompt = build_user_message(lay)
    assert "Green science recipe" in prompt
    assert "place_conveyor_line" in prompt
    assert "Maximizing delivered rate" in prompt
    assert "<<LAYOUT>>" in prompt


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
