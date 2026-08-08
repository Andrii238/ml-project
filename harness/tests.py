"""Harness tests: parser, applier, evaluator with dummy policies."""
from __future__ import annotations

import json

import pytest

from harness.edit_applier import apply_edits
from harness.edit_parser import parse_edits
from harness.edit_schema import AddEntity, EditList, RemoveEntity
from harness.evaluator import evaluate_policy
from harness.prompt_builder import build_prompt
from mini_factorio.layout import Layout, Machine, Resource
from mini_factorio.random_layouts import empty_layout


# ---------- Parser ----------


def test_parse_clean_json():
    r = parse_edits('{"edits": [{"op": "remove_entity", "id": "m1"}]}')
    assert r.parse_ok
    assert len(r.edits.edits) == 1
    assert isinstance(r.edits.edits[0], RemoveEntity)


def test_parse_fenced_json():
    text = 'Sure!\n```json\n{"edits": []}\n```\nDone.'
    r = parse_edits(text)
    assert r.parse_ok
    assert r.edits.edits == []


def test_parse_bare_list_wrapped():
    r = parse_edits('[{"op": "remove_entity", "id": "x"}]')
    assert r.parse_ok
    assert r.edits.edits[0].id == "x"


def test_parse_malformed_returns_flag():
    r = parse_edits("clearly not json")
    assert not r.parse_ok
    assert r.error is not None


def test_parse_wrong_schema_flagged():
    # Missing required 'id'
    r = parse_edits('{"edits": [{"op": "remove_entity"}]}')
    assert not r.parse_ok


# ---------- Applier ----------


def _seed_layout() -> Layout:
    return Layout(
        grid_size=(16, 16),
        resources=[Resource(type="iron-ore", x=0, y=0, size=3)],
        machines=[Machine(id="existing", type="electric-mining-drill", x=0, y=0,
                          target_resource="iron-ore")],
    )


def test_apply_add_entity_success():
    lay = _seed_layout()
    edits = EditList(edits=[
        AddEntity(op="add_entity", id="new1", type="assembling-machine-1",
                  x=5, y=5, recipe="iron-gear-wheel"),
    ])
    res = apply_edits(lay, edits)
    assert res.n_applied == 1
    assert any(m.id == "new1" for m in res.layout.machines)


def test_apply_rejects_overlap_but_keeps_other():
    lay = _seed_layout()
    edits = EditList(edits=[
        # This one overlaps 'existing' at (0,0).
        AddEntity(op="add_entity", id="bad", type="assembling-machine-1",
                  x=1, y=1, recipe="iron-gear-wheel"),
        # This one is fine.
        AddEntity(op="add_entity", id="good", type="assembling-machine-1",
                  x=8, y=8, recipe="iron-gear-wheel"),
    ])
    res = apply_edits(lay, edits)
    assert res.n_applied == 1
    assert "rejected" in res.errors[0]
    assert res.errors[1] == ""


def test_apply_duplicate_id_rejected():
    lay = _seed_layout()
    edits = EditList(edits=[
        AddEntity(op="add_entity", id="existing", type="assembling-machine-1",
                  x=5, y=5, recipe="iron-gear-wheel"),
    ])
    res = apply_edits(lay, edits)
    assert res.n_applied == 0
    assert "already exists" in res.errors[0]


def test_apply_remove_success():
    lay = _seed_layout()
    edits = EditList(edits=[RemoveEntity(op="remove_entity", id="existing")])
    res = apply_edits(lay, edits)
    assert res.n_applied == 1
    assert not res.layout.machines


def test_apply_remove_missing():
    lay = _seed_layout()
    edits = EditList(edits=[RemoveEntity(op="remove_entity", id="nope")])
    res = apply_edits(lay, edits)
    assert res.n_applied == 0
    assert "not found" in res.errors[0]


# ---------- Evaluator ----------


def _null_policy(prompt: str) -> str:
    return '{"edits": []}'


def _malformed_policy(prompt: str) -> str:
    return "I refuse to output JSON."


def _good_policy(prompt: str) -> str:
    return json.dumps({"edits": [{"op": "remove_entity", "id": "existing"}]})


def test_evaluator_null_policy_records_zero_edits():
    layouts = [_seed_layout(), _seed_layout()]
    report = evaluate_policy(_null_policy, layouts)
    assert report.summary()["n_episodes"] == 2
    assert report.valid_edit_rate() == 0.0  # 0/0 → treated as 0
    assert report.invalid_json_rate() == 0.0


def test_evaluator_malformed_policy_flagged():
    report = evaluate_policy(_malformed_policy, [_seed_layout()])
    assert report.invalid_json_rate() == 1.0


def test_evaluator_good_policy_applies_edits():
    report = evaluate_policy(_good_policy, [_seed_layout()])
    assert report.episodes[0].n_edits_applied == 1
    assert report.valid_edit_rate() == 1.0


def test_prompt_contains_layout_and_rules():
    lay = empty_layout(seed=0)
    prompt = build_prompt(lay)
    assert "Game rules" in prompt
    assert "Edit schema" in prompt
    assert "Current layout" in prompt
    # Layout should round-trip via the embedded JSON
    assert "iron-ore" in prompt


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
