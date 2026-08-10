"""Build SFT training pairs from FLE-validated blueprints.

Each pair = (stripped_layout, edit_list_to_re_add_removed_entities).

Source: `results/blueprint_classification.json` + `.new.json` filtered to
`build_errors == 0` and `top_item != invalid`. Blueprints decoded via
`translator.from_fle.blueprint_dict_to_layout`.

Augmentation: for each blueprint, 5 pairs
    - 1 "from-scratch" pair (all machines/inserters/belts stripped → full re-add)
    - 4 "partial-strip" pairs (strip 1, 2, 3, 4 random entities → re-add subset)

Deterministic seeding so the dataset is reproducible.
"""
from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass

from mini_factorio.layout import Belt, Inserter, Layout, Machine

REPO_ROOT = pathlib.Path(__file__).parent.parent
BP_TXT = REPO_ROOT / "translator" / "user_blueprints.txt"
CLASSIF_1 = REPO_ROOT / "results" / "blueprint_classification.json"
CLASSIF_2 = REPO_ROOT / "results" / "blueprint_classification.new.json"


@dataclass
class SFTPair:
    stripped: Layout
    edits: list[dict]      # JSON-serializable edit list
    source_line: int
    strip_kind: str        # "from_scratch" | "partial_N"


def _load_usable_linenos() -> list[int]:
    """Return blueprint line numbers with build_errors=0 and produced output."""
    linenos: list[int] = []
    for path in (CLASSIF_1, CLASSIF_2):
        for entry in json.loads(path.read_text()):
            if (
                entry.get("build_errors") == 0
                and entry.get("top_item")
                and entry.get("top_item") != "invalid"
            ):
                linenos.append(entry["lineno"])
    return sorted(set(linenos))


def _read_line(n: int) -> str:
    with open(BP_TXT) as f:
        for i, raw in enumerate(f, 1):
            if i == n:
                return raw.strip()
    raise ValueError(f"blueprint line {n} not found")


def _load_layouts() -> list[tuple[int, Layout]]:
    """Decode + translate every usable blueprint. Returns (lineno, Layout)."""
    from translator.from_fle import (
        blueprint_dict_to_layout,
        decode_blueprint_string,
        infer_belt_items,
    )
    out: list[tuple[int, Layout]] = []
    for ln in _load_usable_linenos():
        bp = decode_blueprint_string(_read_line(ln))
        layout = infer_belt_items(blueprint_dict_to_layout(bp))
        out.append((ln, layout))
    return out


def _machine_to_edit(m: Machine) -> dict:
    e = {"op": "add_entity", "id": m.id, "type": m.type, "x": m.x, "y": m.y}
    if m.direction and m.direction != "north":
        e["direction"] = m.direction
    if m.recipe is not None:
        e["recipe"] = m.recipe
    if m.target_resource is not None:
        e["target_resource"] = m.target_resource
    return e


def _inserter_to_edit(i: Inserter) -> dict:
    return {"op": "add_inserter", "id": i.id, "x": i.x, "y": i.y,
            "direction": i.direction}


def _belt_to_edit(b: Belt) -> dict:
    return {
        "op": "add_belt", "id": b.id, "item": b.item,
        "tiles": [[t.x, t.y, t.direction] for t in b.tiles],
    }


def _strip_layout(layout: Layout, keep_machine_ids: set[str],
                  keep_inserter_ids: set[str],
                  keep_belt_ids: set[str]) -> Layout:
    """Return a copy of `layout` retaining only entities whose ids are in the keep sets."""
    d = layout.model_dump()
    d["machines"] = [m for m in d["machines"] if m["id"] in keep_machine_ids]
    d["inserters"] = [i for i in d["inserters"] if i["id"] in keep_inserter_ids]
    d["belts"] = [b for b in d["belts"] if b["id"] in keep_belt_ids]
    return Layout.model_validate(d)


def _pair_from_removal(layout: Layout, remove: list, kind: str, lineno: int) -> SFTPair:
    """Build a pair given the list of entity objects to remove."""
    remove_m_ids = {m.id for m in remove if isinstance(m, Machine)}
    remove_i_ids = {i.id for i in remove if isinstance(i, Inserter)}
    remove_b_ids = {b.id for b in remove if isinstance(b, Belt)}
    keep_m = {m.id for m in layout.machines if m.id not in remove_m_ids}
    keep_i = {i.id for i in layout.inserters if i.id not in remove_i_ids}
    keep_b = {b.id for b in layout.belts if b.id not in remove_b_ids}
    stripped = _strip_layout(layout, keep_m, keep_i, keep_b)
    edits: list[dict] = []
    for e in remove:
        if isinstance(e, Machine):
            edits.append(_machine_to_edit(e))
        elif isinstance(e, Inserter):
            edits.append(_inserter_to_edit(e))
        elif isinstance(e, Belt):
            edits.append(_belt_to_edit(e))
    return SFTPair(stripped=stripped, edits=edits, source_line=lineno,
                   strip_kind=kind)


def build_pairs(seed: int = 42) -> list[SFTPair]:
    """Deterministic pair generation.

    Two data sources:
      1. Blueprint pairs (65): strip-and-re-add on 13 FLE-validated blueprints.
      2. Oracle-solver pairs (~224): programmatic miner+belt layouts for each
         resource type on each of 60 training layouts (from `oracle_solver`).

    Blueprint pairs teach assembler chain patterns. Solver pairs teach miner
    placement + belt routing + correct target_resource. Combined ~289 pairs.
    """
    from mini_factorio.random_layouts import train_val_split
    from training.oracle_solver import solve

    rng = random.Random(seed)
    pairs: list[SFTPair] = []
    # 1. Blueprint-derived pairs.
    for lineno, layout in _load_layouts():
        all_entities: list = (
            list(layout.machines) + list(layout.inserters) + list(layout.belts)
        )
        if not all_entities:
            continue
        pairs.append(_pair_from_removal(layout, all_entities,
                                        "from_scratch", lineno))
        for n in (1, 2, 3, 4):
            k = min(n, len(all_entities))
            sample = rng.sample(all_entities, k)
            pairs.append(_pair_from_removal(layout, sample,
                                            f"partial_{n}", lineno))
    # 2. Oracle-solver pairs. Input = the empty training layout; target = the
    # solver's edit list. Uses train seeds so val is untouched.
    train, _ = train_val_split(60, 20)
    for i, layout in enumerate(train):
        for result in solve(layout):
            pairs.append(SFTPair(
                stripped=layout,
                edits=result.edits,
                source_line=-1000 - i,   # negative to distinguish from blueprint lines
                strip_kind=result.template_name,
            ))
    return pairs


def pair_to_prompt_completion(pair: SFTPair) -> dict:
    """Format one pair as {'prompt_messages', 'completion'} for training.

    The prompt uses `build_chat_messages` for consistency with eval prompts.
    The completion is the JSON edit list, prefixed to match our runtime
    assistant-prefill (`{"edits": [`) so the model learns to continue that shape.
    """
    from harness.prompt_builder import build_chat_messages
    messages = build_chat_messages(pair.stripped)
    completion = json.dumps({"edits": pair.edits})
    return {"messages": messages, "completion": completion,
            "source_line": pair.source_line, "strip_kind": pair.strip_kind}
