"""Build SFT training pairs from FLE-validated blueprints.

Format-only SFT: teach JSON schema compliance and valid entity/recipe names,
NOT factory design. Small targets (1-3 edits each), many random samples per
blueprint. Oracle miner+belt pairs excluded (they taught the wrong pattern).

Source filter: blueprints with `build_errors == 0` AND `top_item != invalid`
AND actual green-science production > 0. Lines 17/41/43 fail this filter —
they pass build/top_item checks but produce belts/gears/copper-cable only.

Augmentation: for each GS-producing blueprint, sample many small strips.
Deterministic seeding.
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


GS_ITEM = "logistic-science-pack"


def _load_usable_linenos() -> list[int]:
    """Return blueprint line numbers that actually produce green science.

    Filter: build_errors==0 AND top_item != invalid AND green-science-rate > 0.
    The green-science filter drops lines 17 (belts+gears only), 41/43
    (copper-cable only) which pass the surface checks but teach the wrong
    completion pattern.
    """
    linenos: list[int] = []
    for path in (CLASSIF_1, CLASSIF_2):
        for entry in json.loads(path.read_text()):
            if (
                entry.get("build_errors") == 0
                and entry.get("top_item")
                and entry.get("top_item") != "invalid"
                and (entry.get("rates") or {}).get(GS_ITEM, 0) > 0
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


SAMPLES_PER_STRIP = 12  # random strip-subsets per (blueprint, strip-size)
STRIP_SIZES = (1, 2, 3)  # keep targets SHORT — format-only SFT
MAX_ENTITIES = 80       # drop giant blueprints; prompt token budget can't fit them


def build_pairs(seed: int = 42) -> list[SFTPair]:
    """Deterministic small-target pair generation for format-only SFT.

    For each green-science-producing blueprint, sample many small strips
    (1, 2, or 3 random entities removed). Target = the edits that re-add
    just those removed entities. Small targets keep the model focused on
    JSON format compliance rather than memorizing full factory rebuilds.

    ~10 blueprints × 3 strip-sizes × 8 samples = ~240 pairs.
    Oracle miner+belt pairs are EXCLUDED — they taught "output miners and
    belts, forget assemblers" and hurt SFT.
    """
    rng = random.Random(seed)
    pairs: list[SFTPair] = []
    for lineno, layout in _load_layouts():
        all_entities: list = (
            list(layout.machines) + list(layout.inserters) + list(layout.belts)
        )
        if not all_entities or len(all_entities) > MAX_ENTITIES:
            continue  # skip giant blueprints — prompt exceeds token budget
        for n in STRIP_SIZES:
            k = min(n, len(all_entities))
            for s in range(SAMPLES_PER_STRIP):
                sample = rng.sample(all_entities, k)
                pairs.append(_pair_from_removal(
                    layout, sample, f"partial_{n}_s{s}", lineno))
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
