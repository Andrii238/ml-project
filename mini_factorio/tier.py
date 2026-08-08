"""Tier swap helper: rewrite an existing Layout using higher-tier entities.

Use this to test the same layout geometry at tier-1, tier-2, tier-3 without
re-authoring — matches the plan of comparing GRPO policy improvement across
tier configurations.

Tier mapping (cumulative — a tier-N config allows all entities of tiers 1..N):
    tier | miner                   | furnace         | assembler              | belt                    | inserter
    -----+-------------------------+-----------------+------------------------+-------------------------+---------------
    1    | electric-mining-drill   | stone-furnace   | assembling-machine-1   | transport-belt          | inserter
    2    | electric-mining-drill   | steel-furnace   | assembling-machine-2   | fast-transport-belt     | fast-inserter
    3    | electric-mining-drill   | electric-furnace| assembling-machine-3   | express-transport-belt  | stack-inserter
"""
from __future__ import annotations

from .layout import Belt, Inserter, Layout, Machine

FURNACE_BY_TIER: dict[int, str] = {
    1: "stone-furnace",
    2: "steel-furnace",
    3: "electric-furnace",
}
ASSEMBLER_BY_TIER: dict[int, str] = {
    1: "assembling-machine-1",
    2: "assembling-machine-2",
    3: "assembling-machine-3",
}
BELT_BY_TIER: dict[int, str] = {
    1: "transport-belt",
    2: "fast-transport-belt",
    3: "express-transport-belt",
}
INSERTER_BY_TIER: dict[int, str] = {
    1: "inserter",
    2: "fast-inserter",
    3: "stack-inserter",
}


def to_tier(layout: Layout, tier: int) -> Layout:
    """Return a NEW Layout with all entities swapped to `tier` variants.

    Notes:
    - Miner stays electric-mining-drill (only one tier of that in our schema).
    - Electric furnace (tier 3) is 3x3, not 2x2, so tier-3 swap may cause
      collisions on layouts where a stone-furnace was tightly packed. Callers
      should re-validate the returned Layout and handle errors.
    - Recipes are unchanged (recipe compatibility is checked in layout.validate).
    """
    if tier not in (1, 2, 3):
        raise ValueError(f"tier must be 1, 2, or 3; got {tier}")
    new_machines: list[Machine] = []
    for m in layout.machines:
        if m.type == "electric-mining-drill":
            new_type = m.type
        elif m.type in ("stone-furnace", "steel-furnace", "electric-furnace"):
            new_type = FURNACE_BY_TIER[tier]
        elif m.type in ("assembling-machine-1", "assembling-machine-2",
                       "assembling-machine-3"):
            new_type = ASSEMBLER_BY_TIER[tier]
        else:
            new_type = m.type
        new_machines.append(m.model_copy(update={"type": new_type}))

    new_inserters = [
        i.model_copy(update={"type": INSERTER_BY_TIER[tier]})
        for i in layout.inserters
    ]
    new_belts = [
        b.model_copy(update={"type": BELT_BY_TIER[tier]})
        for b in layout.belts
    ]
    return layout.model_copy(update={
        "machines": new_machines,
        "inserters": new_inserters,
        "belts": new_belts,
    })
