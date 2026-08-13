"""Rate-based simulator for the simplified green-science env.

Given a Layout with per-episode chest emission rates, computes the steady-state
green-science delivery rate at the output-science chest.

Model:
- Each conveyor tile has 2 lanes; each lane carries at most one item type,
  capped at CONVEYOR_LANE_CAPACITY items/sec. Two conveyors may share a tile
  if perpendicular (a "crossing"); each keeps its own lane pool.
- Chest emission divides equally among adjacent conveyors that point away
  from the chest.
- Assembler consumes belts + inserters from any adjacent conveyor carrying
  those items. Conveyor direction does not matter for machine input.
- Assembler output = min(crafting_rate, belts_in, inserters_in) crafts/sec.
- Assembler pushes green-science onto adjacent conveyors that are empty or
  already carrying science. It does not push science onto input conveyors
  carrying transport-belts or inserters. Output divides equally among eligible
  adjacent conveyors.
- 2-lane cap: at most 2 distinct items per conveyor. If the sim would push a
  3rd item onto a conveyor, the surplus is dropped (with a warning). Each
  lane's rate is capped at CONVEYOR_LANE_CAPACITY.

Solver: fixed-point iteration up to 200 rounds. Convergence tolerance 1e-6.

Simplifications documented in plan.md; a stricter LP-based solver may be
added later if needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .entities import (
    ASSEMBLERS,
    CHEST_ITEM,
    CONVEYOR_LANES,
    CONVEYOR_LANE_CAPACITY,
    GREEN_SCIENCE_ITEM,
)
from .layout import Assembler, Chest, Conveyor, Layout

ITEM_BELTS = "transport-belt"
ITEM_INSERTERS = "inserter"
ITEM_SCIENCE = GREEN_SCIENCE_ITEM


@dataclass
class MachineFlow:
    id: str
    tier: int
    belts_in: float = 0.0
    inserters_in: float = 0.0
    science_out: float = 0.0


@dataclass
class SimResult:
    green_science_rate: float = 0.0
    total_science_produced: float = 0.0
    machine_flows: list[MachineFlow] = field(default_factory=list)
    conveyor_items: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False


# --------------------------------------------------------------- helpers

def _cap_lanes(items: dict[str, float], lane_capacity: float,
               warnings: list[str], tile_label: str) -> dict[str, float]:
    """Enforce: at most CONVEYOR_LANES distinct items per tile, and each
    item's rate <= lane_capacity items/sec (tier-dependent)."""
    if len(items) > CONVEYOR_LANES:
        kept = dict(sorted(items.items(), key=lambda kv: -kv[1])[:CONVEYOR_LANES])
        dropped = set(items) - set(kept)
        warnings.append(
            f"{tile_label}: > {CONVEYOR_LANES} items on tile; dropped {sorted(dropped)}"
        )
        items = kept
    capped: dict[str, float] = {}
    for item, rate in items.items():
        if rate > lane_capacity:
            warnings.append(
                f"{tile_label}: {item} rate {rate:.2f} capped at {lane_capacity}"
            )
            capped[item] = lane_capacity
        else:
            capped[item] = rate
    return capped


def _items_equal(a: dict[str, float], b: dict[str, float], tol: float) -> bool:
    keys = set(a) | set(b)
    return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) <= tol for k in keys)


# --------------------------------------------------------------- simulate

def simulate(layout: Layout, *, max_iters: int = 200,
             tol: float = 1e-6) -> SimResult:
    result = SimResult()
    # Non-fatal validation errors surface as warnings; sim still runs.
    validation_errs = layout.validate_layout()
    result.warnings.extend(validation_errs)

    # --------------- indices ---------------

    tile_conveyors: dict[tuple[int, int], list[Conveyor]] = {}
    for cv in layout.conveyors:
        tile_conveyors.setdefault((cv.x, cv.y), []).append(cv)

    tile_chest: dict[tuple[int, int], Chest] = {(c.x, c.y): c for c in layout.chests}

    tile_assembler: dict[tuple[int, int], Assembler] = {}
    for a in layout.assemblers:
        for t in a.footprint:
            tile_assembler[t] = a

    # Precompute per-conveyor connections.
    # conveyors_from_chest[chest_tile] = list of conveyors whose upstream is that tile.
    conveyors_from_chest: dict[tuple[int, int], list[Conveyor]] = {}
    for cv in layout.conveyors:
        conveyors_from_chest.setdefault(cv.upstream_tile(), []).append(cv)

    # For each assembler: adjacent conveyors are input candidates and output
    # candidates. Runtime item content decides whether a conveyor is acting as
    # input (belts/inserters present) or output (empty/science only).
    asm_adjacent_conveyors: dict[str, list[Conveyor]] = {a.id: [] for a in layout.assemblers}
    conveyor_adjacent_assemblers: dict[str, list[Assembler]] = {cv.id: [] for cv in layout.conveyors}
    for a in layout.assemblers:
        border = set(a.border_tiles())
        for cv in layout.conveyors:
            if (cv.x, cv.y) in border:
                asm_adjacent_conveyors[a.id].append(cv)
                conveyor_adjacent_assemblers[cv.id].append(a)

    # Required chests must all be present, otherwise short-circuit to 0.
    if not [c for c in layout.chests if c.kind == "output-science"]:
        return result
    if not [c for c in layout.chests if c.kind == "input-belts"]:
        return result
    if not [c for c in layout.chests if c.kind == "input-inserters"]:
        return result

    # --------------- initial state ---------------

    # per-tile per-item rate on conveyors. Two conveyors on same tile (crossing)
    # keep separate books, keyed by conveyor identity — but many downstream
    # calculations look at the tile-level view. Since the crossing conveyors
    # have different directions, we key by conveyor id for accuracy.
    conveyor_items: dict[str, dict[str, float]] = {cv.id: {} for cv in layout.conveyors}
    machine_flows: dict[str, MachineFlow] = {
        a.id: MachineFlow(id=a.id, tier=a.tier) for a in layout.assemblers
    }

    # Chest emission per adjacent conveyor. Constant across iterations.
    chest_push: dict[str, dict[str, float]] = {cv.id: {} for cv in layout.conveyors}
    for chest in layout.chests:
        item = CHEST_ITEM[chest.kind]
        if item is None:
            continue
        rate = (layout.chest_rates.belts if chest.kind == "input-belts"
                else layout.chest_rates.inserters)
        adjacent = conveyors_from_chest.get((chest.x, chest.y), [])
        if not adjacent or rate <= 0:
            continue
        per_cv = rate / len(adjacent)
        for cv in adjacent:
            chest_push[cv.id][item] = chest_push[cv.id].get(item, 0.0) + per_cv

    # --------------- fixed-point loop ---------------

    converged = False
    iteration = 0
    for iteration in range(1, max_iters + 1):
        new_conveyor_items: dict[str, dict[str, float]] = {cv.id: {} for cv in layout.conveyors}
        new_machine_flows: dict[str, MachineFlow] = {
            a.id: MachineFlow(id=a.id, tier=a.tier) for a in layout.assemblers
        }

        # Assembler outputs based on previous inputs. Eligible output conveyors
        # are adjacent conveyors that are empty or already carrying science.
        assembler_science_out: dict[str, float] = {}
        asm_output_conveyors: dict[str, list[Conveyor]] = {}
        for a in layout.assemblers:
            spec = ASSEMBLERS[a.tier]
            mf_prev = machine_flows[a.id]
            science = min(spec.crafts_per_sec_green_science,
                          mf_prev.belts_in, mf_prev.inserters_in)
            assembler_science_out[a.id] = science
            outs: list[Conveyor] = []
            for cv in asm_adjacent_conveyors[a.id]:
                prev_items = conveyor_items[cv.id]
                if ITEM_BELTS not in prev_items and ITEM_INSERTERS not in prev_items:
                    outs.append(cv)
            asm_output_conveyors[a.id] = outs

        # For each conveyor, aggregate incoming.
        # A cv receives from ANY neighbor conveyor whose downstream tile is
        # cv's own tile — so straight lines AND turns both propagate.
        for cv in layout.conveyors:
            inc: dict[str, float] = dict(chest_push[cv.id])  # start from chest pushes
            here = (cv.x, cv.y)

            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                neighbor = (here[0] + dx, here[1] + dy)
                if neighbor in tile_conveyors:
                    for up_cv in tile_conveyors[neighbor]:
                        if up_cv.id == cv.id:
                            continue
                        if up_cv.downstream_tile() == here:
                            for item, rate in conveyor_items[up_cv.id].items():
                                inc[item] = inc.get(item, 0.0) + rate

            # From assembler output: assembler pushes onto adjacent conveyors
            # that are not currently carrying recipe inputs.
            for a in conveyor_adjacent_assemblers[cv.id]:
                outs = asm_output_conveyors[a.id]
                if outs and cv in outs:
                    per_cv = assembler_science_out[a.id] / len(outs)
                    inc[ITEM_SCIENCE] = inc.get(ITEM_SCIENCE, 0.0) + per_cv

            inc = _cap_lanes(inc, CONVEYOR_LANE_CAPACITY[cv.tier],
                              result.warnings,
                              f"conveyor {cv.id} at ({cv.x},{cv.y}) T{cv.tier}")
            new_conveyor_items[cv.id] = inc

        # For each assembler, sum inputs from any adjacent conveyor.
        for a in layout.assemblers:
            mf = new_machine_flows[a.id]
            for cv in asm_adjacent_conveyors[a.id]:
                for item, rate in new_conveyor_items[cv.id].items():
                    if item == ITEM_BELTS:
                        mf.belts_in += rate
                    elif item == ITEM_INSERTERS:
                        mf.inserters_in += rate
                    # science flowing INTO an assembler is wasted (assembler
                    # doesn't consume it as an input).

        # Convergence check.
        changed = False
        for cvid in new_conveyor_items:
            if not _items_equal(new_conveyor_items[cvid], conveyor_items[cvid], tol):
                changed = True
                break
        if not changed:
            for aid in new_machine_flows:
                mf_new = new_machine_flows[aid]
                mf_old = machine_flows[aid]
                if (abs(mf_new.belts_in - mf_old.belts_in) > tol or
                        abs(mf_new.inserters_in - mf_old.inserters_in) > tol):
                    changed = True
                    break

        conveyor_items = new_conveyor_items
        machine_flows = new_machine_flows

        if not changed:
            converged = True
            break

    # --------------- deliverables ---------------

    # Green-science flowing to output-science chest tiles.
    delivered = 0.0
    for chest in layout.chests:
        if chest.kind != "output-science":
            continue
        chest_tile = (chest.x, chest.y)
        # Any conveyor whose downstream is the chest tile delivers to the chest.
        for cv in layout.conveyors:
            if cv.downstream_tile() == chest_tile:
                delivered += conveyor_items[cv.id].get(ITEM_SCIENCE, 0.0)

    # Fill result.
    total_produced = 0.0
    for a in layout.assemblers:
        mf = machine_flows[a.id]
        spec = ASSEMBLERS[a.tier]
        mf.science_out = min(spec.crafts_per_sec_green_science,
                             mf.belts_in, mf.inserters_in)
        total_produced += mf.science_out
    result.machine_flows = list(machine_flows.values())
    result.total_science_produced = total_produced
    result.green_science_rate = delivered
    # Aggregate conveyor items by tile for external inspection.
    tile_items: dict[tuple[int, int], dict[str, float]] = {}
    for cv in layout.conveyors:
        tt = (cv.x, cv.y)
        tile_items.setdefault(tt, {})
        for item, rate in conveyor_items[cv.id].items():
            tile_items[tt][item] = tile_items[tt].get(item, 0.0) + rate
    result.conveyor_items = tile_items
    result.iterations = iteration
    result.converged = converged
    return result
