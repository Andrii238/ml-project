"""Rate-based flow simulator for Mini-Factorio layouts.

Given a validated Layout, computes steady-state throughputs for every machine,
inserter and belt, and returns the total green-science-pack production rate
(items/sec). All flows are computed analytically — no ticks.

Model (plan.md §Simulator):
- Nodes: machines, inserters, belts.
- Inserter picks up from tile in direction opposite(direction), drops at tile in direction.
  A pickup/drop tile must be adjacent to a machine footprint OR be a belt tile.
- Belt flow: total = min(sum producer supplies, sum consumer demands, BELT_SPEED),
  FCFS-distributed by upstream position.
- Machines cap output at min(nominal, min supply/required across inputs). Furnaces
  additionally require coal fuel; coal_rate_needed = usage_kw / (COAL_ENERGY_MJ * 1000)
  per second at nominal.
- Solved by fixpoint iteration on machine rates. Rates only decrease across
  iterations, so convergence is guaranteed (bounded below by 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .entities import (
    BELT_SPEED,
    COAL_ENERGY_MJ,
    INSERTER_THROUGHPUT,
    MACHINES,
)
from .layout import (
    DIR_DELTA,
    OPPOSITE,
    Belt,
    Inserter,
    Layout,
    Machine,
    _machine_kind,
)


def _inserter_throughput(insr: Inserter) -> float:
    """items/sec swing rate for this inserter's tier."""
    return INSERTER_THROUGHPUT[insr.type]


def _belt_speed(belt: Belt) -> float:
    """items/sec cap for one belt (min across all tiles = tier speed)."""
    return BELT_SPEED[belt.type]
from .recipes import GREEN_SCIENCE_ITEM, RECIPES

FUEL_ITEM = "coal"
MAX_ITER = 200
CONVERGENCE_TOL = 1e-9


@dataclass
class InserterConn:
    """Resolved connection info for one inserter."""
    id: str
    pickup_tile: tuple[int, int]
    drop_tile: tuple[int, int]
    source_kind: str | None  # 'machine' | 'belt' | None (unconnected)
    source_id: str | None
    sink_kind: str | None
    sink_id: str | None
    item: str | None  # the item flowing through this inserter (None if invalid)
    # Position on belt if source is belt (index into belt.tiles), else None
    source_pos: int | None = None
    sink_pos: int | None = None


@dataclass
class SimResult:
    green_science_rate: float
    machine_rate: dict[str, float]  # machine.id → output items/sec
    inserter_rate: dict[str, float]  # inserter.id → items/sec moved
    belt_flow: dict[str, float]  # belt.id → items/sec on belt
    errors: list[str] = field(default_factory=list)


# ------------------------- Connection resolution -------------------------


def _tile_owner_map(layout: Layout) -> dict[tuple[int, int], tuple[str, str]]:
    """Map every occupied tile → (kind, entity_id). kind ∈ {'machine','belt','inserter'}."""
    m: dict[tuple[int, int], tuple[str, str]] = {}
    for machine in layout.machines:
        for t in layout.machine_footprint(machine):
            m[t] = ("machine", machine.id)
    for i in layout.inserters:
        m[(i.x, i.y)] = ("inserter", i.id)
    for b in layout.belts:
        for bt in b.tiles:
            m[(bt.x, bt.y)] = ("belt", b.id)
    return m


def _belt_position_map(layout: Layout) -> dict[str, dict[tuple[int, int], int]]:
    """For each belt, map (x, y) → tile index (0 = upstream)."""
    out: dict[str, dict[tuple[int, int], int]] = {}
    for b in layout.belts:
        out[b.id] = {(bt.x, bt.y): idx for idx, bt in enumerate(b.tiles)}
    return out


def _machine_output_item(m: Machine) -> str | None:
    kind = _machine_kind(m.type)
    if kind == "miner":
        return m.target_resource
    if m.recipe is None or m.recipe not in RECIPES:
        return None
    products = RECIPES[m.recipe].products
    if len(products) != 1:
        return None
    return next(iter(products))


def _machine_input_items(m: Machine) -> dict[str, float]:
    kind = _machine_kind(m.type)
    if kind == "miner":
        return {}
    if m.recipe is None or m.recipe not in RECIPES:
        return {}
    return {k: float(v) for k, v in RECIPES[m.recipe].ingredients.items()}


def _nominal_output_rate(m: Machine) -> float:
    """Items/sec produced at 100% supply."""
    kind = _machine_kind(m.type)
    spec = MACHINES[m.type]
    if kind == "miner":
        # Mining recipes have time=1 and output {resource: 1}, so rate = speed.
        r = RECIPES.get(m.target_resource or "")
        if r is None:
            return 0.0
        return spec.crafting_speed * next(iter(r.products.values())) / r.time
    if m.recipe is None or m.recipe not in RECIPES:
        return 0.0
    r = RECIPES[m.recipe]
    return spec.crafting_speed * next(iter(r.products.values())) / r.time


def _nominal_coal_rate(m: Machine) -> float:
    """Coal items/sec at nominal for a burner furnace, else 0."""
    spec = MACHINES[m.type]
    if spec.fuel_category != "chemical":
        return 0.0
    # usage is kW = kJ/sec. Coal energy = 4 MJ = 4000 kJ.
    return spec.usage_kw / (COAL_ENERGY_MJ * 1000.0)


def resolve_inserters(layout: Layout) -> tuple[list[InserterConn], list[str]]:
    """Resolve source/sink and item for every inserter. Returns (conns, warnings)."""
    owner = _tile_owner_map(layout)
    belt_pos = _belt_position_map(layout)
    machines_by_id = {m.id: m for m in layout.machines}

    conns: list[InserterConn] = []
    warnings: list[str] = []

    for insr in layout.inserters:
        dx, dy = DIR_DELTA[insr.direction]
        drop = (insr.x + dx, insr.y + dy)
        odx, ody = DIR_DELTA[OPPOSITE[insr.direction]]
        pickup = (insr.x + odx, insr.y + ody)

        src_kind = src_id = None
        snk_kind = snk_id = None
        src_pos: int | None = None
        snk_pos: int | None = None

        if pickup in owner:
            k, oid = owner[pickup]
            if k in ("machine", "belt"):
                src_kind, src_id = k, oid
                if k == "belt":
                    src_pos = belt_pos[oid][pickup]
        if drop in owner:
            k, oid = owner[drop]
            if k in ("machine", "belt"):
                snk_kind, snk_id = k, oid
                if k == "belt":
                    snk_pos = belt_pos[oid][drop]

        # Determine item flowing
        item: str | None = None
        if src_kind == "machine":
            src_machine = machines_by_id[src_id]
            if _machine_kind(src_machine.type) == "miner":
                # Real Factorio: mining drills output only at drop_position;
                # inserters cannot pick from the drill body. Layouts must route
                # miners via a belt tile placed at drop_position, then have the
                # inserter pick from the belt.
                warnings.append(
                    f"inserter {insr.id}: cannot pick directly from miner "
                    f"{src_id}; place a belt at drop_position {src_machine.drop_position()} "
                    f"and pick from that belt"
                )
                src_kind = src_id = None  # invalidate the source
            else:
                item = _machine_output_item(src_machine)
        elif src_kind == "belt":
            item = next(b.item for b in layout.belts if b.id == src_id)
        # Validate sink accepts this item
        if item is not None and snk_kind == "machine":
            m = machines_by_id[snk_id]
            valid_inputs = set(_machine_input_items(m).keys())
            if _machine_kind(m.type) == "furnace":
                valid_inputs.add(FUEL_ITEM)
            if item not in valid_inputs:
                warnings.append(
                    f"inserter {insr.id}: sink machine {snk_id} does not accept {item!r}"
                )
                item = None
        elif item is not None and snk_kind == "belt":
            belt_item = next(b.item for b in layout.belts if b.id == snk_id)
            if item != belt_item:
                warnings.append(
                    f"inserter {insr.id}: sink belt {snk_id} carries {belt_item!r}, "
                    f"not {item!r}"
                )
                item = None

        conns.append(InserterConn(
            id=insr.id, pickup_tile=pickup, drop_tile=drop,
            source_kind=src_kind, source_id=src_id,
            sink_kind=snk_kind, sink_id=snk_id,
            item=item, source_pos=src_pos, sink_pos=snk_pos,
        ))
    return conns, warnings


# ------------------------- Fixpoint solver -------------------------


def simulate(layout: Layout, *, implicit_sinks: set[str] | None = None) -> SimResult:
    """Compute steady-state rates for the layout.

    `implicit_sinks`: set of item names treated as having an infinite consumer
    (like green-science-pack's implicit lab). Use for testing individual chain
    segments without needing a full green-science layout to avoid dead-end
    back-pressure. Default: only GREEN_SCIENCE_ITEM.
    """
    sink_items: set[str] = {GREEN_SCIENCE_ITEM} | (implicit_sinks or set())
    errs = layout.validate_layout()
    if errs:
        return SimResult(0.0, {}, {}, {}, errs)

    conns, warns = resolve_inserters(layout)
    machines_by_id = {m.id: m for m in layout.machines}
    conns_by_id = {c.id: c for c in conns}

    # Group inserters by source and sink for quick lookup.
    inserters_from_machine: dict[str, list[str]] = {}  # machine_id → inserter ids picking from it
    inserters_to_machine_by_item: dict[str, dict[str, list[str]]] = {}
    #                                       # machine_id → item → inserter ids delivering
    producers_on_belt: dict[str, list[str]] = {b.id: [] for b in layout.belts}
    consumers_on_belt: dict[str, list[str]] = {b.id: [] for b in layout.belts}

    for c in conns:
        if c.item is None:
            continue
        if c.source_kind == "machine":
            inserters_from_machine.setdefault(c.source_id, []).append(c.id)
        elif c.source_kind == "belt":
            consumers_on_belt[c.source_id].append(c.id)
        if c.sink_kind == "machine":
            inserters_to_machine_by_item.setdefault(c.sink_id, {}) \
                .setdefault(c.item, []).append(c.id)
        elif c.sink_kind == "belt":
            producers_on_belt[c.sink_id].append(c.id)

    # Miners as implicit belt producers: a miner whose drop_position tile lies on
    # a belt (with matching item) feeds that belt directly, no inserter needed.
    # This is the ONLY way a miner produces output in the tightened sim.
    # miner_on_belt[belt_id] = list of (miner_id, tile_index_on_belt)
    miner_on_belt: dict[str, list[tuple[str, int]]] = {b.id: [] for b in layout.belts}
    miner_output_belt: dict[str, str | None] = {}  # miner_id → belt_id (or None if idle)
    for m in layout.machines:
        if _machine_kind(m.type) != "miner":
            continue
        dp = m.drop_position()
        found = False
        for b in layout.belts:
            for idx, t in enumerate(b.tiles):
                if (t.x, t.y) == dp and b.item == m.target_resource:
                    miner_on_belt[b.id].append((m.id, idx))
                    miner_output_belt[m.id] = b.id
                    found = True
                    break
            if found:
                break
        if not found:
            miner_output_belt[m.id] = None
            warns.append(
                f"miner {m.id}: drop_position {dp} has no matching-item belt tile; "
                f"drill produces 0 (needs belt carrying {m.target_resource!r})"
            )

    # Nominal rates
    machine_nominal = {m.id: _nominal_output_rate(m) for m in layout.machines}
    machine_coal_nominal = {m.id: _nominal_coal_rate(m) for m in layout.machines}

    # State
    machine_rate = dict(machine_nominal)
    inserter_rate = {c.id: 0.0 for c in conns}
    belt_flow = {b.id: 0.0 for b in layout.belts}

    inserters_by_id = {i.id: i for i in layout.inserters}

    def _demand_of_inserter(insr_id: str) -> float:
        """How much this inserter WANTS to move, capped at its tier throughput."""
        c = conns_by_id[insr_id]
        tp = _inserter_throughput(inserters_by_id[insr_id])
        if c.sink_kind == "machine":
            m = machines_by_id[c.sink_id]
            nominal = machine_nominal[m.id]
            if c.item == FUEL_ITEM and _machine_kind(m.type) == "furnace":
                required_total = machine_coal_nominal[m.id]
            else:
                inputs = _machine_input_items(m)
                if c.item not in inputs:
                    return 0.0
                r = RECIPES[m.recipe]
                out_amt = next(iter(r.products.values()))
                required_total = inputs[c.item] * nominal / out_amt
            siblings = inserters_to_machine_by_item.get(m.id, {}).get(c.item, [])
            share = required_total / max(1, len(siblings))
            return min(tp, share)
        return tp

    def _supply_available_to_inserter(insr_id: str) -> float:
        """How much this inserter CAN pick up from its source, given source state."""
        c = conns_by_id[insr_id]
        tp = _inserter_throughput(inserters_by_id[insr_id])
        if c.source_kind == "machine":
            m = machines_by_id[c.source_id]
            out = machine_rate[m.id]
            siblings = inserters_from_machine.get(m.id, [])
            share = out / max(1, len(siblings))
            return min(tp, share)
        if c.source_kind == "belt":
            return tp
        return 0.0

    for _ in range(MAX_ITER):
        prev_machine = dict(machine_rate)

        # --- Pass A: inserters going FROM machines (their rate = min(supply, demand)) ---
        for c in conns:
            if c.item is None:
                inserter_rate[c.id] = 0.0
                continue
            if c.source_kind == "machine" and c.sink_kind in ("machine", "belt"):
                supply = _supply_available_to_inserter(c.id)
                demand = _demand_of_inserter(c.id)
                inserter_rate[c.id] = min(supply, demand)
            # belt-source inserters get set in belt pass

        # --- Pass B: for each belt, compute FCFS flow ---
        # Producers on a belt come from two sources:
        #   1. Inserters whose drop tile is on the belt (`producers_on_belt`).
        #   2. Miners whose drop_position tile is on the belt (`miner_on_belt`).
        # Both are combined into one FCFS list ordered by their position on the belt.
        for b in layout.belts:
            # Build unified producer entries: (kind, id, pos_on_belt, nominal_rate)
            prod_entries: list[tuple[str, str, int, float]] = []
            for pid in producers_on_belt[b.id]:
                prod_entries.append(
                    ("inserter", pid, conns_by_id[pid].sink_pos or 0, inserter_rate[pid])
                )
            for mid, mpos in miner_on_belt[b.id]:
                prod_entries.append(("miner", mid, mpos, machine_rate[mid]))
            prod_entries.sort(key=lambda e: e[2])

            cons = consumers_on_belt[b.id]
            cons_sorted = sorted(cons, key=lambda cid: conns_by_id[cid].source_pos or 0)
            consumer_demands = [_demand_of_inserter(cid) for cid in cons_sorted]

            # No-consumer belts are treated as feeding an implicit chest sink.
            # This mirrors to_fle.py, which auto-places an inserter+steel-chest
            # at each dead-end belt tip in the FLE build so sim and FLE agree.
            consumer_cap = sum(consumer_demands) if cons_sorted else _belt_speed(b)
            total = min(sum(e[3] for e in prod_entries), consumer_cap, _belt_speed(b))
            belt_flow[b.id] = total

            # FCFS distribution to producers (update inserter rates only;
            # miner rates stay at nominal — belt cap is an emergent lossy behavior).
            remaining = total
            for kind, pid, _pos, nom in prod_entries:
                got = min(nom, remaining)
                if kind == "inserter":
                    inserter_rate[pid] = got
                remaining -= got
            # Consumers
            remaining = total
            for cid, dem in zip(cons_sorted, consumer_demands):
                got = min(dem, remaining)
                inserter_rate[cid] = got
                remaining -= got

        # --- Pass C: update machine rates from inserter supplies ---
        for m in layout.machines:
            kind = _machine_kind(m.type)
            nominal = machine_nominal[m.id]
            if nominal == 0.0:
                machine_rate[m.id] = 0.0
                continue
            if kind == "miner":
                # Miner runs at nominal only if a matching-item belt exists at
                # drop_position (checked earlier). Otherwise output = 0.
                machine_rate[m.id] = nominal if miner_output_belt.get(m.id) else 0.0
                continue
            inputs = _machine_input_items(m)
            supply: dict[str, float] = {}
            for item, insr_ids in inserters_to_machine_by_item.get(m.id, {}).items():
                supply[item] = sum(inserter_rate[i] for i in insr_ids)
            # ratio_i = supply_i / (required_at_nominal_i)
            r = RECIPES[m.recipe]
            out_amt = next(iter(r.products.values()))
            ratios: list[float] = []
            for item, amt in inputs.items():
                required = amt * nominal / out_amt
                if required <= 0:
                    continue
                ratios.append(min(1.0, supply.get(item, 0.0) / required))
            if kind == "furnace":
                required_coal = machine_coal_nominal[m.id]
                if required_coal > 0:
                    ratios.append(min(1.0, supply.get(FUEL_ITEM, 0.0) / required_coal))
            actual = nominal * (min(ratios) if ratios else 1.0)
            # Output-extraction rule: a machine's steady-state rate is bounded
            # by the rate at which output actually leaves the machine. Real
            # Factorio: if extract inserters can't deliver (belt full, no
            # consumer), the output slot fills and the machine stops.
            # We enforce this by capping actual at total inserter-extraction rate.
            # (Inserter rates were already updated by Pass B above, which
            # accounts for belt FCFS and consumer demand.)
            # EXCEPTION: machines producing items in `sink_items` (default:
            # only green-science pack, which has an implicit research-lab sink)
            # never back-pressure. Extra sinks can be added via the
            # `implicit_sinks` param for unit tests that exercise the sim on
            # isolated chain segments.
            output_item = _machine_output_item(m)
            if output_item not in sink_items:
                extractors = inserters_from_machine.get(m.id, [])
                total_extraction = sum(inserter_rate[i] for i in extractors)
                actual = min(actual, total_extraction)
            machine_rate[m.id] = actual

        # Convergence check
        delta = max(abs(machine_rate[k] - prev_machine[k]) for k in machine_rate) \
            if machine_rate else 0.0
        if delta < CONVERGENCE_TOL:
            break

    gs_rate = sum(
        machine_rate[m.id] for m in layout.machines if m.recipe == GREEN_SCIENCE_ITEM
    )

    return SimResult(
        green_science_rate=gs_rate,
        machine_rate=machine_rate,
        inserter_rate=inserter_rate,
        belt_flow=belt_flow,
        errors=warns,
    )
