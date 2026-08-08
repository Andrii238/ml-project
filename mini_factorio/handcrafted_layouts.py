"""Handcrafted reference layouts used for FLE cross-check + sim regression.

Each function returns a valid Layout whose expected sim rate is documented in
its docstring. These serve as (a) golden regression targets for the sim, and
(b) the FLE cross-check batch that closes plan.md's Task 2 ship gate.

Layouts:
- `iron_plate_smoke` — miner→belt→furnace→belt. Expected iron-plate: 0.3125/s.
- `iron_gear_no_extractor` — iron chain then gear assembler with no output
  inserter. Output-consumer rule ⇒ gear rate = 0.
- `iron_gear_with_extractor` — same but gear assembler has extractor onto a
  gear belt (items fall off the end). Expected gear rate: 0.156/s (iron-limited).
- `green_science_min` — full green-science chain on 20x20.
"""
from __future__ import annotations

from .layout import Belt, BeltTile, Inserter, Layout, Machine, Resource


def iron_plate_smoke() -> Layout:
    """Miner → belt → inserter → furnace → output inserter → belt.

    All flows valid; furnace has an output inserter dropping onto a plate belt,
    so output-consumer rule is satisfied. Expected iron-plate rate = 0.3125/sec
    (stone-furnace nominal on iron-plate recipe: crafting_speed=1.0, time=3.2s).
    """
    return Layout(
        grid_size=(16, 16),
        resources=[
            Resource(type="iron-ore", x=0, y=0, size=3),
            Resource(type="coal", x=0, y=8, size=3),
        ],
        machines=[
            # Iron miner facing east; drop_position = (3, 1)
            Machine(id="mi", type="electric-mining-drill", x=0, y=0,
                    direction="east", target_resource="iron-ore"),
            # Coal miner facing east; drop_position = (3, 9)
            Machine(id="mc", type="electric-mining-drill", x=0, y=8,
                    direction="east", target_resource="coal"),
            # Iron furnace at (7, 4). Occupies (7,4),(8,4),(7,5),(8,5).
            Machine(id="f_iron", type="stone-furnace", x=7, y=4, recipe="iron-plate"),
        ],
        belts=[
            # Iron ore: (3,1) → east through (4,1)(5,1), then south through (5,2)(5,3)(5,4)
            # so ore ends up on (5,4) — the tile 2 west of furnace top-left,
            # picked up by inserter at (6,4) dir east.
            Belt(id="b_iron_ore", item="iron-ore", tiles=[
                BeltTile(x=3, y=1, direction="east"),
                BeltTile(x=4, y=1, direction="east"),
                BeltTile(x=5, y=1, direction="south"),
                BeltTile(x=5, y=2, direction="south"),
                BeltTile(x=5, y=3, direction="south"),
                BeltTile(x=5, y=4, direction="east"),
            ]),
            # Coal: (3,9) → east then north to (5,5) so inserter at (6,5) picks from it.
            Belt(id="b_coal", item="coal", tiles=[
                BeltTile(x=3, y=9, direction="east"),
                BeltTile(x=4, y=9, direction="east"),
                BeltTile(x=5, y=9, direction="north"),
                BeltTile(x=5, y=8, direction="north"),
                BeltTile(x=5, y=7, direction="north"),
                BeltTile(x=5, y=6, direction="north"),
                BeltTile(x=5, y=5, direction="north"),  # pickup tile for coal inserter at (6,5)
            ]),
            # Output iron-plate belt: starts at (10, 4), collects plate from furnace via inserter at (9,4).
            Belt(id="b_iron_plate", item="iron-plate", tiles=[
                BeltTile(x=10, y=4, direction="east"),
                BeltTile(x=11, y=4, direction="east"),
                BeltTile(x=12, y=4, direction="east"),
            ]),
        ],
        inserters=[
            # Feed iron ore into furnace: pickup (5,4), drop (7,4). Inserter at (6,4) dir east.
            Inserter(id="i_ore_in", x=6, y=4, direction="east"),
            # Feed coal into furnace: pickup (5,6) belt, drop (7,6)? No — furnace bottom is y=5,
            # so drop into (5,5)? Not adjacent. Fix: coal inserter at (6,5) dir east: pickup (5,5) drop (7,5).
            # But belt (5,6) isn't at (5,5). Adjust belt to include (5,5).
            Inserter(id="i_coal_in", x=6, y=5, direction="east"),
            # Extract iron-plate: pickup (8,4) furnace, drop (10,4) belt. Inserter at (9,4) dir east.
            Inserter(id="i_plate_out", x=9, y=4, direction="east"),
        ],
    )


def _iron_smelting_20x20() -> tuple[list[Machine], list[Belt], list[Inserter], list[Resource]]:
    """Shared iron+coal smelting module.

    Layout:
      Iron ore (0-2, 0-2), iron miner at (0,0) east → drop (3,1).
      Iron-ore belt: (3,1) east → (5,1). Inserter at (6,1) east → iron furnace at (7,1) 2x2.
      Coal ore (0-2, 5-7), coal miner at (0,5) east → drop (3,6).
      Coal belt: (3,6)E(4,6)E(5,6)E(6,6)E(7,6)N(7,5)N(7,4)N.
      Coal inserter at (7,3) dir north: pickup (7,4)=belt, drop (7,2)=furnace bottom.
      Iron-plate extract inserter at (9,1) dir east: pickup (8,1)=furnace, drop (10,1)=belt.
      Iron-plate belt starts at (10,1) direction east.

    Total footprint used: rows 0-7, cols 0-10. Rest of 20x20 grid is free for
    downstream chains.
    """
    resources = [
        Resource(type="iron-ore", x=0, y=0, size=3),
        Resource(type="coal", x=0, y=5, size=3),
    ]
    machines = [
        Machine(id="mi", type="electric-mining-drill", x=0, y=0,
                direction="east", target_resource="iron-ore"),
        Machine(id="mc", type="electric-mining-drill", x=0, y=5,
                direction="east", target_resource="coal"),
        Machine(id="f_iron", type="stone-furnace", x=7, y=1, recipe="iron-plate"),
    ]
    belts = [
        Belt(id="b_iron_ore", item="iron-ore", tiles=[
            BeltTile(x=3, y=1, direction="east"),
            BeltTile(x=4, y=1, direction="east"),
            BeltTile(x=5, y=1, direction="east"),
        ]),
        Belt(id="b_coal", item="coal", tiles=[
            BeltTile(x=3, y=6, direction="east"),
            BeltTile(x=4, y=6, direction="east"),
            BeltTile(x=5, y=6, direction="east"),
            BeltTile(x=6, y=6, direction="east"),
            BeltTile(x=7, y=6, direction="north"),
            BeltTile(x=7, y=5, direction="north"),
            BeltTile(x=7, y=4, direction="north"),
        ]),
    ]
    inserters = [
        Inserter(id="i_ore_in", x=6, y=1, direction="east"),
        Inserter(id="i_coal_in", x=7, y=3, direction="north"),
        Inserter(id="i_iron_out", x=9, y=1, direction="east"),
    ]
    return machines, belts, inserters, resources


def iron_gear_with_extractor() -> Layout:
    """Iron smelting + gear assembler with output extractor onto a gear belt.

    Iron chain feeds one gear assembler (needs 2 iron-plate per gear). Iron
    supply = 0.3125/s ⇒ gear rate ≤ 0.156/s. Gear assembler has an output
    inserter dropping onto a short belt (items fall off the end), so
    output-consumer rule is satisfied. Expected gear rate: 0.1563/s.
    """
    machines, belts, inserters, resources = _iron_smelting_20x20()

    # Iron-plate belt: from (10,1) east, then curl south to feed a gear assembler
    # placed at (12,3) 3x3 → tiles (12,3)-(14,5).
    # Route: (10,1)E → (11,1)S → (11,2)S → (11,3)S → (11,4)S — ends at (11,4).
    # Gear feed inserter at (11,4)? overlaps belt. Change: end belt at (11,3), then
    # inserter at (11,4) picks from (11,3) drop into (11,5)? But (11,5) is gear tile.
    # Simpler: end belt at (10,3), inserter at (11,3) dir east: pickup (10,3), drop (12,3)=gear.
    belts.append(Belt(id="b_iron_plate", item="iron-plate", tiles=[
        BeltTile(x=10, y=1, direction="south"),
        BeltTile(x=10, y=2, direction="south"),
        BeltTile(x=10, y=3, direction="south"),
    ]))

    machines.append(Machine(id="a_gear", type="assembling-machine-1",
                            x=12, y=3, recipe="iron-gear-wheel"))
    inserters.append(Inserter(id="i_plate_to_gear", x=11, y=3, direction="east"))

    # Gear extract: inserter at (15,4) dir east: pickup (14,4)=gear, drop (16,4)=belt.
    inserters.append(Inserter(id="i_gear_out", x=15, y=4, direction="east"))
    belts.append(Belt(id="b_gear", item="iron-gear-wheel", tiles=[
        BeltTile(x=16, y=4, direction="east"),
        BeltTile(x=17, y=4, direction="east"),
    ]))

    return Layout(
        grid_size=(20, 20),
        resources=resources,
        machines=machines,
        inserters=inserters,
        belts=belts,
    )


def iron_gear_no_extractor() -> Layout:
    """Same as iron_gear_with_extractor but without the gear output inserter+belt.

    Under output-consumer rule: no inserter picks from a_gear ⇒ gear rate = 0.
    Expected gear rate: 0.0000/s.
    """
    lay = iron_gear_with_extractor()
    # Remove the gear output inserter and belt.
    lay.inserters = [i for i in lay.inserters if i.id != "i_gear_out"]
    lay.belts = [b for b in lay.belts if b.id != "b_gear"]
    return lay


def belt_asm_chain() -> Layout:
    """Iron smelting → gear assembler → transport-belt assembler.

    Tests a 2-assembler chain with the belt assembler taking iron-plate AND
    iron-gear as inputs (multi-input recipe). Iron supply = 0.3125/s.

    Rate math:
    - iron-plate/s = 0.3125 available.
    - gear asm: needs 2 iron/gear. Splits iron with belt asm 1:1? Actually iron
      goes: some to gear asm (2×gear_rate), some to belt asm (1×belt_asm_rate).
    - transport-belt recipe: 1 iron + 1 gear → 2 belts, time 0.5s.
      belt_asm nominal = 0.5*2/0.5 = 2/s. Needs 1 iron/s + 1 gear/s.
    - gear asm nominal = 1/s. Needs 2 iron/s.
    - Combined iron demand at nominal = 2 (gear) + 1 (belt asm direct) = 3/s.
      Available = 0.3125/s → limiting ratio = 0.3125/3 = 0.1042.
    - Actual gear rate = 1 × 0.1042 = 0.1042/s.
    - Actual belt asm iron supply = 1 × 0.1042 = 0.1042/s. Gear supply
      = 0.1042 (matches demand). Ratio = 0.1042.
    - Actual belt rate = 2 × 0.1042 = 0.2083/s (transport-belt items).

    Actual sim result: a_gear = 0.156/s (iron-limited at 0.3125/2), a_belt = 0.
    Reason: with a single iron furnace (supply 0.3125/s) and FCFS belt
    allocation, the upstream gear inserter (demand up to 0.83/s) consumes ALL
    the iron on the belt, starving the belt asm's iron feed. This is a real
    Factorio behavior and a valid sim result — but for FLE cross-check purposes
    it degenerates to a trivial 0=0 comparison on a_belt. The value of this
    layout is confirming FLE agrees with sim's FCFS starvation.
    """
    machines, belts, inserters, resources = _iron_smelting_20x20()

    # Iron-plate belt: (10,1)S(10,2)S(10,3)S(10,4)S — long enough to serve two consumers.
    belts.append(Belt(id="b_iron_plate", item="iron-plate", tiles=[
        BeltTile(x=10, y=1, direction="south"),
        BeltTile(x=10, y=2, direction="south"),
        BeltTile(x=10, y=3, direction="south"),
        BeltTile(x=10, y=4, direction="south"),
    ]))

    # Gear assembler at (12,1) 3x3 → (12-14, 1-3).
    machines.append(Machine(id="a_gear", type="assembling-machine-1",
                            x=12, y=1, recipe="iron-gear-wheel"))
    # Iron-plate → gear feed: inserter at (11,1) dir east: pickup (10,1)=belt, drop (12,1)=gear.
    inserters.append(Inserter(id="i_plate_to_gear", x=11, y=1, direction="east"))
    # Gear extract: inserter at (15,2) dir east: pickup (14,2)=gear, drop (16,2)=gear belt.
    inserters.append(Inserter(id="i_gear_out", x=15, y=2, direction="east"))
    # Gear belt goes south to feed belt asm (last tile must be at (16,6) so the
    # gear→belt-asm inserter at (15,6) dir west can pick from (16,6)).
    belts.append(Belt(id="b_gear", item="iron-gear-wheel", tiles=[
        BeltTile(x=16, y=2, direction="south"),
        BeltTile(x=16, y=3, direction="south"),
        BeltTile(x=16, y=4, direction="south"),
        BeltTile(x=16, y=5, direction="south"),
        BeltTile(x=16, y=6, direction="south"),
    ]))

    # Belt assembler at (12,6) 3x3 → (12-14, 6-8). Needs iron-plate + gear.
    machines.append(Machine(id="a_belt", type="assembling-machine-1",
                            x=12, y=6, recipe="transport-belt"))
    # Iron-plate → belt asm feed: inserter at (11,6) dir east: pickup (10,6)... need iron-plate belt at (10,6).
    # Extend iron-plate belt down to (10,6).
    belts[-2].tiles.extend([
        BeltTile(x=10, y=5, direction="south"),
        BeltTile(x=10, y=6, direction="south"),
    ])
    inserters.append(Inserter(id="i_plate_to_belt", x=11, y=6, direction="east"))
    # Gear → belt asm feed: inserter at (15,6) dir west: pickup (16,6)=gear belt, drop (14,6)=belt asm.
    inserters.append(Inserter(id="i_gear_to_belt", x=15, y=6, direction="west"))
    # Belt asm output extract: inserter at (13,9) dir south: pickup (13,8)=asm, drop (13,10)=output belt.
    inserters.append(Inserter(id="i_belt_out", x=13, y=9, direction="south"))
    belts.append(Belt(id="b_out_belts", item="transport-belt", tiles=[
        BeltTile(x=13, y=10, direction="south"),
        BeltTile(x=13, y=11, direction="south"),
    ]))

    return Layout(
        grid_size=(20, 20),
        resources=resources,
        machines=machines,
        inserters=inserters,
        belts=belts,
    )
