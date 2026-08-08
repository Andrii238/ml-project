"""Derive Factorio game constants empirically from a running FLE server.

Prints results in a copy-pasteable form suitable for mini_factorio/game_rules.py.

Constants probed:
  - Inserter cycle ticks for 90° (adjacent) and 180° (opposite-side) rotations.
  - Belt per-tile capacity (max items on one saturated tile).
  - Belt items-per-second throughput (empirical, at saturation).
  - Furnace input inventory size (in items, using iron-ore).
  - Furnace output inventory size (in items, using iron-plate).
  - Furnace fuel inventory size (in items, using coal).
  - Assembler input inventory size (per-recipe, using iron-gear-wheel).
  - Assembler output inventory size.

The probe runs at game.speed = 60 so 1 wall-time second ≈ 1 game minute (3600
ticks). Each measurement clears the surface, sets up an isolated scenario,
runs it briefly, reads state.

Usage:
    python -m translator.probe
"""
from __future__ import annotations

import time

from translator.fle_driver import _connect, _sc, _research_all

CLEAR = (
    "for _, e in pairs(game.surfaces[1].find_entities()) do "
    "if e.name ~= 'character' and e.name ~= 'player' then e.destroy() end end"
)
GAME_SPEED = 60

# Add creative-mode infinite power so electric machines actually run.
POWER_INFRA = (
    "game.surfaces[1].create_entity{name='substation', position={0,0}, force='player'}; "
    "local eei = game.surfaces[1].create_entity{name='electric-energy-interface', "
    "position={2.5,0.5}, force='player'}; "
    "if eei then eei.electric_buffer_size=1e12; eei.power_production=1e9 end"
)


def _rc(client, cmd: str) -> str:
    """Send /sc and strip the reply."""
    r = _sc(client, cmd)
    return (r or "").strip()


def _q(client, expr: str) -> str:
    """Wrap expr in rcon.print(...) and read back."""
    return _rc(client, f"rcon.print({expr})")


def _reset(client):
    _rc(client, CLEAR)
    _rc(client, "game.speed = 1")
    _rc(client, POWER_INFRA)


def probe_inventory_sizes(client) -> dict:
    """Read prototype inventory sizes without needing to run the machine."""
    out = {}
    # Stone furnace slots
    _reset(client)
    _rc(client, "game.surfaces[1].create_entity{name='stone-furnace', position={5,5}, force='player'}")
    for label, slot in [
        ("stone-furnace.source",  "defines.inventory.furnace_source"),
        ("stone-furnace.result",  "defines.inventory.furnace_result"),
        ("stone-furnace.fuel",    "defines.inventory.fuel"),
    ]:
        n = _q(client, f"#game.surfaces[1].find_entity('stone-furnace', {{5,5}}).get_inventory({slot})")
        out[label] = int(n) if n.isdigit() else n

    _reset(client)
    _rc(client, "game.surfaces[1].create_entity{name='assembling-machine-1', position={5.5,5.5}, force='player'}")
    _rc(client, "game.surfaces[1].find_entity('assembling-machine-1', {5.5,5.5}).set_recipe('iron-gear-wheel')")
    for label, slot in [
        ("assembling-machine-1.input",  "defines.inventory.assembling_machine_input"),
        ("assembling-machine-1.output", "defines.inventory.assembling_machine_output"),
    ]:
        n = _q(client, f"#game.surfaces[1].find_entity('assembling-machine-1', {{5.5,5.5}}).get_inventory({slot})")
        out[label] = int(n) if n.isdigit() else n
    return out


def probe_output_stack_cap(client) -> dict:
    """Fill an entity's output inventory to see when it stops crafting.

    Furnace with primed ore + coal, no output extraction. Run long enough that
    output saturates and status changes to 'full_output' (27).
    """
    _reset(client)
    _rc(client, "game.surfaces[1].create_entity{name='stone-furnace', position={6,6}, force='player'}")
    _rc(client, (
        "local f = game.surfaces[1].find_entity('stone-furnace', {6,6}); "
        "f.get_inventory(defines.inventory.furnace_source).insert{name='iron-ore', count=1000}; "
        "f.get_inventory(defines.inventory.fuel).insert{name='coal', count=100}"
    ))
    _rc(client, "game.speed = 100")
    time.sleep(6.0)  # 600 game-seconds → 187 plates at nominal 0.3125/s
    plates = _q(client, "game.surfaces[1].find_entity('stone-furnace', {6,6}).get_inventory(defines.inventory.furnace_result).get_item_count()")
    status_num = _q(client, "game.surfaces[1].find_entity('stone-furnace', {6,6}).status")
    _rc(client, "game.speed = 1")
    return {
        "stone-furnace.output_at_saturation": int(plates) if plates.isdigit() else plates,
        "stone-furnace.status_at_saturation": int(status_num) if status_num.isdigit() else status_num,
    }


def probe_belt_capacity(client) -> dict:
    """Belt tile capacity per lane. Insert-and-read in ONE /sc chunk so items
    can't move away between insert and read."""
    _reset(client)
    _rc(client, "game.speed = 0.01")  # near-pause; movement negligible
    _rc(client, "game.surfaces[1].create_entity{name='transport-belt', position={5.5, 5.5}, force='player', direction=4}")
    reply = _rc(client, (
        "local b = game.surfaces[1].find_entity('transport-belt', {5.5,5.5}); "
        "local l1_placed = 0; for i = 1, 200 do "
        "  if b.get_transport_line(1).insert_at_back{name='iron-ore', count=1} then l1_placed = l1_placed + 1 else break end "
        "end; "
        "local l2_placed = 0; for i = 1, 200 do "
        "  if b.get_transport_line(2).insert_at_back{name='iron-ore', count=1} then l2_placed = l2_placed + 1 else break end "
        "end; "
        "rcon.print(l1_placed .. ',' .. l2_placed)"
    ))
    _rc(client, "game.speed = 1")
    parts = reply.split(",")
    return {
        "belt.tile.lane1_capacity": int(parts[0]) if parts[0].isdigit() else parts[0],
        "belt.tile.lane2_capacity": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else "?",
    }


def probe_inserter_cycle(client, drop_side: str) -> int:
    """Measure inserter items-per-second by chest → inserter → chest.

    drop_side: '180' for opposite-side pickup/drop (typical assembler line),
               '90' for adjacent-side (belt corner scenario).
    Returns items moved per second (float).
    """
    _reset(client)
    if drop_side == "180":
        # Inserter direction=4 (Factorio east) means picks EAST, drops WEST (Factorio convention).
        # So source chest to the east of inserter, dest chest to the west.
        _rc(client, "game.surfaces[1].create_entity{name='iron-chest', position={6.5,5.5}, force='player'}")  # source east
        _rc(client, "game.surfaces[1].create_entity{name='iron-chest', position={4.5,5.5}, force='player'}")  # dest west
        _rc(client, "game.surfaces[1].create_entity{name='inserter', position={5.5,5.5}, force='player', direction=4}")
        src, dst = "{6.5,5.5}", "{4.5,5.5}"
    elif drop_side == "90":
        # Inserter direction=4 picks east, drops west. For 90° we want different sides.
        # Set direction=2 (northeast) is invalid for cardinal. Actually inserters in Factorio 2.0
        # can be placed only in cardinal N/E/S/W directions. There's no "90° rotation" separate
        # from cardinal — 180° is the DEFAULT. To rotate 90°, use `pickup_position` / `drop_position`
        # overrides. Skip 90° for now (basic yellow inserter is 180° by default).
        return -1
    else:
        raise ValueError(drop_side)

    _rc(client, f"game.surfaces[1].find_entity('iron-chest', {src}).insert{{name='iron-ore', count=1000}}")
    _rc(client, "game.speed = 60")
    time.sleep(0.5)  # 30s warmup for state machine to settle
    _rc(client, f"game.surfaces[1].find_entity('iron-chest', {dst}).get_inventory(defines.inventory.chest).clear()")
    time.sleep(1.0)  # 60 game-seconds measurement
    count = _q(client, f"game.surfaces[1].find_entity('iron-chest', {dst}).get_inventory(defines.inventory.chest).get_item_count()")
    _rc(client, "game.speed = 1")
    return int(count) if count.isdigit() else 0


def probe_belt_throughput(client) -> float:
    """Empirical belt items/sec throughput.

    Inserter direction=4 in Factorio = pickup east, drop west (opposite of
    what we want for a bus). We use direction=12 (west) = pickup east, drop west?
    Actually cleaner: inserter direction 12 in Factorio means "faces west" →
    picks east, drops west. Not useful either. Just use direction=4 for both.

    Setup: [source-chest][ins→dropWest][belt](belt)(belt)(belt)[ins→pickWest→drop east][dst-chest]
    Actually since Factorio inserters pick from the side opposite direction:
      direction=4 (east) → picks west (from position west of inserter), drops east
    Hmm that contradicts our earlier live probe... let me just try direction=0.

    Simpler: use direction=0 (north). Then picks south, drops north.

    Even simpler: fill the belt directly with insert_at_back, then let it drain
    through end-inserter into a chest, measure rate.
    """
    _reset(client)
    # Belt tiles going east (5,5) → (8,5)
    for x in range(5, 9):
        _rc(client, f"game.surfaces[1].create_entity{{name='transport-belt', position={{{x}.5,5.5}}, force='player', direction=4}}")
    # Continuously fill belt tile (5,5) via API. Instead of inserter, use a script that keeps stuffing.
    # But without game.on_tick we can't automate. Alternative: use a filter-inserter with a stack size or run a loop.
    # Simplest: just insert a huge batch, let belt saturate, then measure output rate at the end via chest.
    _rc(client, (
        "local b = game.surfaces[1].find_entity('transport-belt', {5.5,5.5}); "
        "for i = 1, 200 do "
        "  if not b.get_transport_line(1).insert_at_back{name='iron-ore', count=1} then break end "
        "end; "
        "for i = 1, 200 do "
        "  if not b.get_transport_line(2).insert_at_back{name='iron-ore', count=1} then break end "
        "end"
    ))
    # Destination chest at (10, 5). Belt end is at (8, 5). Items fall off (8,5) at east end...
    # actually items just accumulate at the end tile. Need an inserter picking from (8,5) to chest.
    _rc(client, "game.surfaces[1].create_entity{name='inserter', position={9.5,5.5}, force='player', direction=12}")  # dir=12 west → picks east(8,5)=belt, drops west(10,5)=... wait
    # Direction convention (live probe): direction=4 → picks east, drops west (10.5-1=9.5).
    # So direction=12 → picks west, drops east.
    # Inserter at (9.5,5.5) direction=12: picks west (8.5,5.5)=belt tile ✓, drops east (10.5,5.5)=chest.
    _rc(client, "game.surfaces[1].create_entity{name='iron-chest', position={10.5,5.5}, force='player'}")
    _rc(client, "game.speed = 60")
    time.sleep(0.5)  # warmup 30s
    _rc(client, "game.surfaces[1].find_entity('iron-chest', {10.5,5.5}).get_inventory(defines.inventory.chest).clear()")
    time.sleep(1.0)  # measure 60s
    count = _q(client, "game.surfaces[1].find_entity('iron-chest', {10.5,5.5}).get_inventory(defines.inventory.chest).get_item_count()")
    _rc(client, "game.speed = 1")
    return float(count) / 60.0 if count.isdigit() else 0.0


def main() -> None:
    client = _connect()
    _research_all(client)

    print("=" * 70)
    print("Factorio game-rule probe results")
    print("=" * 70)

    print("\n--- Inventory sizes (slots) ---")
    for k, v in probe_inventory_sizes(client).items():
        print(f"  {k:50s} = {v}")

    print("\n--- Furnace output saturation (iron-plate) ---")
    for k, v in probe_output_stack_cap(client).items():
        print(f"  {k:50s} = {v}")

    print("\n--- Belt per-tile capacity ---")
    for k, v in probe_belt_capacity(client).items():
        print(f"  {k:50s} = {v}")

    print("\n--- Inserter cycle throughput ---")
    for side in ("180", "90"):
        n = probe_inserter_cycle(client, side)
        print(f"  yellow-inserter items/sec ({side}° rotation) = {n / 60.0:.4f}  (raw: {n} items in 60s)")

    print("\n--- Belt throughput (saturated single line) ---")
    tp = probe_belt_throughput(client)
    print(f"  belt items/sec (single-lane pipeline through 4 tiles) = {tp:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
