"""Programmatic layout solver — produces valid working layouts for SFT seeds.

Given a random layout with resources but no machines, this solver returns a
JSON edit list that:
  - Places miners on iron-ore, coal patches with a valid drop direction.
  - Routes a belt from each miner drop tile toward a central factory area.
  - Places a stone-furnace with an inserter for iron-ore input, an inserter
    for coal input, and an output inserter dropping iron-plate onto a belt.

Goal: iron-plate/sec > 0 after apply_edits. That's a valid "working"
mini-layout the SFT model can imitate.

Multi-template strategy: layouts are classified by which grid edge iron+coal
patches sit on (`west`, `top`, `east`, `south`, `mixed`). Each edge case gets
a compact template. Layouts that don't fit any template return None and are
skipped from the SFT dataset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from mini_factorio.layout import Layout, DIR_DELTA, OPPOSITE


@dataclass
class SolveResult:
    edits: list[dict]
    template_name: str


# --- helpers ---

def _resource(layout: Layout, rtype: str):
    return next((r for r in layout.resources if r.type == rtype), None)


def _edge_of(r, W: int, H: int) -> str:
    if r.x == 0: return "west"
    if r.y == 0: return "north"
    if r.x + r.size == W: return "east"
    if r.y + r.size == H: return "south"
    return "interior"


def _miner_direction_for_edge(edge: str) -> str:
    """Face the miner AWAY from the edge, into the map."""
    return {
        "west": "east",
        "north": "south",
        "east": "west",
        "south": "north",
    }.get(edge, "east")


def _miner_drop(r, direction: str) -> tuple[int, int]:
    """Where a miner sitting on `r` with `direction` deposits ore."""
    # Miner size 3x3, top-left at (r.x, r.y). Center: (r.x+1, r.y+1).
    if direction == "north": return (r.x + 1, r.y - 1)
    if direction == "south": return (r.x + 1, r.y + 3)
    if direction == "east":  return (r.x + 3, r.y + 1)
    return (r.x - 1, r.y + 1)  # west


def _in_bounds(t: tuple[int, int], W: int, H: int) -> bool:
    return 0 <= t[0] < W and 0 <= t[1] < H


def _tiles_of(edit: dict) -> list[tuple[int, int]]:
    """List of grid tiles occupied by this edit."""
    op = edit["op"]
    if op == "add_entity":
        # Machine footprints: miner/assembler = 3x3, furnace = 2x2.
        w = h = 3
        if edit["type"] == "stone-furnace":
            w = h = 2
        return [(edit["x"] + dx, edit["y"] + dy) for dx in range(w) for dy in range(h)]
    if op == "add_inserter":
        return [(edit["x"], edit["y"])]
    if op == "add_belt":
        return [(t[0], t[1]) for t in edit["tiles"]]
    return []


def _collision(new_tiles: list[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
    return any(t in used for t in new_tiles)


# --- template ---

def _solve_iron_plate(layout: Layout) -> SolveResult | None:
    """Build a minimum iron-plate producer: iron miner + coal miner + belts + furnace + inserters + output belt.

    Compact template placed near iron miner. Coal reaches via a routed belt.
    Handles west/north/east/south edge placements for the iron patch.
    Fails if geometry (bounds, collisions) doesn't fit.
    """
    W, H = layout.grid_size
    iron = _resource(layout, "iron-ore")
    coal = _resource(layout, "coal")
    if iron is None or coal is None:
        return None

    used: set[tuple[int, int]] = set()
    # Reserve ore tiles other than iron/coal so we don't build on stone/copper patches.
    for r in layout.resources:
        for dx in range(r.size):
            for dy in range(r.size):
                if r.type in ("iron-ore", "coal"):
                    continue  # miners will sit on these
                used.add((r.x + dx, r.y + dy))
    # Any existing entities in a random_partial_layout are also occupied.
    for m in layout.machines:
        w = h = 3
        if m.type == "stone-furnace": w = h = 2
        for dx in range(w):
            for dy in range(h):
                used.add((m.x + dx, m.y + dy))
    for i in layout.inserters:
        used.add((i.x, i.y))
    for b in layout.belts:
        for t in b.tiles:
            used.add((t.x, t.y))

    iron_edge = _edge_of(iron, W, H)
    iron_dir = _miner_direction_for_edge(iron_edge)
    iron_drop = _miner_drop(iron, iron_dir)
    coal_edge = _edge_of(coal, W, H)
    coal_dir = _miner_direction_for_edge(coal_edge)
    coal_drop = _miner_drop(coal, coal_dir)

    if not _in_bounds(iron_drop, W, H) or not _in_bounds(coal_drop, W, H):
        return None

    edits: list[dict] = []

    def add(edit: dict) -> bool:
        tiles = _tiles_of(edit)
        if any(not _in_bounds(t, W, H) for t in tiles):
            return False
        if _collision(tiles, used):
            return False
        edits.append(edit)
        for t in tiles:
            used.add(t)
        return True

    # 1. Iron miner on iron patch. Miner footprint occupies iron patch — this
    # is intentional. Add to `used` after placing so subsequent placements
    # avoid the miner. Note: we DON'T add ore tiles to `used` initially for
    # iron/coal so the miner can sit on them.
    iron_miner = {
        "op": "add_entity", "id": "m_iron", "type": "electric-mining-drill",
        "x": iron.x, "y": iron.y, "direction": iron_dir,
        "target_resource": "iron-ore",
    }
    if not add(iron_miner):
        return None

    # 2. Coal miner similarly.
    coal_miner = {
        "op": "add_entity", "id": "m_coal", "type": "electric-mining-drill",
        "x": coal.x, "y": coal.y, "direction": coal_dir,
        "target_resource": "coal",
    }
    if not add(coal_miner):
        return None

    # 3. Belt tile at each drop position.
    iron_belt_start = {
        "op": "add_belt", "id": "b_iron", "item": "iron-ore",
        "tiles": [[iron_drop[0], iron_drop[1], iron_dir]],
    }
    if not add(iron_belt_start):
        return None

    coal_belt_start = {
        "op": "add_belt", "id": "b_coal", "item": "coal",
        "tiles": [[coal_drop[0], coal_drop[1], coal_dir]],
    }
    if not add(coal_belt_start):
        return None

    # 4. Extend iron belt 2 more tiles in iron_dir to give room for inserter+furnace.
    dxdy = DIR_DELTA[iron_dir]
    p1 = (iron_drop[0] + dxdy[0], iron_drop[1] + dxdy[1])
    p2 = (iron_drop[0] + 2*dxdy[0], iron_drop[1] + 2*dxdy[1])
    if not _in_bounds(p1, W, H) or not _in_bounds(p2, W, H) \
            or p1 in used or p2 in used:
        return None
    # Extend belt via extend_belt
    ext = {"op": "extend_belt", "id": "b_iron",
           "tiles": [[p1[0], p1[1], iron_dir], [p2[0], p2[1], iron_dir]]}
    edits.append(ext)
    used.add(p1); used.add(p2)

    # 5. Furnace placement. Furnace is 2x2. Leave ONE tile between belt and
    # furnace for the input inserter. So furnace top-left = p2 + 2*dxdy (+ perp offset).
    perp = (-dxdy[1], dxdy[0])
    furnace_tl = None
    for offset in (0, -1, 1):
        cand = (p2[0] + 2*dxdy[0] + perp[0] * offset,
                p2[1] + 2*dxdy[1] + perp[1] * offset)
        fx, fy = cand
        f_tiles = [(fx + dx, fy + dy) for dx in range(2) for dy in range(2)]
        if any(not _in_bounds(t, W, H) for t in f_tiles): continue
        if any(t in used for t in f_tiles): continue
        # Also the tile immediately after p2 (where the inserter will sit) must be free.
        ins_slot = (p2[0] + dxdy[0], p2[1] + dxdy[1])
        if ins_slot in used or not _in_bounds(ins_slot, W, H): continue
        # And ins_slot must be adjacent to some furnace tile.
        adj = any((abs(t[0] - ins_slot[0]) + abs(t[1] - ins_slot[1])) == 1 for t in f_tiles)
        if not adj: continue
        furnace_tl = cand
        break
    if furnace_tl is None:
        return None

    furnace = {
        "op": "add_entity", "id": "f_iron", "type": "stone-furnace",
        "x": furnace_tl[0], "y": furnace_tl[1], "recipe": "iron-plate",
    }
    if not add(furnace):
        return None

    # 6. Iron-input inserter: sits at a tile adjacent to both p2 (belt) and furnace.
    # Inserter direction = drop direction. It drops INTO the furnace.
    # Find a tile T such that T is adjacent to p2 AND has furnace on the opposite side.
    f_tiles_set = set((furnace_tl[0]+dx, furnace_tl[1]+dy) for dx in range(2) for dy in range(2))
    iron_ins_pos = None
    iron_ins_dir = None
    for dir_name, (dxi, dyi) in DIR_DELTA.items():
        cand = (p2[0] + dxi, p2[1] + dyi)
        opp = (cand[0] + dxi, cand[1] + dyi)
        if cand in used or not _in_bounds(cand, W, H): continue
        if opp not in f_tiles_set: continue
        iron_ins_pos = cand
        iron_ins_dir = dir_name
        break
    if iron_ins_pos is None:
        return None
    iron_ins = {"op": "add_inserter", "id": "i_iron", "x": iron_ins_pos[0],
                "y": iron_ins_pos[1], "direction": iron_ins_dir}
    if not add(iron_ins):
        return None

    # 7. Coal belt extension: route coal belt to a tile adjacent to furnace.
    # Naive routing: extend coal belt straight in coal_dir until it can reach the furnace
    # perimeter, then bend. Fallback: fail if too far.
    coal_dxdy = DIR_DELTA[coal_dir]
    coal_path = [coal_drop]
    cur = coal_drop
    for _ in range(20):
        nxt = (cur[0] + coal_dxdy[0], cur[1] + coal_dxdy[1])
        if not _in_bounds(nxt, W, H) or nxt in used:
            break
        # Check if any tile adjacent to `nxt` is a furnace tile.
        adj_furnace = any((abs(nxt[0] - ft[0]) + abs(nxt[1] - ft[1])) == 1 for ft in f_tiles_set)
        coal_path.append(nxt)
        cur = nxt
        if adj_furnace:
            break
    else:
        return None

    # After straight run, try one perpendicular step to reach furnace-adjacent tile.
    if not any((abs(cur[0] - ft[0]) + abs(cur[1] - ft[1])) == 1 for ft in f_tiles_set):
        perp = (-coal_dxdy[1], coal_dxdy[0])
        for sign in (-1, 1):
            step = (cur[0] + perp[0]*sign, cur[1] + perp[1]*sign)
            if not _in_bounds(step, W, H) or step in used:
                continue
            adj = any((abs(step[0] - ft[0]) + abs(step[1] - ft[1])) == 1 for ft in f_tiles_set)
            if adj:
                coal_path.append(step)
                # Direction of this last tile is the perpendicular.
                perp_dir = None
                for dn, dv in DIR_DELTA.items():
                    if dv == (perp[0]*sign, perp[1]*sign):
                        perp_dir = dn; break
                if perp_dir is None: return None
                break
        else:
            return None

    if len(coal_path) < 2:
        return None
    # Extend coal belt with all path tiles after coal_drop (which is already the start).
    extend_tiles = []
    for i in range(1, len(coal_path)):
        # Direction is the vector from previous to this tile.
        px, py = coal_path[i-1]
        cx, cy = coal_path[i]
        dvec = (cx - px, cy - py)
        dname = next((k for k, v in DIR_DELTA.items() if v == dvec), None)
        if dname is None: return None
        extend_tiles.append([cx, cy, dname])
        used.add((cx, cy))
    if extend_tiles:
        edits.append({"op": "extend_belt", "id": "b_coal", "tiles": extend_tiles})

    coal_belt_end = coal_path[-1]

    # 8. Coal inserter: adjacent to coal_belt_end, dropping into furnace.
    coal_ins_pos = None
    coal_ins_dir = None
    for dir_name, (dxi, dyi) in DIR_DELTA.items():
        cand = (coal_belt_end[0] + dxi, coal_belt_end[1] + dyi)
        opp = (cand[0] + dxi, cand[1] + dyi)
        if cand in used or not _in_bounds(cand, W, H): continue
        if opp not in f_tiles_set: continue
        coal_ins_pos = cand
        coal_ins_dir = dir_name
        break
    if coal_ins_pos is None:
        return None
    coal_ins = {"op": "add_inserter", "id": "i_coal", "x": coal_ins_pos[0],
                "y": coal_ins_pos[1], "direction": coal_ins_dir}
    if not add(coal_ins):
        return None

    # 9. Output inserter picking from furnace, dropping onto an output belt.
    # Find any tile adjacent to a furnace tile that's not already taken.
    out_ins_pos = None
    out_ins_dir = None
    out_drop_tile = None
    for ft in f_tiles_set:
        for dir_name, (dxi, dyi) in DIR_DELTA.items():
            cand = (ft[0] + dxi, ft[1] + dyi)
            opp = (cand[0] + dxi, cand[1] + dyi)
            if cand in used or not _in_bounds(cand, W, H): continue
            if opp in used or not _in_bounds(opp, W, H): continue
            # For an output inserter, the tile "opposite" (opp) must be empty
            # and we'll drop an output belt tile there.
            out_ins_pos = cand
            out_ins_dir = dir_name
            out_drop_tile = opp
            break
        if out_ins_pos: break
    if out_ins_pos is None:
        return None
    out_ins = {"op": "add_inserter", "id": "i_out", "x": out_ins_pos[0],
               "y": out_ins_pos[1], "direction": out_ins_dir}
    if not add(out_ins):
        return None

    # 10. Output belt (1 tile) at out_drop_tile, direction = same as inserter.
    out_belt = {"op": "add_belt", "id": "b_iron_plate", "item": "iron-plate",
                "tiles": [[out_drop_tile[0], out_drop_tile[1], out_ins_dir]]}
    if not add(out_belt):
        return None

    return SolveResult(edits=edits, template_name="iron_plate_v1")


def _solve_miner_only(layout: Layout, resource_type: str) -> SolveResult | None:
    """Simple template: place miner on the given resource + 3 belt tiles carrying ore."""
    W, H = layout.grid_size
    r = _resource(layout, resource_type)
    if r is None:
        return None
    used: set[tuple[int, int]] = set()
    for other in layout.resources:
        if other.type == resource_type:
            continue
        for dx in range(other.size):
            for dy in range(other.size):
                used.add((other.x + dx, other.y + dy))
    for m in layout.machines:
        w = h = 3
        if m.type == "stone-furnace": w = h = 2
        for dx in range(w):
            for dy in range(h):
                used.add((m.x + dx, m.y + dy))
    for i in layout.inserters:
        used.add((i.x, i.y))
    for b in layout.belts:
        for t in b.tiles:
            used.add((t.x, t.y))
    edge = _edge_of(r, W, H)
    direction = _miner_direction_for_edge(edge)
    drop = _miner_drop(r, direction)
    if not _in_bounds(drop, W, H) or drop in used:
        return None
    edits: list[dict] = []
    # Miner
    edits.append({
        "op": "add_entity", "id": f"m_{resource_type.split('-')[0]}",
        "type": "electric-mining-drill",
        "x": r.x, "y": r.y, "direction": direction,
        "target_resource": resource_type,
    })
    for dx in range(3):
        for dy in range(3):
            used.add((r.x + dx, r.y + dy))
    # Belt: start at drop, extend 2 more tiles in direction.
    dxdy = DIR_DELTA[direction]
    tiles = [[drop[0], drop[1], direction]]
    cur = drop
    for _ in range(2):
        nxt = (cur[0] + dxdy[0], cur[1] + dxdy[1])
        if not _in_bounds(nxt, W, H) or nxt in used:
            break
        tiles.append([nxt[0], nxt[1], direction])
        used.add(nxt)
        cur = nxt
    used.add(drop)
    edits.append({"op": "add_belt", "id": f"b_{resource_type.split('-')[0]}",
                  "item": resource_type, "tiles": tiles})
    return SolveResult(edits=edits, template_name=f"miner_only_{resource_type}")


def solve(layout: Layout) -> list[SolveResult]:
    """Return ALL working templates for this layout (miner+belt for each ore type).

    Empty list if none fit. Each SolveResult can become one SFT pair.
    """
    results: list[SolveResult] = []
    for rtype in ("iron-ore", "copper-ore", "coal", "stone"):
        r = _solve_miner_only(layout, rtype)
        if r is not None:
            results.append(r)
    ip = _solve_iron_plate(layout)
    if ip is not None:
        results.append(ip)
    return results


if __name__ == "__main__":
    # Smoke test: solve all 60 training layouts, report success rate.
    import sys
    sys.path.insert(0, ".")
    from mini_factorio.random_layouts import train_val_split
    from harness.edit_parser import parse_edits
    from harness.edit_applier import apply_edits
    from mini_factorio.reward import compute_reward
    from harness.edit_schema import EditList

    train, _ = train_val_split(60, 20)
    from mini_factorio.simulator import simulate
    n_templates = 0
    n_layouts_solved = 0
    templates_by_kind: dict[str, int] = {}
    for i, layout in enumerate(train):
        results = solve(layout)
        if not results:
            print(f"[{i:2d}] SKIP")
            continue
        n_layouts_solved += 1
        for r in results:
            n_templates += 1
            templates_by_kind[r.template_name] = templates_by_kind.get(r.template_name, 0) + 1
            el = EditList.model_validate({"edits": r.edits})
            ar = apply_edits(layout, el)
            print(f"[{i:2d}] {r.template_name}: applied {ar.n_applied}/{len(r.edits)}")
    print(f"\nSummary: {n_layouts_solved}/{len(train)} layouts got at least one template, "
          f"{n_templates} total pairs")
    print(f"by template: {templates_by_kind}")
