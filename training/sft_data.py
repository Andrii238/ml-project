"""SFT data generation for the simplified env.

Takes a Layout with only 3 chests placed (like `empty_episode(seed)`) and
produces a list of edits that build a working single-assembler chain that
delivers green science to the output chest. Used as (prompt, completion)
pairs for LoRA SFT.

Approach:
- Pick a 3x3 free area for the assembler close to the input chests.
- BFS-route conveyors from each input chest into the assembler footprint.
- BFS-route a conveyor chain from an assembler border tile to the output chest.
- Verify simulate() > 0 before accepting.

Skips seeds where no valid layout can be constructed — SFT dataset is a
subset of the train split, not the whole thing.
"""
from __future__ import annotations

import json
from collections import deque
from typing import Iterable

from harness.prompt_builder import build_user_message
from mini_factorio.entities import DIR_DELTA, Direction
from mini_factorio.layout import Layout
from mini_factorio.random_layouts import empty_episode
from mini_factorio.simulator import simulate


# --------------------------------------------------------------- helpers

def _bfs_path(start: tuple[int, int], target_tiles: set[tuple[int, int]],
                blocked: set[tuple[int, int]], grid_size: tuple[int, int]
              ) -> list[tuple[int, int]] | None:
    w, h = grid_size
    if start in target_tiles:
        return [start]
    q: deque[list[tuple[int, int]]] = deque([[start]])
    seen: set[tuple[int, int]] = {start}
    while q:
        path = q.popleft()
        x, y = path[-1]
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in seen:
                continue
            if (nx, ny) in blocked and (nx, ny) not in target_tiles:
                continue
            seen.add((nx, ny))
            new_path = path + [(nx, ny)]
            if (nx, ny) in target_tiles:
                return new_path
            q.append(new_path)
    return None


def _direction_from_to(a: tuple[int, int], b: tuple[int, int]) -> Direction | None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    for name, delta in DIR_DELTA.items():
        if delta == (dx, dy):
            return name  # type: ignore[return-value]
    return None


def _pick_assembler_anchor(lay: Layout) -> tuple[int, int] | None:
    """A 3x3 free area near the input chests' centroid."""
    w, h = lay.grid_size
    ins = [c for c in lay.chests if c.kind.startswith("input-")]
    if ins:
        tx = sum(c.x for c in ins) / len(ins)
        ty = sum(c.y for c in ins) / len(ins)
    else:
        tx, ty = w / 2, h / 2
    occupied = {(c.x, c.y) for c in lay.chests}
    cands = []
    for ax in range(w - 2):
        for ay in range(h - 2):
            fp = {(ax + dx, ay + dy) for dx in range(3) for dy in range(3)}
            if fp & occupied:
                continue
            cx, cy = ax + 1, ay + 1
            cands.append(((cx - tx) ** 2 + (cy - ty) ** 2, ax, ay))
    if not cands:
        return None
    cands.sort()
    return cands[0][1], cands[0][2]


# --------------------------------------------------------------- oracle

def solve(lay: Layout) -> list[dict] | None:
    by_kind = {c.kind: c for c in lay.chests}
    if not all(k in by_kind for k in ("input-belts", "input-inserters", "output-science")):
        return None
    anchor = _pick_assembler_anchor(lay)
    if anchor is None:
        return None
    ax, ay = anchor
    fp = {(ax + dx, ay + dy) for dx in range(3) for dy in range(3)}
    blocked: set[tuple[int, int]] = {(c.x, c.y) for c in lay.chests} | fp

    edits: list[dict] = [{"op": "place_assembler", "id": "sft_a", "tier": 1,
                          "x": ax, "y": ay}]
    counter = [0]

    def _new_conv_id() -> str:
        counter[0] += 1
        return f"sft_c{counter[0]}"

    def _add_conveyor(x: int, y: int, direction: Direction) -> None:
        edits.append({"op": "place_conveyor", "id": _new_conv_id(),
                       "tier": 1, "x": x, "y": y, "direction": direction})
        blocked.add((x, y))

    border = set()
    for (fx, fy) in fp:
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            t = (fx + dx, fy + dy)
            if 0 <= t[0] < lay.grid_size[0] and 0 <= t[1] < lay.grid_size[1] and t not in fp:
                border.add(t)

    def _snapshot():
        return list(edits), set(blocked), counter[0]

    def _restore(s):
        edits[:] = s[0]
        blocked.clear(); blocked.update(s[1])
        counter[0] = s[2]

    # --- Input route: chest at C, first tile at C+d1 direction d1 (away from chest),
    # then BFS from C+2*d1 to any border tile B; final conveyor at B into footprint.
    def _route_input(chest: Chest) -> bool:
        for d1_name, (dx1, dy1) in DIR_DELTA.items():
            first_tile = (chest.x + dx1, chest.y + dy1)
            if not (0 <= first_tile[0] < lay.grid_size[0] and 0 <= first_tile[1] < lay.grid_size[1]):
                continue
            if first_tile in blocked:
                continue
            # If first_tile itself is a border tile adjacent to footprint AND
            # its neighbor toward footprint is a valid target, we can just
            # end here. Try that first.
            saved = _snapshot()
            _add_conveyor(first_tile[0], first_tile[1], d1_name)
            # Case A: first_tile is a border tile AND d1 already points into
            # the footprint. Single conveyor delivers. (A perpendicular crossing
            # here doesn't work because the two conveyors don't share flow.)
            if first_tile in border:
                for (fx, fy) in fp:
                    if abs(fx - first_tile[0]) + abs(fy - first_tile[1]) == 1:
                        if _direction_from_to(first_tile, (fx, fy)) == d1_name:
                            return True
            # Case B: BFS from the tile beyond first_tile to any border tile.
            beyond = (first_tile[0] + dx1, first_tile[1] + dy1)
            if not (0 <= beyond[0] < lay.grid_size[0] and 0 <= beyond[1] < lay.grid_size[1]) \
                    or beyond in blocked:
                _restore(saved)
                continue
            path = _bfs_path(beyond, border - blocked, blocked, lay.grid_size)
            if path is None:
                _restore(saved)
                continue
            # Emit path conveyors: first tile is `beyond`; first direction is from beyond to path[1].
            # `beyond` should follow d1 to be a straight continuation? Not strictly — the sim
            # accepts turns now, so `beyond` can have any direction.
            for i in range(len(path) - 1):
                d = _direction_from_to(path[i], path[i + 1])
                if d is None:
                    _restore(saved); return False
                _add_conveyor(path[i][0], path[i][1], d)
            # Final at path[-1] into footprint.
            last = path[-1]
            for (fx, fy) in fp:
                if abs(fx - last[0]) + abs(fy - last[1]) == 1:
                    d = _direction_from_to(last, (fx, fy))
                    if d is None:
                        continue
                    _add_conveyor(last[0], last[1], d)
                    return True
            _restore(saved)
        return False

    for kind in ("input-belts", "input-inserters"):
        if not _route_input(by_kind[kind]):
            return None

    # --- Output route: pick a border tile B, with a specific first-conveyor direction
    # pointing AWAY from an adjacent footprint tile F. Then BFS from B+direction to
    # a tile adjacent to output chest; end with final conveyor into chest.
    out = by_kind["output-science"]
    out_neigh_set = {(out.x + dx, out.y + dy)
                     for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
                     if 0 <= out.x + dx < lay.grid_size[0] and 0 <= out.y + dy < lay.grid_size[1]}
    for B in border:
        if B in blocked:
            continue
        for (fx, fy) in fp:
            if abs(fx - B[0]) + abs(fy - B[1]) != 1:
                continue
            # d = direction FROM F TO B — this is the direction the first conveyor
            # should have (so its upstream = F, which is on footprint).
            d = _direction_from_to((fx, fy), B)
            if d is None:
                continue
            saved = _snapshot()
            _add_conveyor(B[0], B[1], d)
            # If B is adjacent to out chest, add final conveyor into chest.
            if any(abs(B[0] - n[0]) + abs(B[1] - n[1]) == 0 for n in out_neigh_set):
                # B itself is adjacent to chest; final conveyor at B pointing into chest.
                # But B already has a conveyor with direction d. Add a crossing if needed.
                d_to_chest = _direction_from_to(B, (out.x, out.y))
                if d_to_chest is None:
                    _restore(saved); continue
                if d_to_chest == d:
                    return edits
                _add_conveyor(B[0], B[1], d_to_chest)
                return edits
            # Otherwise BFS from B+d to an out-chest neighbor.
            dx, dy = DIR_DELTA[d]
            beyond = (B[0] + dx, B[1] + dy)
            if not (0 <= beyond[0] < lay.grid_size[0] and 0 <= beyond[1] < lay.grid_size[1]) \
                    or beyond in blocked:
                _restore(saved); continue
            targets = out_neigh_set - blocked
            if not targets:
                _restore(saved); continue
            path = _bfs_path(beyond, targets, blocked, lay.grid_size)
            if path is None:
                _restore(saved); continue
            for i in range(len(path) - 1):
                dd = _direction_from_to(path[i], path[i + 1])
                if dd is None:
                    _restore(saved); path = None; break
                _add_conveyor(path[i][0], path[i][1], dd)
            if path is None:
                continue
            last = path[-1]
            d_final = _direction_from_to(last, (out.x, out.y))
            if d_final is None:
                _restore(saved); continue
            _add_conveyor(last[0], last[1], d_final)
            return edits
    return None


# --------------------------------------------------------------- SFT pair builders

def build_sft_pair(seed: int) -> dict | None:
    lay = empty_episode(seed=seed)
    edits = solve(lay)
    if edits is None:
        return None
    # Verify by applying + simulating.
    from harness.edit_applier import apply_edits
    from harness.edit_schema import parse_edit
    typed = []
    for d in edits:
        e, err = parse_edit(d)
        if e is None:
            return None
        typed.append(e)
    res = apply_edits(lay, typed)
    if res.applied != len(typed):
        return None
    sim = simulate(res.layout)
    if sim.green_science_rate <= 0:
        return None
    completion = json.dumps(edits, separators=(",", ":"))
    return {"seed": seed, "prompt": build_user_message(lay),
             "completion": completion, "sim_gs_rate": sim.green_science_rate}


def build_sft_dataset(seeds: Iterable[int]) -> list[dict]:
    """Build verified SFT examples.

    The old single-chain oracle remains above for reference, but the active
    dataset uses compact template-random examples with full-build and
    partial-repair prompts.
    """
    from training.template_sft_generator import build_template_dataset

    return build_template_dataset(seeds, variants_per_seed=4)
