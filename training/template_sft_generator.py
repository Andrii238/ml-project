"""Template-random SFT example generator.

Generates verified (prompt, completion) pairs for green-science layout edits.
Prompts start from the same random chest-only layouts used in evaluation, then
completions build or repair a working green-science factory.

Every returned sample is accepted only if:
- completion parses and applies without errors,
- final simulation delivers green science,
- completion length is <= MAX_EDITS.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, Literal

from harness.edit_applier import apply_edits
from harness.edit_parser import MAX_EDITS
from harness.edit_schema import parse_edit
from harness.prompt_builder import build_user_message
from mini_factorio.entities import DIR_DELTA, Direction
from mini_factorio.layout import Assembler, Chest, ChestRates, Conveyor, Layout
from mini_factorio.random_layouts import empty_episode, sample_chest_rates
from mini_factorio.simulator import simulate

Mode = Literal["full", "partial"]


@dataclass(frozen=True)
class GeneratedPair:
    seed: int
    prompt: str
    completion: str
    sim_gs_rate: float
    n_assemblers: int
    tiers: tuple[int, ...]
    mode: Mode


@dataclass(frozen=True)
class _AsmSpec:
    id: str
    tier: int
    x: int
    y: int

    @property
    def footprint(self) -> set[tuple[int, int]]:
        return {(self.x + dx, self.y + dy) for dx in range(3) for dy in range(3)}


def _direction_from_to(a: tuple[int, int], b: tuple[int, int]) -> Direction | None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    for name, delta in DIR_DELTA.items():
        if delta == (dx, dy):
            return name  # type: ignore[return-value]
    return None


def _neighbors(t: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = t
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _in_bounds(t: tuple[int, int], grid: tuple[int, int]) -> bool:
    return 0 <= t[0] < grid[0] and 0 <= t[1] < grid[1]


def _paired_input_episode(seed: int, grid_size: tuple[int, int] = (20, 20)) -> Layout:
    """Chest-only episode with input chests adjacent/near-adjacent and output
    random. This keeps routing learnable while preserving varied locations."""
    rng = random.Random(seed)
    lay = Layout(grid_size=grid_size, chest_rates=sample_chest_rates(rng))
    w, h = grid_size

    bx = rng.randint(0, w - 1)
    by = rng.randint(0, h - 1)
    candidates = [(bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)]
    candidates = [t for t in candidates if _in_bounds(t, grid_size)]
    if not candidates:
        return empty_episode(seed, grid_size=grid_size)
    ix, iy = rng.choice(candidates)

    occupied = {(bx, by), (ix, iy)}
    out_choices = [(x, y) for x in range(w) for y in range(h) if (x, y) not in occupied]
    ox, oy = rng.choice(out_choices)
    lay.chests = [
        Chest(id="chest_input-belts", kind="input-belts", x=bx, y=by),
        Chest(id="chest_input-inserters", kind="input-inserters", x=ix, y=iy),
        Chest(id="chest_output-science", kind="output-science", x=ox, y=oy),
    ]
    return lay


def _bfs(start: tuple[int, int], targets: set[tuple[int, int]],
         blocked: set[tuple[int, int]], grid: tuple[int, int]
         ) -> list[tuple[int, int]] | None:
    if start in targets:
        return [start]
    q: list[list[tuple[int, int]]] = [[start]]
    seen = {start}
    while q:
        path = q.pop(0)
        for nxt in _neighbors(path[-1]):
            if nxt in seen or not _in_bounds(nxt, grid):
                continue
            if nxt in blocked and nxt not in targets:
                continue
            seen.add(nxt)
            new_path = path + [nxt]
            if nxt in targets:
                return new_path
            q.append(new_path)
    return None


def _add_conveyor(edits: list[dict], blocked: set[tuple[int, int]],
                  x: int, y: int, direction: Direction, tier: int = 1) -> None:
    cid = f"gen_c{sum(1 for e in edits if e['op'] == 'place_conveyor') + 1}"
    edits.append({"op": "place_conveyor", "id": cid, "tier": tier,
                  "x": x, "y": y, "direction": direction})
    blocked.add((x, y))


def _route_from_chest_to_machine(
    edits: list[dict],
    blocked: set[tuple[int, int]],
    chest: Chest,
    target_tile: tuple[int, int],
    machine_tile: tuple[int, int],
    grid: tuple[int, int],
    rng: random.Random,
) -> bool:
    """Route one item path from a chest to a conveyor tile that points into a machine."""
    final_dir = _direction_from_to(target_tile, machine_tile)
    if final_dir is None:
        return False

    dirs = list(DIR_DELTA.items())
    rng.shuffle(dirs)
    for d1_name, (dx, dy) in dirs:
        first = (chest.x + dx, chest.y + dy)
        if not _in_bounds(first, grid) or first in blocked:
            continue
        saved_edits = list(edits)
        saved_blocked = set(blocked)
        _add_conveyor(edits, blocked, first[0], first[1], d1_name)

        if first == target_tile:
            # Direct chest-adjacent input only works if the conveyor points into the machine.
            if d1_name == final_dir:
                return True
            edits[:] = saved_edits
            blocked.clear(); blocked.update(saved_blocked)
            continue

        beyond = (first[0] + dx, first[1] + dy)
        if not _in_bounds(beyond, grid) or beyond in blocked:
            edits[:] = saved_edits
            blocked.clear(); blocked.update(saved_blocked)
            continue

        path = _bfs(beyond, {target_tile}, blocked, grid)
        if path is None:
            edits[:] = saved_edits
            blocked.clear(); blocked.update(saved_blocked)
            continue

        ok = True
        for i, tile in enumerate(path):
            if tile == target_tile:
                direction = final_dir
            else:
                direction = _direction_from_to(tile, path[i + 1])
            if direction is None:
                ok = False
                break
            _add_conveyor(edits, blocked, tile[0], tile[1], direction)
        if ok:
            return True
        edits[:] = saved_edits
        blocked.clear(); blocked.update(saved_blocked)
    return False


def _route_from_machine_to_chest(
    edits: list[dict],
    blocked: set[tuple[int, int]],
    machine_tile: tuple[int, int],
    first_tile: tuple[int, int],
    chest: Chest,
    grid: tuple[int, int],
    rng: random.Random,
) -> bool:
    """Route green science from a machine-adjacent first tile to output chest."""
    first_dir = _direction_from_to(machine_tile, first_tile)
    if first_dir is None or first_tile in blocked or not _in_bounds(first_tile, grid):
        return False

    saved_edits = list(edits)
    saved_blocked = set(blocked)
    _add_conveyor(edits, blocked, first_tile[0], first_tile[1], first_dir)

    chest_neighbors = {t for t in _neighbors((chest.x, chest.y))
                       if _in_bounds(t, grid) and t not in blocked}
    if first_tile in chest_neighbors:
        to_chest = _direction_from_to(first_tile, (chest.x, chest.y))
        if to_chest == first_dir:
            return True

    dx, dy = DIR_DELTA[first_dir]
    beyond = (first_tile[0] + dx, first_tile[1] + dy)
    if not _in_bounds(beyond, grid) or beyond in blocked:
        edits[:] = saved_edits
        blocked.clear(); blocked.update(saved_blocked)
        return False

    rng_targets = list(chest_neighbors)
    rng.shuffle(rng_targets)
    path = _bfs(beyond, set(rng_targets), blocked, grid)
    if path is None:
        edits[:] = saved_edits
        blocked.clear(); blocked.update(saved_blocked)
        return False

    for i, tile in enumerate(path):
        if i == len(path) - 1:
            direction = _direction_from_to(tile, (chest.x, chest.y))
        else:
            direction = _direction_from_to(tile, path[i + 1])
        if direction is None:
            edits[:] = saved_edits
            blocked.clear(); blocked.update(saved_blocked)
            return False
        _add_conveyor(edits, blocked, tile[0], tile[1], direction)
    return True


def _choose_tiers(rng: random.Random, n: int) -> list[int]:
    if rng.random() < 0.7:
        tier = rng.choice((1, 2, 3))
        return [tier] * n
    return [rng.choice((1, 2, 3)) for _ in range(n)]


def _block_assemblers(rng: random.Random, n: int, grid: tuple[int, int]) -> list[_AsmSpec]:
    cols = min(2, n)
    rows = math.ceil(n / cols)
    stride = 4
    block_w = 3 + (cols - 1) * stride
    block_h = 3 + (rows - 1) * stride
    # Leave margin for chests and routing.
    min_x, min_y = 4, 4
    max_x = grid[0] - block_w - 4
    max_y = grid[1] - block_h - 4
    if max_x < min_x or max_y < min_y:
        raise ValueError("assembler block does not fit")
    base_x = rng.randint(min_x, max_x)
    base_y = rng.randint(min_y, max_y)
    tiers = _choose_tiers(rng, n)
    out: list[_AsmSpec] = []
    for i in range(n):
        col = i % cols
        row = i // cols
        out.append(_AsmSpec(id=f"gen_a{i + 1}", tier=tiers[i],
                            x=base_x + col * stride,
                            y=base_y + row * stride))
    return out


def _make_chest_layout(rng: random.Random, assemblers: list[_AsmSpec],
                       grid: tuple[int, int]) -> Layout:
    min_x = min(a.x for a in assemblers)
    max_x = max(a.x + 2 for a in assemblers)
    min_y = min(a.y for a in assemblers)
    max_y = max(a.y + 2 for a in assemblers)
    center_y = (min_y + max_y) // 2
    center_x = (min_x + max_x) // 2

    # Compact, non-overlapping chest placement around the planned factory.
    belt = (max(0, min_x - 3), center_y)
    inserter = (center_x, max(0, min_y - 3))
    output = (min(grid[0] - 1, max_x + 4), center_y)
    used = {belt, inserter, output}
    if len(used) < 3:
        raise ValueError("chest collision")
    return Layout(
        grid_size=grid,
        chest_rates=sample_chest_rates(rng),
        chests=[
            Chest(id="chest_input-belts", kind="input-belts", x=belt[0], y=belt[1]),
            Chest(id="chest_input-inserters", kind="input-inserters", x=inserter[0], y=inserter[1]),
            Chest(id="chest_output-science", kind="output-science", x=output[0], y=output[1]),
        ],
    )


def _build_full_pair(seed: int, variant: int, *, grid: tuple[int, int] = (20, 20)) -> GeneratedPair | None:
    """Build a full solution from a random chest-only episode.

    This deliberately matches evaluation: the prompt starts with random chest
    positions, and the completion must place assemblers/conveyors around them.
    """
    rng = random.Random(seed * 1009 + variant * 9176 + 17)
    initial = _paired_input_episode(seed * 1000 + variant, grid_size=grid)
    chests = {c.kind: c for c in initial.chests}
    if set(chests) != {"input-belts", "input-inserters", "output-science"}:
        return None

    wanted_n = 2 if rng.random() < 0.55 else 1
    n_options = [wanted_n, 1] if wanted_n == 2 else [1, 2]

    occupied_chests = {(c.x, c.y) for c in initial.chests}

    def _block_candidates(n: int) -> list[list[_AsmSpec]]:
        candidates: list[tuple[float, list[_AsmSpec]]] = []
        tiers = _choose_tiers(rng, n)
        if n == 1:
            for x in range(1, grid[0] - 3):
                for y in range(1, grid[1] - 3):
                    a = _AsmSpec(id="gen_a1", tier=tiers[0], x=x, y=y)
                    if a.footprint & occupied_chests:
                        continue
                    belt_target = (x - 1, y + 1)
                    ins_target = (x + 1, y - 1)
                    out_first = (x + 3, y + 1)
                    if not all(_in_bounds(t, grid) for t in (belt_target, ins_target, out_first)):
                        continue
                    cost = (
                        abs(chests["input-belts"].x - belt_target[0]) + abs(chests["input-belts"].y - belt_target[1])
                        + abs(chests["input-inserters"].x - ins_target[0]) + abs(chests["input-inserters"].y - ins_target[1])
                        + abs(chests["output-science"].x - out_first[0]) + abs(chests["output-science"].y - out_first[1])
                        + rng.random() * 0.25
                    )
                    candidates.append((cost, [a]))
        else:
            # Two horizontal assemblers with enough room for west/north inputs and east outputs.
            for x in range(1, grid[0] - 9):
                for y in range(1, grid[1] - 3):
                    left = _AsmSpec(id="gen_a1", tier=tiers[0], x=x, y=y)
                    right = _AsmSpec(id="gen_a2", tier=tiers[1], x=x + 6, y=y)
                    fp = left.footprint | right.footprint
                    if fp & occupied_chests:
                        continue
                    targets = [
                        (left.x - 1, y + 1), (left.x + 1, y - 1), (left.x + 3, y + 1),
                        (right.x - 1, y + 1), (right.x + 1, y - 1), (right.x + 3, y + 1),
                    ]
                    if not all(_in_bounds(t, grid) for t in targets):
                        continue
                    cost = 0.0
                    for a in (left, right):
                        belt_target = (a.x - 1, a.y + 1)
                        ins_target = (a.x + 1, a.y - 1)
                        out_first = (a.x + 3, a.y + 1)
                        cost += (
                            abs(chests["input-belts"].x - belt_target[0]) + abs(chests["input-belts"].y - belt_target[1])
                            + abs(chests["input-inserters"].x - ins_target[0]) + abs(chests["input-inserters"].y - ins_target[1])
                            + abs(chests["output-science"].x - out_first[0]) + abs(chests["output-science"].y - out_first[1])
                        )
                    candidates.append((cost + rng.random() * 0.25, [left, right]))
        candidates.sort(key=lambda c: c[0])
        return [c[1] for c in candidates[:40]]

    for n_assemblers in n_options:
        for assemblers in _block_candidates(n_assemblers):
            blocked = set(occupied_chests)
            for a in assemblers:
                blocked.update(a.footprint)

            edits: list[dict] = []
            for a in assemblers:
                edits.append({"op": "place_assembler", "id": a.id, "tier": a.tier,
                              "x": a.x, "y": a.y})

            ok = True
            for a in assemblers:
                ok = _route_from_chest_to_machine(
                    edits, blocked, chests["input-belts"],
                    target_tile=(a.x - 1, a.y + 1), machine_tile=(a.x, a.y + 1),
                    grid=grid, rng=rng,
                )
                if not ok:
                    break
                ok = _route_from_chest_to_machine(
                    edits, blocked, chests["input-inserters"],
                    target_tile=(a.x + 1, a.y - 1), machine_tile=(a.x + 1, a.y),
                    grid=grid, rng=rng,
                )
                if not ok:
                    break
                ok = _route_from_machine_to_chest(
                    edits, blocked, machine_tile=(a.x + 2, a.y + 1),
                    first_tile=(a.x + 3, a.y + 1), chest=chests["output-science"],
                    grid=grid, rng=rng,
                )
                if not ok:
                    break
            if not ok or len(edits) > MAX_EDITS:
                continue

            typed = []
            for raw in edits:
                e, err = parse_edit(raw)
                if e is None:
                    typed = []
                    break
                typed.append(e)
            if not typed:
                continue
            applied = apply_edits(initial, typed)
            if applied.errors or applied.applied != len(typed):
                continue
            sim = simulate(applied.layout)
            if sim.green_science_rate <= 0:
                continue

            completion = json.dumps(edits, separators=(",", ":"))
            return GeneratedPair(
                seed=seed,
                prompt=build_user_message(initial),
                completion=completion,
                sim_gs_rate=sim.green_science_rate,
                n_assemblers=len(assemblers),
                tiers=tuple(a.tier for a in assemblers),
                mode="full",
            )
    return None


def _partial_from_full(pair: GeneratedPair, seed: int, variant: int) -> GeneratedPair | None:
    """Create a partial-repair sample by deleting entities from a verified
    full solution. The completion re-adds the deleted entities.

    Deletion policy requested by Andrii:
    - delete 1-5 assemblers, bounded by how many exist,
    - delete 0-10 conveyors, bounded by how many exist.
    """
    rng = random.Random(seed * 811 + variant * 1193 + 42)
    raw_edits = json.loads(pair.completion)
    asm_idx = [i for i, e in enumerate(raw_edits) if e["op"] == "place_assembler"]
    conv_idx = [i for i, e in enumerate(raw_edits) if e["op"] == "place_conveyor"]
    if not asm_idx:
        return None

    n_del_asm = rng.randint(1, min(5, len(asm_idx)))
    n_del_conv = rng.randint(0, min(10, len(conv_idx)))
    deleted_idx = set(rng.sample(asm_idx, n_del_asm))
    if n_del_conv > 0:
        deleted_idx.update(rng.sample(conv_idx, n_del_conv))

    kept = [e for i, e in enumerate(raw_edits) if i not in deleted_idx]
    deleted = [e for i, e in enumerate(raw_edits) if i in deleted_idx]
    if not deleted or len(deleted) > MAX_EDITS:
        return None

    from training.reward_wrapper import layout_from_prompt

    initial = layout_from_prompt(pair.prompt)
    if initial is None:
        return None

    typed_kept = []
    for raw in kept:
        e, err = parse_edit(raw)
        if e is None:
            return None
        typed_kept.append(e)
    partial = apply_edits(initial, typed_kept)
    if partial.errors:
        return None

    typed_deleted = []
    for raw in deleted:
        e, err = parse_edit(raw)
        if e is None:
            return None
        typed_deleted.append(e)
    final = apply_edits(partial.layout, typed_deleted)
    if final.errors or final.applied != len(typed_deleted):
        return None
    sim = simulate(final.layout)
    if sim.green_science_rate <= 0:
        return None

    return GeneratedPair(
        seed=seed,
        prompt=build_user_message(partial.layout),
        completion=json.dumps(deleted, separators=(",", ":")),
        sim_gs_rate=sim.green_science_rate,
        n_assemblers=pair.n_assemblers,
        tiers=pair.tiers,
        mode="partial",
    )


def build_template_pairs(seed: int, *, variants: int = 4) -> list[dict]:
    target_full = variants // 2
    target_partial = variants - target_full
    full_rows: list[GeneratedPair] = []
    partial_rows: list[GeneratedPair] = []
    attempts = 0
    variant = 0
    while (len(full_rows) < target_full or len(partial_rows) < target_partial) and attempts < variants * 50:
        attempts += 1
        full = _build_full_pair(seed, variant)
        variant += 1
        if full is None:
            continue
        if len(full_rows) < target_full:
            full_rows.append(full)
        if len(partial_rows) < target_partial:
            partial = _partial_from_full(full, seed, variant)
            variant += 1
            if partial is not None:
                partial_rows.append(partial)
    out = full_rows[:target_full] + partial_rows[:target_partial]
    return [
        {
            "seed": p.seed,
            "prompt": p.prompt,
            "completion": p.completion,
            "sim_gs_rate": p.sim_gs_rate,
            "n_assemblers": p.n_assemblers,
            "tiers": list(p.tiers),
            "mode": p.mode,
        }
        for p in out
    ]


def build_template_dataset(seeds: Iterable[int], *, variants_per_seed: int = 4) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        rows.extend(build_template_pairs(seed, variants=variants_per_seed))
    return rows
