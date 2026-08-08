"""A* pathfinding for belt tile routing.

Finds the shortest chain of adjacent tiles from `start` to `end` avoiding a set of
blocked tiles. Returns the tile chain as a list of (x, y, direction) triples where
direction is the flow direction on that tile — suitable for building a Belt.

Not used by the simulator directly (belts in a Layout are already-placed tile lists),
but used by random_layouts.py and by tests that need to construct valid belt runs.
"""
from __future__ import annotations

import heapq
from collections.abc import Iterable

from .layout import DIR_DELTA


def _neighbors(x: int, y: int, w: int, h: int) -> Iterable[tuple[int, int, str]]:
    for d, (dx, dy) in DIR_DELTA.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield nx, ny, d


def route_belt(
    start: tuple[int, int],
    end: tuple[int, int],
    grid_size: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int, str]] | None:
    """A* from start to end on a 4-connected grid.

    Returns a list of (x, y, direction) tiles. The last tile's direction points
    toward `end`+step (so belts naturally flow into whatever consumes at `end`).
    If start == end the return is a single tile with direction 'east' by default.
    Returns None if unreachable.
    """
    w, h = grid_size
    if start == end:
        return [(start[0], start[1], "east")]
    if start in blocked or end in blocked:
        return None

    def heuristic(x: int, y: int) -> int:
        return abs(x - end[0]) + abs(y - end[1])

    # State: (f, g, (x, y), came_from_step)
    #   came_from_step maps node -> (prev_node, direction_stepping_into_this_node)
    came_from: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
    g_score: dict[tuple[int, int], int] = {start: 0}
    pq: list[tuple[int, int, tuple[int, int]]] = [(heuristic(*start), 0, start)]

    while pq:
        _, g, node = heapq.heappop(pq)
        if node == end:
            # Reconstruct: each tile's direction = direction stepping OUT of it
            # toward the next tile. Walk back to build the chain in reverse.
            chain: list[tuple[int, int, str]] = []
            cur = end
            path: list[tuple[int, int]] = [end]
            while cur in came_from:
                cur, _ = came_from[cur]
                path.append(cur)
            path.reverse()  # start ... end
            for i, (x, y) in enumerate(path):
                if i + 1 < len(path):
                    nx, ny = path[i + 1]
                    for d, (dx, dy) in DIR_DELTA.items():
                        if (x + dx, y + dy) == (nx, ny):
                            chain.append((x, y, d))
                            break
                else:
                    # Last tile: inherit prior direction so it keeps flowing outward
                    chain.append((x, y, chain[-1][2]) if chain else (x, y, "east"))
            return chain

        if g > g_score.get(node, 10**9):
            continue
        for nx, ny, d in _neighbors(node[0], node[1], w, h):
            nn = (nx, ny)
            if nn in blocked and nn != end:
                continue
            tentative = g + 1
            if tentative < g_score.get(nn, 10**9):
                g_score[nn] = tentative
                came_from[nn] = (node, d)
                heapq.heappush(pq, (tentative + heuristic(nx, ny), tentative, nn))
    return None
