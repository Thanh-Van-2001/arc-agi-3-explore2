"""Object-abstraction + grid A* planner for ARC-AGI-3 (class-A games).

Standalone, dependency-light helpers used by the PlannerAgent as a GATED
sub-policy on top of explore2. Kept free of any agent/graph state so it can be
unit-tested in isolation (tests/smoke_planner.py) — the entanglement with parent
bookkeeping is exactly what broke world-model v1.

Pipeline (per the source-grounded design, docs/object_abstraction_design.md):
  1. Detect the avatar = the connected-component whose centroid translates by a
     consistent vector under movement actions (learned online from real
     transitions, reusing explore2's _observe_transition data via the agent).
  2. Build a coarse occupancy grid from non-background components (walls/blocks).
  3. A* from the avatar cell to a goal cell, returning the move-action sequence.

This module only provides the pure functions; the agent supplies observations
and applies the returned plan, re-verifying by frame re-read (never trusting a
hardcoded goal).
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

# Movement action name -> unit (dr, dc) in grid space. The actual on-screen
# pixel delta (cell_size) is learned; here we plan in CELL units.
MOVE_NAMES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")


def detect_avatar_delta(prev_cells: Dict[int, list], now_cells: Dict[int, list]
                        ) -> Optional[Tuple[int, Tuple[int, int]]]:
    """Given color->[(r,c),...] maps before/after one action, return
    (color, (dr,dc)) of the single color whose cells all translated by one
    constant nonzero vector. None if ambiguous."""
    for col, P in prev_cells.items():
        N = now_cells.get(col)
        if not P or not N or len(P) != len(N):
            continue
        P2, N2 = sorted(P), sorted(N)
        vecs = {(N2[i][0] - P2[i][0], N2[i][1] - P2[i][1]) for i in range(len(P2))}
        if len(vecs) == 1:
            v = next(iter(vecs))
            if v != (0, 0):
                return col, v
    return None


def detect_avatar_by_diff(prev_grid: list, now_grid: list, bg: int,
                          max_blob: int = 64
                          ) -> Optional[Tuple[int, Tuple[int, int]]]:
    """Robust avatar detection by frame diff: find cells that changed between
    two frames; if the change is one small object translating (a compact set of
    cells that vanished from positions A and appeared at A+v for one vector v),
    return (color_at_new, v). Handles games where many same-colored sprites
    exist (the per-color 'all cells translate' test fails there).

    Heuristic: collect 'appeared' cells (now != prev, now != bg) and 'vanished'
    cells (prev != bg, now == bg or different). If appeared is a small blob and
    vanished is the same-size blob shifted by a single vector v, that's the
    avatar moving by v; its color = the modal non-bg color among appeared cells.
    """
    h = len(now_grid)
    w = len(now_grid[0]) if h else 0
    if h != len(prev_grid):
        return None
    appeared = []   # cells that are non-bg now but were different/bg before
    vanished = []   # cells that were non-bg before but changed now
    for r in range(h):
        pr, nr = prev_grid[r], now_grid[r]
        for c in range(w):
            if pr[c] != nr[c]:
                if nr[c] != bg:
                    appeared.append((r, c, nr[c]))
                if pr[c] != bg:
                    vanished.append((r, c))
    if not appeared or not vanished:
        return None
    if len(appeared) > max_blob or len(vanished) > max_blob:
        return None
    app_pos = sorted((r, c) for r, c, _ in appeared)
    van_pos = sorted(vanished)
    if len(app_pos) != len(van_pos):
        return None
    vecs = {(app_pos[i][0] - van_pos[i][0], app_pos[i][1] - van_pos[i][1])
            for i in range(len(app_pos))}
    if len(vecs) != 1:
        return None
    v = next(iter(vecs))
    if v == (0, 0):
        return None
    # avatar color = modal color among appeared cells
    from collections import Counter
    col = Counter(cc for _, _, cc in appeared).most_common(1)[0][0]
    return col, v


def astar(start: Tuple[int, int], goal: Tuple[int, int],
          blocked: set, h: int, w: int,
          moves: Dict[str, Tuple[int, int]],
          max_expand: int = 20000) -> Optional[List[str]]:
    """A* on a grid. start/goal = (r,c). blocked = set of impassable cells.
    moves = action_name -> (dr,dc). Returns list of action names, or None.
    4-connected, unit cost, Manhattan heuristic."""
    if start == goal:
        return []

    def hf(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    openq: list = [(hf(start), 0, start)]
    came: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
    gscore = {start: 0}
    seen = set()
    expansions = 0
    while openq:
        _, g, cur = heapq.heappop(openq)
        if cur == goal:
            # reconstruct
            path: List[str] = []
            node = cur
            while node in came:
                prev, act = came[node]
                path.append(act)
                node = prev
            path.reverse()
            return path
        if cur in seen:
            continue
        seen.add(cur)
        expansions += 1
        if expansions > max_expand:
            return None
        for name, (dr, dc) in moves.items():
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < h and 0 <= nb[1] < w):
                continue
            if nb in blocked:
                continue
            ng = g + 1
            if ng < gscore.get(nb, 1 << 30):
                gscore[nb] = ng
                came[nb] = (cur, name)
                heapq.heappush(openq, (ng + hf(nb), ng, nb))
    return None


def occupancy_from_components(comps: list, bg_color: int,
                             avatar_color: Optional[int],
                             goal_color: Optional[int]) -> set:
    """Cells occupied by non-background, non-avatar, non-goal components ->
    treated as walls/blocks for planning. comps = list of dicts from
    _connected_components (color, cells via bbox not enough -> use 'cells' if
    present, else bbox fill)."""
    blocked: set = set()
    for comp in comps:
        col = comp["color"]
        if col == bg_color or col == avatar_color or col == goal_color:
            continue
        cells = comp.get("cells")
        if cells:
            blocked.update(cells)
        else:
            r0, c0, r1, c1 = comp["bbox"]
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    blocked.add((r, c))
    return blocked
