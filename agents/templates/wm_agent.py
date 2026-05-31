"""World-model agent for ARC-AGI-3.

Subclass of the graph Explore agent. On top of blind frontier exploration it
learns a *level-invariant* dynamics model online and plans with it:

  * avatar detection   - the object that translates consistently under moves;
  * dynamics T         - per-action avatar delta + learned blocking colors;
  * planning (MPC)     - BFS in the learned model toward a known goal color, or
                         the nearest unvisited reachable cell; emit first step,
                         re-plan every turn, self-correct when reality differs;
  * reward transfer    - the color the avatar steps onto to clear a level is
                         remembered and re-targeted on later levels.

When the avatar/dynamics aren't known yet, or movement is exhausted, it falls
back to the proven Explore policy (interaction, clicks, reset-loop breaker), so
it never regresses below the explore baseline. Rationale: docs/world_model_design.md.

Tested offline by tests/smoke_wm.py + tests/sim_gridworld.py (no network).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from arcengine import GameAction

from agents.templates.explore_agent import (
    Explore,
    _background_color,
    _grid_key,
    _mask_borders,
)

_MOVE_NAMES = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}


def _manh(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _colormap(grid, bg) -> Dict[int, List[Tuple[int, int]]]:
    d: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v != bg:
                d[v].append((r, c))
    return d


class WorldModel(Explore):
    """Explore + an online executable dynamics model and a planner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_grid: Optional[List[List[int]]] = None
        self._avatar_color: Optional[int] = None
        self._deltas: Dict[str, Tuple[int, int]] = {}     # action -> (dr,dc)
        self._blocking: set = set()                        # colors that block
        self._passable: set = set()                        # colors avatar enters
        self._goal_colors: set = set()                     # clear-the-level colors
        self._visited: set = set()                         # avatar cells this level
        self._level = 0
        self._apos: Optional[Tuple[int, int]] = None
        self._intended: Optional[Tuple[str, Tuple[int, int]]] = None
        self._last_action: Optional[str] = None  # name of last action we emitted
        # telemetry (handy in benchmarks)
        self.planned_moves = 0
        self.fallback_moves = 0

    # -- grid helpers -----------------------------------------------------
    def _masked(self, latest) -> List[List[int]]:
        frame = getattr(latest, "frame", None)
        if not frame:
            return []
        # _mask_borders returns a tuple-of-tuples; we need mutable rows for
        # indexed lookups, so re-materialise as lists.
        return [list(row) for row in _mask_borders(frame[-1])]

    def _avatar_pos(self, grid) -> Optional[Tuple[int, int]]:
        if self._avatar_color is None:
            return None
        cells = [(r, c) for r, row in enumerate(grid)
                 for c, v in enumerate(row) if v == self._avatar_color]
        return min(cells) if cells else None

    def _detect_avatar(self, prev, now):
        """Find the color whose cells translate by one constant nonzero vector."""
        bg = _background_color(now)
        pb, nb = _colormap(prev, bg), _colormap(now, bg)
        for col, P in pb.items():
            N = nb.get(col, [])
            if not P or len(P) != len(N):
                continue
            P, N = sorted(P), sorted(N)
            vecs = {(N[i][0] - P[i][0], N[i][1] - P[i][1]) for i in range(len(P))}
            if len(vecs) == 1:
                v = next(iter(vecs))
                if v != (0, 0):
                    return col, v
        return None, None

    # -- learning ---------------------------------------------------------
    def _learn(self, latest):
        grid = self._masked(latest)
        lvl = latest.levels_completed or 0
        if self._prev_grid is not None and self._last_action in _MOVE_NAMES:
            if lvl > self._level:
                # the previous move cleared a level: the cell the avatar entered
                # (in the *previous* layout) carried the goal color.
                if self._intended is not None:
                    _, tgt = self._intended
                    if 0 <= tgt[0] < len(self._prev_grid) and \
                            0 <= tgt[1] < len(self._prev_grid[0]):
                        gc = self._prev_grid[tgt[0]][tgt[1]]
                        if gc != _background_color(self._prev_grid):
                            self._goal_colors.add(gc)
            else:
                self._learn_dynamics(self._prev_grid, grid, self._last_action)
        # level / reset bookkeeping
        if lvl > self._level:
            self._level = lvl
            self._visited.clear()
        if self._last_action == "RESET":
            self._visited.clear()
        # current avatar position
        self._apos = self._avatar_pos(grid)
        if self._apos is not None:
            self._visited.add(self._apos)
        self._prev_grid = grid

    def _learn_dynamics(self, prev, now, action):
        if self._avatar_color is None:
            col, vec = self._detect_avatar(prev, now)
            if col is not None:
                self._avatar_color = col
                self._deltas[action] = vec
            return
        ap = self._avatar_pos(prev)
        an = self._avatar_pos(now)
        if ap is None or an is None:
            return
        vec = (an[0] - ap[0], an[1] - ap[1])
        H, W = len(now), len(now[0])
        if vec != (0, 0):
            self._deltas[action] = vec
            self._passable.add(prev[an[0]][an[1]])
        elif action in self._deltas:
            dr, dc = self._deltas[action]
            br, bc = ap[0] + dr, ap[1] + dc
            if 0 <= br < H and 0 <= bc < W:
                self._blocking.add(prev[br][bc])

    # -- planning ---------------------------------------------------------
    def _plan(self, latest) -> Optional[str]:
        if self._avatar_color is None:
            return None
        grid = self._masked(latest)
        pos = self._avatar_pos(grid)
        if pos is None:
            return None
        avail = {self._as_action(a).name for a in (latest.available_actions or [])}
        moves = {n: d for n, d in self._deltas.items() if n in avail}
        if not moves:
            return None
        H, W = len(grid), len(grid[0])
        came: Dict[Tuple[int, int], Optional[Tuple[Tuple[int, int], str]]] = {pos: None}
        q = deque([pos])
        while q:
            cur = q.popleft()
            for n, (dr, dc) in moves.items():
                nb = (cur[0] + dr, cur[1] + dc)
                if not (3 <= nb[0] < H - 3 and 3 <= nb[1] < W - 3):
                    continue
                if grid[nb[0]][nb[1]] in self._blocking:
                    continue
                if nb in came:
                    continue
                came[nb] = (cur, n)
                q.append(nb)
        # target: known goal color first, else nearest unvisited reachable cell
        goal_cells = [p for p in came if p != pos
                      and grid[p[0]][p[1]] in self._goal_colors]
        target = None
        if goal_cells:
            target = min(goal_cells, key=lambda p: _manh(p, pos))
        else:
            unvis = [p for p in came if p != pos and p not in self._visited]
            if unvis:
                target = min(unvis, key=lambda p: _manh(p, pos))
        if target is None:
            return None
        # reconstruct path; return its first action
        path = []
        cur = target
        while came[cur] is not None:
            prv, act = came[cur]
            path.append((cur, act))
            cur = prv
        if not path:
            return None
        path.reverse()
        first_target, first_act = path[0]
        self._intended = (first_act, first_target)
        return first_act

    # -- policy -----------------------------------------------------------
    def choose_action(self, frames, latest):
        self._learn(latest)
        planned = self._plan(latest)
        if planned is not None:
            self.planned_moves += 1
            action = self._as_action(GameAction[planned])
        else:
            # Defer to the proven Explore policy. It maintains its own graph
            # bookkeeping; we only need to record what was emitted so the
            # dynamics learner can attribute the next observed transition.
            self.fallback_moves += 1
            action = super().choose_action(frames, latest)
        self._last_action = action.name
        return action
