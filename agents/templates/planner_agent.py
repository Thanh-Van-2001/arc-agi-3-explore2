"""PlannerAgent — object-abstraction + grid A* as a GATED sub-policy on explore2.

Strategy for class-A games (avatar must reach a goal cell): learn the avatar by
probing the 4 move actions, build an occupancy grid (non-background = walls),
then A* the avatar to candidate goal cells and execute the move plan, verifying
success by a levels_completed increment. If the planner cannot apply (no avatar
found, no plan, or a plan failed to make progress), it FALLS BACK to the proven
explore2 policy — so the worst case is parity with explore2's 15/25.

Design + source-grounded rationale: docs/object_abstraction_design.md.
Pure primitives + unit tests: agents/templates/planner.py, tests/smoke_planner.py.

Toggle: --agent=planner  (or ARC_PLANNER=1 with explore2). Default explore2 is
untouched; this is an additive new agent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from arcengine import FrameData, GameAction

from agents.templates.explore2_agent import Explore2
from agents.templates.explore_agent import (
    _background_color, _connected_components, _last_grid, _mask_borders,
)
from agents.templates.planner import MOVE_NAMES, astar, detect_avatar_delta

# Candidate win offsets (avatar cell relative to goal cell). g50t wins at
# goal+1; most class-A games win on exact overlap or 4-adjacency. Try in order.
_GOAL_OFFSETS = [(0, 0), (1, 1), (-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0)]


class PlannerAgent(Explore2):
    """explore2 + a gated avatar→goal A* planner."""

    PROBE_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
    MAX_PLAN_LEN = 400        # don't emit absurd plans
    REPLAN_AFTER_FAIL = 3     # give up planning after this many useless plans

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._probe_i = 0
        self._probe_prev: Optional[Dict[int, list]] = None
        self._avatar_color: Optional[int] = None
        self._move_to_delta: Dict[str, Tuple[int, int]] = {}  # action -> (dr,dc) in CELLS
        self._plan_queue: List[str] = []
        self._tried_targets: set = set()
        self._failed_plans = 0
        self._planner_dead = False     # gate: fell back permanently this episode
        self._last_levels = 0

    # -- helpers ----------------------------------------------------------
    def _grid(self, latest) -> Optional[list]:
        g = _last_grid(latest.frame)
        return [list(r) for r in _mask_borders(g)] if g else None

    def _color_cells(self, grid) -> Dict[int, list]:
        bg = _background_color(grid)
        d: Dict[int, list] = {}
        for r, row in enumerate(grid):
            for c, v in enumerate(row):
                if v != bg:
                    d.setdefault(v, []).append((r, c))
        return d

    def _avatar_cell(self, grid) -> Optional[Tuple[int, int]]:
        if self._avatar_color is None:
            return None
        cells = [(r, c) for r, row in enumerate(grid)
                 for c, v in enumerate(row) if v == self._avatar_color]
        if not cells:
            return None
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        return (sum(rs) // len(rs), sum(cs) // len(cs))

    def _build_plan(self, grid) -> Optional[List[str]]:
        """A* from avatar to an untried candidate goal cell (component centroid
        + offset). Returns a fresh move-name plan, or None."""
        if len(self._move_to_delta) < 2:
            return None
        h, w = len(grid), len(grid[0])
        # only support unit-cell moves we actually learned, and normalize to ±1
        moves = {}
        for name, (dr, dc) in self._move_to_delta.items():
            sr = (dr > 0) - (dr < 0)
            sc = (dc > 0) - (dc < 0)
            if (sr, sc) != (0, 0):
                moves[name] = (sr, sc)
        if len(moves) < 2:
            return None
        avatar = self._avatar_cell(grid)
        if avatar is None:
            return None
        bg = _background_color(grid)
        blocked = {(r, c) for r, row in enumerate(grid)
                   for c, v in enumerate(row)
                   if v != bg and v != self._avatar_color}
        comps = [c for c in _connected_components(grid)
                 if c["color"] not in (bg, self._avatar_color)]
        # candidate goal cells = each non-avatar component centroid + offsets
        for comp in comps:
            cx, cy = comp["centroid"]      # (col, row)
            base = (cy, cx)                # (row, col)
            for (orow, ocol) in _GOAL_OFFSETS:
                target = (base[0] + orow, base[1] + ocol)
                if not (0 <= target[0] < h and 0 <= target[1] < w):
                    continue
                key = (target, self._avatar_color)
                if key in self._tried_targets:
                    continue
                # allow stepping onto the target even if "blocked" (goal may be a
                # sprite); remove target from blocked for the search
                blk = blocked - {target}
                plan = astar(avatar, target, blk, h, w, moves,
                             max_expand=8000)
                if plan and 0 < len(plan) <= self.MAX_PLAN_LEN:
                    self._tried_targets.add(key)
                    return plan
        return None

    # -- main policy ------------------------------------------------------
    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        # keep explore2's learning (mask + effectiveness) alive for fallback
        self._observe_transition(latest_frame.frame)

        lv = latest_frame.levels_completed or 0
        if lv > self._last_levels:
            # progress! reset planner search for the new layout
            self._last_levels = lv
            self._plan_queue.clear()
            self._tried_targets.clear()
            self._failed_plans = 0
            self._planner_dead = False
            self._avatar_color = None
            self._probe_i = 0
            self._probe_prev = None
            self._move_to_delta.clear()

        grid = self._grid(latest_frame)

        if not self._planner_dead and grid is not None:
            act = self._planner_step(grid, frames, latest_frame)
            if act is not None:
                self._prev_emitted = act.name
                return act

        # ---- fallback: proven explore2 ----
        action = super().choose_action(frames, latest_frame)
        self._prev_emitted = getattr(action, "name", str(action))
        return action

    def _planner_step(self, grid, frames, latest_frame) -> Optional[GameAction]:
        avail = {self._as_action(a).name for a in (latest_frame.available_actions or [])}
        move_avail = [m for m in self.PROBE_ACTIONS if m in avail]
        if len(move_avail) < 2:
            self._planner_dead = True   # not a movement game -> explore2
            return None

        # Phase 1: probe to learn the avatar + per-move cell delta.
        if self._avatar_color is None or len(self._move_to_delta) < len(move_avail):
            cells = self._color_cells(grid)
            if self._probe_prev is not None and self._probe_i > 0:
                last_move = move_avail[(self._probe_i - 1) % len(move_avail)]
                d = detect_avatar_delta(self._probe_prev, cells)
                if d is not None:
                    col, vec = d
                    self._avatar_color = col
                    self._move_to_delta[last_move] = vec
            if self._probe_i < len(move_avail) * 2:
                nxt = move_avail[self._probe_i % len(move_avail)]
                self._probe_prev = cells
                self._probe_i += 1
                return GameAction[nxt]
            # probing budget done; if no avatar, give up planning
            if self._avatar_color is None:
                self._planner_dead = True
                return None

        # Phase 2: execute queued plan
        if self._plan_queue:
            nxt = self._plan_queue.pop(0)
            return GameAction[nxt]

        # Phase 3: build a new plan toward an untried goal candidate
        plan = self._build_plan(grid)
        if plan:
            self._plan_queue = plan[1:]
            return GameAction[plan[0]]

        # no plan available
        self._failed_plans += 1
        if self._failed_plans >= self.REPLAN_AFTER_FAIL:
            self._planner_dead = True   # exhausted goal candidates -> explore2
        return None
