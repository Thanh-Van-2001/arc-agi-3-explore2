"""Graph-based exploration agent for ARC-AGI-3.

Strategy (no LLM, no GPU training - see PLAN.md):
  * Treat each distinct frame as a node in a per-level state graph.
  * Edge = (action) that transformed one frame into another.
  * From the current node, try its untried actions first. When a node is
    exhausted, BFS to the nearest node that still has untried actions and
    replay the path to get there ("frontier exploration").
  * For the complex click action (ACTION6) we do not enumerate all 64*64
    coordinates blindly: we segment the frame into same-color connected
    components and propose their centroids as click candidates, ordered by a
    crude "button-likeness" heuristic (small, compact, non-background blobs).

This is intentionally deterministic and cheap. It is the v1 baseline we will
iterate on (world-model, object abstraction) per PLAN.md.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger()


def _grid_key(frame: Optional[list]) -> tuple:
    """Hashable signature of the (possibly multi-grid) frame."""
    if not frame:
        return ()
    return tuple(tuple(tuple(row) for row in grid) for grid in frame)


def _last_grid(frame: Optional[list]) -> Optional[list]:
    if not frame:
        return None
    return frame[-1]


def _connected_components(grid: list) -> list:
    """4-connectivity same-color components. Returns list of dicts with
    color, size, bbox, and centroid (x=col, y=row)."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    seen = [[False] * w for _ in range(h)]
    comps: list = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc]:
                continue
            color = grid[sr][sc]
            q = deque([(sr, sc)])
            seen[sr][sc] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        q.append((nr, nc))
            rows = [c[0] for c in cells]
            cols = [c[1] for c in cells]
            comps.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "centroid": (sum(cols) // len(cells), sum(rows) // len(cells)),
                }
            )
    return comps


def _background_color(grid: list) -> int:
    """Most common color is assumed to be background."""
    counts: dict = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


def _click_candidates(grid: Optional[list], limit: int = 12) -> list:
    """Propose (x, y) click points, ordered by button-likeness.

    Heuristic: prefer small, compact, non-background blobs (likely buttons/items)
    over large background regions. Returns centroids in (x, y) = (col, row).
    """
    if not grid:
        return []
    bg = _background_color(grid)
    comps = _connected_components(grid)
    scored = []
    for comp in comps:
        if comp["color"] == bg:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        bbox_area = max(1, (r1 - r0 + 1) * (c1 - c0 + 1))
        fill = comp["size"] / bbox_area
        score = fill / (1.0 + comp["size"])
        scored.append((score, comp["centroid"]))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [cxy for _, cxy in scored[:limit]]


class Explore(Agent):
    """Frontier graph exploration agent."""

    # ls20 enforces an ~80-action episode budget server-side (GAME_OVER after),
    # so a high cap mainly matters for games that allow longer episodes / many
    # RESET-and-retry trajectories. Keep generous; the graph carries across resets.
    MAX_ACTIONS = 600

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._nodes: dict = {}
        self._edges: dict = {}
        self._level_seen_reset = -1
        self._replay_queue: deque = deque()
        self._last_key: Optional[tuple] = None
        self._last_plan: Optional[Any] = None

    def is_done(self, frames: list, latest_frame: FrameData) -> bool:
        done = latest_frame.state is GameState.WIN
        # Lightweight diagnostics: log state changes + periodic heartbeat so we
        # can see WIN/GAME_OVER transitions (action log lines don't show state).
        st = getattr(latest_frame.state, "name", str(latest_frame.state))
        if st != getattr(self, "_last_logged_state", None):
            logger.info(
                "STATE -> %s (action %d, levels_completed=%d, nodes=%d)",
                st, self.action_counter, latest_frame.levels_completed, len(self._nodes),
            )
            self._last_logged_state = st
        return done

    @staticmethod
    def _as_action(a: Any) -> GameAction:
        """available_actions may arrive as ints (pydantic-coerced) or enums."""
        if isinstance(a, GameAction):
            return a
        return GameAction.from_id(a)

    def _simple_actions(self, latest_frame: FrameData) -> list:
        avail = getattr(latest_frame, "available_actions", None)
        if avail:
            acts = [self._as_action(a) for a in avail]
            acts = [a for a in acts if a is not GameAction.RESET]
        else:
            acts = [a for a in GameAction if a is not GameAction.RESET]
        return acts

    def _build_plans(self, latest_frame: FrameData) -> list:
        """A plan is either a GameAction (simple) or ('CLICK', x, y) for ACTION6."""
        plans: list = []
        avail = self._simple_actions(latest_frame)
        for a in avail:
            # ACTION6 is the complex click action; handled separately below.
            # (GameAction enum members are shared/mutated by set_data, so
            #  a.is_simple() can flip True after a click — exclude explicitly.)
            if a is GameAction.ACTION6:
                continue
            if a.is_simple():
                plans.append(a)
        if any(a is GameAction.ACTION6 for a in avail):
            grid = _last_grid(latest_frame.frame)
            for (x, y) in _click_candidates(grid):
                plans.append(("CLICK", x, y))
        return plans

    def _plan_to_action(self, plan: Any) -> GameAction:
        if isinstance(plan, tuple) and plan and plan[0] == "CLICK":
            action = GameAction.ACTION6
            action.set_data({"x": plan[1], "y": plan[2]})
            action.reasoning = {"strategy": "explore-click", "x": plan[1], "y": plan[2]}
            return action
        action = plan
        action.reasoning = {"strategy": "explore-simple"}
        return action

    def _register(self, key: tuple, latest_frame: FrameData) -> None:
        if key not in self._nodes:
            self._nodes[key] = {"untried": self._build_plans(latest_frame)}
            self._edges.setdefault(key, [])

    def _bfs_to_frontier(self, start: tuple) -> Optional[list]:
        """Return plans leading from start to a node with untried plans."""
        q: deque = deque([(start, [])])
        visited = {start}
        while q:
            node, path = q.popleft()
            info = self._nodes.get(node)
            if info and info["untried"] and node != start:
                return path
            for plan, child in self._edges.get(node, []):
                if child not in visited:
                    visited.add(child)
                    q.append((child, path + [plan]))
        return None

    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        state = latest_frame.state

        if state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._replay_queue.clear()
            self._last_key = None
            return GameAction.RESET

        if latest_frame.levels_completed != self._level_seen_reset:
            self._level_seen_reset = latest_frame.levels_completed

        key = _grid_key(latest_frame.frame)
        self._register(key, latest_frame)

        if self._last_key is not None and self._last_plan is not None:
            edges = self._edges.setdefault(self._last_key, [])
            if not any(p == self._last_plan and c == key for p, c in edges):
                edges.append((self._last_plan, key))

        if self._replay_queue:
            plan = self._replay_queue.popleft()
            self._last_key, self._last_plan = key, plan
            return self._plan_to_action(plan)

        info = self._nodes[key]
        if info["untried"]:
            plan = info["untried"].pop(0)
            self._last_key, self._last_plan = key, plan
            return self._plan_to_action(plan)

        path = self._bfs_to_frontier(key)
        if path:
            self._replay_queue = deque(path)
            plan = self._replay_queue.popleft()
            self._last_key, self._last_plan = key, plan
            return self._plan_to_action(plan)

        self._replay_queue.clear()
        self._last_key, self._last_plan = None, None
        return GameAction.RESET
