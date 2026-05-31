"""Offline gridworld simulator that mimics the arc_env interface
(env.step / env.observation_space / env.reset) so the world-model agent can be
driven and tested with NO network.

Mechanics the agent must *learn* (it is told none of this):
  - one avatar cell (color AVATAR) moves under movement actions 1..4;
  - the action->direction mapping is fixed but hidden;
  - WALL cells block movement; out-of-interior blocks too;
  - stepping the avatar onto the GOAL cell completes the level and loads the
    next layout (used to verify cross-level dynamics transfer);
  - actions 5/6/7 are no-ops here (exercise the agent's fallback path).

All content sits inside a 4-cell margin so the agent's 3-cell border mask never
erases the avatar/walls.
"""
from __future__ import annotations

from typing import List

from arcengine import FrameData, GameState

BG, AVATAR, WALL, GOAL = 0, 2, 8, 4
# Hidden, fixed mapping action-id -> (dr, dc). The agent never sees this.
_MOVE = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def _blank(h: int, w: int) -> List[List[int]]:
    return [[BG] * w for _ in range(h)]


def _layout(h, w, avatar, goal, walls):
    g = _blank(h, w)
    for (r, c) in walls:
        g[r][c] = WALL
    gr, gc = goal
    g[gr][gc] = GOAL
    ar, ac = avatar
    g[ar][ac] = AVATAR
    return g, (ar, ac), (gr, gc)


# A few distinct layouts (same physics, different geometry) to test transfer.
def _default_levels(h=18, w=18):
    levels = []
    # L0: straight shot right, one wall to route around
    levels.append(dict(avatar=(6, 5), goal=(6, 12),
                       walls=[(6, 9), (5, 9), (7, 9), (5, 12), (7, 12)]))
    # L1: avatar elsewhere, goal up-left, a different wall
    levels.append(dict(avatar=(13, 13), goal=(5, 6),
                       walls=[(9, 9), (9, 10), (9, 8), (10, 9)]))
    # L2: tight corridor
    levels.append(dict(avatar=(4, 4), goal=(13, 13),
                       walls=[(r, 8) for r in range(4, 13)] + [(8, c) for c in range(9, 14)]))
    return levels


class GridSim:
    """Minimal env with the subset of the arc_env API the agent base uses."""

    def __init__(self, h=18, w=18, levels=None, max_steps=10_000):
        self.h, self.w = h, w
        self._defs = levels or _default_levels(h, w)
        self.max_steps = max_steps
        self.reset()

    # -- arc_env-like surface --------------------------------------------
    def reset(self):
        self._li = 0
        self.levels_completed = 0
        self.steps = 0
        self._load(self._li)
        return self.observation_space

    def _load(self, i):
        d = self._defs[i % len(self._defs)]
        self.grid, self.apos, self.gpos = _layout(
            self.h, self.w, d["avatar"], d["goal"], d["walls"])

    def _set(self, pos, color):
        self.grid[pos[0]][pos[1]] = color

    def step(self, aid):
        self.steps += 1
        done = False
        if aid in _MOVE and self.levels_completed < len(self._defs):
            dr, dc = _MOVE[aid]
            r, c = self.apos
            nr, nc = r + dr, c + dc
            blocked = (
                nr < 3 or nr >= self.h - 3 or nc < 3 or nc >= self.w - 3
                or self.grid[nr][nc] == WALL
            )
            if not blocked:
                onto_goal = (nr, nc) == self.gpos
                self._set(self.apos, BG)
                self.apos = (nr, nc)
                self._set(self.apos, AVATAR)
                if onto_goal:
                    self.levels_completed += 1
                    if self.levels_completed < len(self._defs):
                        self._load(self.levels_completed)
        # actions 5/6/7: no-op
        if self.levels_completed >= len(self._defs):
            done = True
        return self.observation_space, 0, done, {}

    @property
    def observation_space(self) -> FrameData:
        won = self.levels_completed >= len(self._defs)
        return FrameData(
            frame=[[row[:] for row in self.grid]],
            available_actions=[1, 2, 3, 4, 5],
            state=GameState.WIN if won else GameState.NOT_FINISHED,
            score=0,
            win_levels=len(self._defs),
            levels_completed=self.levels_completed,
            full_reset=False,
            guid="sim",
            game_id="sim-0000",
        )
