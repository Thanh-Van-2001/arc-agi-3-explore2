"""Explore2 — graph exploration with two techniques ported from Occam.

Occam (github.com/g-baskin/occam, MIT, "Sean Donahoe") uses the same reset-replay
graph-exploration core as our Explore but reaches 17/25 public games (57.6% RHAE,
CPU-only, no LLM) where the identical core gets 14/25. Two of its upgrades are the
named reasons, both pure-Python / no-training, reimplemented here in our own
architecture (the env API differs, so this is a port, not a copy):

  1. AUTO-LEARNED COUNTER/ANIMATION MASK (Occam: detect_counter_mask). Our Explore
     hard-masks a fixed 3-cell border to kill edge step-counters. But some games
     render the counter / an animation *inside* the scene, so every frame hashes
     unique and the state graph explodes -> BFS never goes deep. Occam ANDs the
     pixels that change on every consecutive transition over ~3 actions and zeros
     them before hashing. We do the robust online version: accumulate, over a
     warmup window, the cells that change in >= MASK_CHANGE_RATIO of transitions,
     then freeze that as the mask. Guarded: never mask more than MASK_MAX_FRACTION
     of the interior (else we'd collapse real game state -> keep border-only).

  2. EFFECTIVE-ACTION ORDERING (Occam: _discover_and_prune + action_effectiveness).
     Occam probes each action once from reset and hard-drops the ones that don't
     change the masked hash. We use the softer online form: track, per simple
     action, P(it changed the masked state), and when expanding a fresh node try
     historically-effective actions first (a disabled/no-op button sinks to the
     back). Softer than a hard one-shot prune because an action that is a no-op in
     one state may matter in another; ordering keeps it available but late.

Everything else (frontier BFS replay, click candidates, reset-loop breaker) is
inherited unchanged from Explore, so `explore` stays the A/B control. Offline
unit tests in tests/smoke_explore2.py drive the mask learner and action ranker
with hand-built frame sequences of known ground truth (no network, no game sim).
"""
from __future__ import annotations

import os
from typing import Any, Optional

from arcengine import FrameData, GameAction

from agents.templates.explore_agent import Explore, _mask_borders, STATUS_BAR_BORDER


class Explore2(Explore):
    """Explore + learned counter mask + effective-action ordering."""

    # Observe this many transitions before freezing the learned counter mask.
    MASK_WARMUP = 12
    # A cell is "counter/animation" if it changed in >= this fraction of the
    # observed transitions during warmup. Env-overridable for sweeps.
    MASK_CHANGE_RATIO = float(os.getenv("ARC_E2_MASK_RATIO", "0.8"))
    # Safety: never mask more than this fraction of interior cells. Env-overridable.
    # Diagnosis (2026-06-01): 8/9 failing games freeze an EMPTY mask because the
    # changing region exceeds this cap (>20% of interior animates every step), so
    # the state graph explodes and BFS never reaches a goal. Raising the cap (and/
    # or the change ratio) is the lever under test — but it risks merging real
    # state on the deep winners (tu93=5, vc33=2), so any change must be swept over
    # all 25 games, not just the failing ones.
    MASK_MAX_FRACTION = float(os.getenv("ARC_E2_MASK_MAXFRAC", "0.20"))

    # Ablation toggles (env-overridable) so we can A/B which component helps on a
    # given game. Both default ON. ARC_E2_MASK=0 disables the counter mask;
    # ARC_E2_REORDER=0 disables effective-action ordering. Used to diagnose the
    # cn04 regression (explore2 lost a level explore had) without forking code.
    USE_MASK = os.getenv("ARC_E2_MASK", "1") != "0"
    USE_REORDER = os.getenv("ARC_E2_REORDER", "1") != "0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mask_cells: Optional[frozenset] = None  # frozen once learned
        self._change_counts: dict = {}                # (r,c) -> times changed
        self._transitions = 0
        self._prev_interior: Optional[tuple] = None   # border-masked prev grid
        self._prev_emitted: Optional[str] = None      # action name we last sent
        # effectiveness: action name -> [n_changed, n_total]
        self._effect: dict = {}

    # -- learned mask -----------------------------------------------------
    def _state_key(self, frame: Optional[list]) -> tuple:
        if not frame:
            return ()
        out = []
        for grid in frame:
            m = _mask_borders(grid)  # tuple-of-tuples, border already zeroed
            if self._mask_cells and self.USE_MASK:
                rows = [list(r) for r in m]
                h = len(rows)
                w = len(rows[0]) if h else 0
                for (r, c) in self._mask_cells:
                    if 0 <= r < h and 0 <= c < w:
                        rows[r][c] = 0
                m = tuple(tuple(r) for r in rows)
            out.append(m)
        return tuple(out)

    def _observe_transition(self, frame: Optional[list]) -> None:
        """Diff the new interior grid against the previous one; accumulate which
        cells change and how effective the last emitted action was. Freeze the
        counter mask once enough transitions are seen."""
        if not frame:
            return
        interior = _mask_borders(frame[-1])
        prev = self._prev_interior
        self._prev_interior = interior
        if prev is None or len(prev) != len(interior):
            return
        h = len(interior)
        w = len(interior[0]) if h else 0
        changed_any = False
        learning = self._mask_cells is None  # only scan cells while unfrozen
        for r in range(h):
            pr, cr = prev[r], interior[r]
            if pr == cr:
                continue
            changed_any = True
            if learning:
                for c in range(w):
                    if pr[c] != cr[c]:
                        k = (r, c)
                        self._change_counts[k] = self._change_counts.get(k, 0) + 1
        self._transitions += 1
        # attribute effectiveness to the action that produced this transition
        if self._prev_emitted is not None:
            e = self._effect.setdefault(self._prev_emitted, [0, 0])
            e[1] += 1
            if changed_any:
                e[0] += 1
        if learning and self._transitions >= self.MASK_WARMUP:
            self._freeze_mask(h, w)

    def _freeze_mask(self, h: int, w: int) -> None:
        b = STATUS_BAR_BORDER
        interior_area = max(1, (h - 2 * b) * (w - 2 * b))
        threshold = self.MASK_CHANGE_RATIO * self._transitions
        cells = {k for k, n in self._change_counts.items() if n >= threshold}
        # guard: a too-large mask would erase real game state -> keep border-only
        if len(cells) > self.MASK_MAX_FRACTION * interior_area:
            cells = set()
        self._mask_cells = frozenset(cells)
        self._change_counts = {}  # free memory

    # -- effective-action ordering ---------------------------------------
    def _build_plans(self, latest_frame: FrameData) -> list:
        plans = super()._build_plans(latest_frame)
        if not self._effect or not self.USE_REORDER:
            return plans

        def rank(plan: Any) -> float:
            name = getattr(plan, "name", str(plan))
            n_chg, n_tot = self._effect.get(name, (0, 0))
            # unknown actions -> 0.5 (worth exploring); known -> measured P(change)
            return (n_chg / n_tot) if n_tot else 0.5

        simple = [p for p in plans if not isinstance(p, tuple)]
        clicks = [p for p in plans if isinstance(p, tuple)]
        simple.sort(key=rank, reverse=True)  # most effective first
        return simple + clicks               # clicks keep inherited likeness order

    # -- hook into the loop ----------------------------------------------
    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        # Learn from the transition the previous action produced, THEN decide so
        # this step's state key already uses the up-to-date mask.
        self._observe_transition(latest_frame.frame)
        action = super().choose_action(frames, latest_frame)
        self._prev_emitted = getattr(action, "name", str(action))
        return action
