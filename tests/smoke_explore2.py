"""Offline unit tests for Explore2 (learned counter mask + action ranking).

No network / no game sim — we drive the agent with hand-built FrameData
sequences where we KNOW the ground truth (which pixels are a counter, which
action is a no-op) and assert the learned behaviour. Reliable, unlike a full
game simulator.

Run:  .venv\\Scripts\\python -m tests.smoke_explore2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import FrameData, GameAction, GameState  # noqa: E402

from agents import AVAILABLE_AGENTS  # noqa: E402
from agents.templates.explore2_agent import Explore2  # noqa: E402


def _frame(grid, actions=None, state=GameState.NOT_FINISHED, levels=0):
    return FrameData(
        game_id="t", frame=[grid], state=state, levels_completed=levels,
        win_levels=1, guid="g", full_reset=False,
        available_actions=actions or [GameAction.ACTION1, GameAction.ACTION2],
    )


def _agent():
    return Explore2(card_id="", game_id="t", agent_name="t", ROOT_URL="",
                    record=False, arc_env=None)


def _blank(n=16):
    return [[0] * n for _ in range(n)]


def _tests():
    out = []

    # 1. registry
    out.append(("registry_has_explore2", AVAILABLE_AGENTS.get("explore2") is Explore2))

    # 2. counter mask is LEARNED: an interior cell that flips every step gets
    #    masked; two frames differing only there hash the SAME afterwards.
    ag = _agent()
    counter = (8, 8)            # interior cell that toggles every transition
    real = (5, 5)              # a real game cell that changes once, then static
    for i in range(ag.MASK_WARMUP + 2):
        g = _blank()
        g[counter[0]][counter[1]] = (i % 3) + 1   # always changing
        if i >= 6:
            g[real[0]][real[1]] = 7               # appears midway, then static
        ag.choose_action([], _frame(g))
    out.append(("mask_frozen", ag._mask_cells is not None))
    out.append(("counter_in_mask", counter in (ag._mask_cells or set())))
    out.append(("real_cell_not_masked", real not in (ag._mask_cells or set())))
    # two frames differing only in the counter cell must hash equal now
    a = _blank(); a[counter[0]][counter[1]] = 1
    b = _blank(); b[counter[0]][counter[1]] = 2
    out.append(("counter_collapsed", ag._state_key([a]) == ag._state_key([b])))
    # but a difference in a real interior cell must still distinguish
    c = _blank(); c[real[0]][real[1]] = 4
    out.append(("real_distinguished", ag._state_key([a]) != ag._state_key([c])))

    # 3. guard: if almost everything changes every step, DON'T mask (else we'd
    #    erase real state). Mask should be empty (border-only behaviour).
    ag2 = _agent()
    for i in range(ag2.MASK_WARMUP + 2):
        g = [[(i + r + col) % 5 for col in range(16)] for r in range(16)]
        ag2.choose_action([], _frame(g))
    out.append(("guard_no_overmask", ag2._mask_cells == frozenset()))

    # 4. effectiveness tracking: ACTION1 always changes state, ACTION2 never.
    #    A transition is attributed to _prev_emitted (the action that produced the
    #    NEW frame), so we set _prev_emitted to the action that caused THIS frame,
    #    and only bump the frame when that action is the "effective" one.
    ag3 = _agent()
    g0 = _blank(); g0[4][4] = 1
    ag3._observe_transition([g0])       # seed prev_interior (no attribution)
    val = 1
    for i in range(8):
        emitted = "ACTION1" if i % 2 == 0 else "ACTION2"
        if emitted == "ACTION1":
            val += 1                    # ACTION1 -> this frame differs from prev
        g = _blank(); g[4][4] = val     # ACTION2 -> identical frame (no change)
        ag3._prev_emitted = emitted
        ag3._observe_transition([g])
    e1 = ag3._effect.get("ACTION1", [0, 0])
    e2 = ag3._effect.get("ACTION2", [0, 0])
    p1 = e1[0] / e1[1] if e1[1] else 0
    p2 = e2[0] / e2[1] if e2[1] else 0
    out.append(("effective_action_ranked_higher", p1 > p2))

    # 5. _build_plans puts the effective action ahead of the no-op one
    #    (fresh agent: don't inherit ag3's accumulated state).
    ag5 = _agent()
    ag5._effect = {"ACTION1": [9, 9], "ACTION2": [0, 9]}
    plans = ag5._build_plans(_frame(_blank(),
                                    actions=[GameAction.ACTION1, GameAction.ACTION2]))
    simple = [p for p in plans if not isinstance(p, tuple)]
    names = [getattr(p, "name", str(p)) for p in simple]
    out.append(("plan_order_effective_first",
                bool(names) and names[0] == "ACTION1"))

    # 6. with no learning yet, behaviour == parent (RESET on NOT_PLAYED)
    ag4 = _agent()
    f = _frame(_blank(), state=GameState.NOT_PLAYED)
    out.append(("resets_when_not_played", ag4.choose_action([], f) is GameAction.RESET))

    return out


def main():
    res = _tests()
    ok = True
    for name, passed in res:
        print(("PASS" if passed else "FAIL"), name)
        ok = ok and bool(passed)
    print("---", "ALL PASS" if ok else "SOME FAILED",
          "(%d/%d)" % (sum(1 for _, p in res if p), len(res)))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
