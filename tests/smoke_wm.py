"""Offline smoke tests for the WorldModel agent (no network).

Run: ./.venv/bin/python -m tests.smoke_wm
"""
from __future__ import annotations

import sys

from arcengine import FrameData, GameState

from agents import AVAILABLE_AGENTS
from agents.templates.wm_agent import WorldModel
from tests.sim_gridworld import GOAL, GridSim


def _frame(grid, actions=(1, 2, 3, 4, 5), state=GameState.NOT_FINISHED, levels=0):
    return FrameData(
        frame=[grid], available_actions=list(actions), state=state, score=0,
        win_levels=1, levels_completed=levels, full_reset=False,
        guid="test", game_id="test-0000",
    )


def _make(sim=None):
    return WorldModel(card_id="c", game_id="g", agent_name="t",
                      ROOT_URL="https://three.arcprize.org",
                      record=False, arc_env=sim)


def _run(ag, sim, max_steps):
    steps = 0
    while steps < max_steps:
        latest = ag._convert_raw_frame_data(sim.observation_space)
        if latest.state is GameState.WIN:
            break
        a = ag.choose_action(ag.frames, latest)
        fr = ag.take_action(a)
        if fr:
            ag.append_frame(fr)
        steps += 1
    return sim.levels_completed, steps


def _tests():
    out = []

    # 1. registry exposes wm
    out.append(("registry_has_wm", AVAILABLE_AGENTS.get("wm") is WorldModel))

    # 2. avatar + at least one delta learned after a short warmup
    sim = GridSim()
    ag = _make(sim)
    _run(ag, sim, 60)
    out.append(("avatar_detected", ag._avatar_color is not None))
    out.append(("deltas_learned", len(ag._deltas) >= 2))

    # 3. solves level 1 within a tight budget
    sim = GridSim()
    ag = _make(sim)
    lvl, steps = _run(ag, sim, 400)
    out.append(("clears_level1", lvl >= 1))

    # 4. cross-level transfer: reaches level 2, planning dominates, deltas stable
    sim = GridSim()
    ag = _make(sim)
    lvl, steps = _run(ag, sim, 1500)
    out.append(("clears_level2_transfer", lvl >= 2))
    out.append(("planning_dominates", ag.planned_moves > ag.fallback_moves))
    out.append(("all_deltas_stable", len(ag._deltas) == 4))

    # 5. goal color learned after clearing a level
    out.append(("goal_color_learned", GOAL in ag._goal_colors))

    # 6. fallback never crashes when avatar is undetectable
    ag = _make(None)
    g = [[0] * 18 for _ in range(18)]
    g[8][8] = 5  # a lone static cell, no movement possible offline
    ok = True
    try:
        for _ in range(3):
            a = ag.choose_action(ag.frames, _frame(g, actions=(5,)))
            ok = ok and hasattr(a, "name")
    except Exception as e:  # noqa: BLE001
        ok = False
        print("   fallback raised:", repr(e))
    out.append(("fallback_no_crash", ok))

    return out


def main():
    res = _tests()
    ok = True
    for name, passed in res:
        print(("PASS" if passed else "FAIL"), name)
        ok = ok and passed
    print("---", "ALL PASS" if ok else "SOME FAILED",
          "(%d/%d)" % (sum(1 for _, p in res if p), len(res)))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
