"""Offline smoke test for the Explore agent's decision logic.

No network / no API key needed. We instantiate the agent with arc_env=None
(choose_action never touches arc_env) and drive it with hand-built FrameData
frames representing a tiny deterministic game, asserting the exploration
behaviour: RESET when not playing, try each untried action, register graph
nodes/edges, and stop on WIN.

Run:  .venv\\Scripts\\python tests\\smoke_explore.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import FrameData, GameAction, GameState  # noqa: E402

from agents.templates.explore_agent import (  # noqa: E402
    Explore,
    _click_candidates,
    _connected_components,
    _grid_key,
)


def make_frame(grid2d, state, levels=0, actions=None):
    return FrameData(
        game_id="test",
        frame=[grid2d],
        state=state,
        levels_completed=levels,
        win_levels=1,
        guid="g",
        full_reset=False,
        available_actions=actions
        or [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION6],
    )


def new_agent():
    return Explore(
        card_id="",
        game_id="test",
        agent_name="test",
        ROOT_URL="",
        record=False,
        arc_env=None,
    )


def test_components_and_clicks():
    # 4x4 grid: background 0, a 1x1 blob of color 5 at (row1,col2), 2x2 blob color 3
    grid = [
        [0, 0, 0, 0],
        [0, 0, 5, 0],
        [0, 3, 3, 0],
        [0, 3, 3, 0],
    ]
    comps = _connected_components(grid)
    colors = sorted({c["color"] for c in comps})
    assert colors == [0, 3, 5], colors
    cands = _click_candidates(grid)
    # background (0) must not be a candidate; the small blob (5) should rank first
    assert cands, "expected click candidates"
    assert cands[0] == (2, 1), f"expected small blob (x=2,y=1) first, got {cands[0]}"
    print("OK  components/clicks: colors=%s candidates=%s" % (colors, cands))


def test_reset_when_not_played():
    ag = new_agent()
    f = make_frame([[0, 0], [0, 0]], GameState.NOT_PLAYED)
    a = ag.choose_action([f], f)
    assert a is GameAction.RESET, a
    print("OK  resets on NOT_PLAYED")


def test_explores_untried_then_graph():
    ag = new_agent()
    gridA = [[0, 0], [0, 1]]
    fA = make_frame(gridA, GameState.NOT_FINISHED,
                    actions=[GameAction.ACTION1, GameAction.ACTION2])
    # First visit to A: should pop an untried simple action (ACTION1 or 2)
    a1 = ag.choose_action([fA], fA)
    assert a1 in (GameAction.ACTION1, GameAction.ACTION2), a1
    keyA = _grid_key(fA.frame)
    assert keyA in ag._nodes, "node A not registered"
    untried_after = len(ag._nodes[keyA]["untried"])
    # Stay on same frame A again -> should pop the *other* untried action
    a2 = ag.choose_action([fA], fA)
    assert a2 in (GameAction.ACTION1, GameAction.ACTION2)
    assert a2 != a1, "should not repeat the same untried action"
    assert len(ag._nodes[keyA]["untried"]) == untried_after - 1
    print("OK  explores distinct untried actions; node registered")


def test_win_is_done():
    ag = new_agent()
    fwin = make_frame([[1]], GameState.WIN, levels=1)
    assert ag.is_done([fwin], fwin) is True
    fnot = make_frame([[0]], GameState.NOT_FINISHED)
    assert ag.is_done([fnot], fnot) is False
    print("OK  is_done True on WIN, False otherwise")


def test_frontier_replay():
    """When current node is exhausted, agent should navigate (replay) toward a
    node that still has untried actions rather than getting stuck."""
    ag = new_agent()
    gridA = [[0, 0], [0, 1]]
    gridB = [[0, 0], [1, 1]]
    fA = make_frame(gridA, GameState.NOT_FINISHED, actions=[GameAction.ACTION1])
    fB = make_frame(gridB, GameState.NOT_FINISHED,
                    actions=[GameAction.ACTION1, GameAction.ACTION2])
    # Visit A (only 1 untried action). Pop it.
    ag.choose_action([fA], fA)               # decides from A, last_plan set
    # Transition to B (records edge A->B), B has untried actions
    ag.choose_action([fB], fB)               # registers B, pops one untried at B
    keyA, keyB = _grid_key(fA.frame), _grid_key(fB.frame)
    assert keyB in ag._nodes
    assert any(c == keyB for _, c in ag._edges.get(keyA, [])), "edge A->B not recorded"
    print("OK  frontier graph edges recorded (A->B)")


if __name__ == "__main__":
    tests = [
        test_components_and_clicks,
        test_reset_when_not_played,
        test_explores_untried_then_graph,
        test_win_is_done,
        test_frontier_replay,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print("FAIL", t.__name__, "->", e)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("ERROR", t.__name__, "->", repr(e))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
