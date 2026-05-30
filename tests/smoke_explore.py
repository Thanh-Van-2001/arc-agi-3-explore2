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
    # 4x4 grid: background 0, a 1x1 blob of color 5, a 2x2 blob of color 3
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
    # background (0) must not be a candidate; both blobs should appear
    assert cands, "expected click candidates"
    assert (2, 1) in cands, f"small blob (x=2,y=1) missing from {cands}"
    print("OK  components/clicks: colors=%s candidates=%s" % (colors, cands))


def test_priority_tiers():
    # Small rare-color button should outrank a large common-color block.
    grid = [[0] * 32 for _ in range(32)]
    for r in range(5, 8):
        for c in range(5, 8):
            grid[r][c] = 5  # small salient button (rare color)
    for r in range(15, 27):
        for c in range(15, 27):
            grid[r][c] = 1  # large common block
    cands = _click_candidates(grid)
    assert cands[0] == (6, 6), f"expected button (6,6) first, got {cands[0]}"
    print("OK  priority tiers: button ranks first (%s)" % (cands[0],))


def test_border_mask():
    # Frames differing ONLY in the border must hash the same; interior differs.
    import copy

    g1 = [[0] * 8 for _ in range(8)]
    g1[4][4] = 5
    g2 = copy.deepcopy(g1)
    g2[0][0] = 9
    g2[7][7] = 9  # border-only change
    g3 = copy.deepcopy(g1)
    g3[4][3] = 7  # interior change (cols 3-4 are the unmasked interior of an 8-wide grid, border=3)
    assert _grid_key([g1]) == _grid_key([g2]), "border change must not change key"
    assert _grid_key([g1]) != _grid_key([g3]), "interior change must change key"
    print("OK  border mask: edge ignored, interior distinguished")


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
    a1 = ag.choose_action([fA], fA)
    assert a1 in (GameAction.ACTION1, GameAction.ACTION2), a1
    keyA = _grid_key(fA.frame)
    assert keyA in ag._nodes, "node A not registered"
    untried_after = len(ag._nodes[keyA]["untried"])
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
    ag = new_agent()
    gridA = [[0, 0], [0, 1]]
    gridB = [[0, 0], [1, 1]]
    fA = make_frame(gridA, GameState.NOT_FINISHED, actions=[GameAction.ACTION1])
    fB = make_frame(gridB, GameState.NOT_FINISHED,
                    actions=[GameAction.ACTION1, GameAction.ACTION2])
    ag.choose_action([fA], fA)
    ag.choose_action([fB], fB)
    keyA = _grid_key(fA.frame)
    keyB = _grid_key(fB.frame)
    assert keyB in ag._nodes
    assert any(c == keyB for _, c in ag._edges.get(keyA, [])), "edge A->B not recorded"
    print("OK  frontier graph edges recorded (A->B)")


if __name__ == "__main__":
    tests = [
        test_components_and_clicks,
        test_priority_tiers,
        test_border_mask,
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
