"""Offline unit tests for the planner primitives (no network, no game)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.templates.planner import (  # noqa: E402
    astar, detect_avatar_delta, occupancy_from_components,
)

MOVES = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}


def _tests():
    out = []

    # 1. A* straight line
    p = astar((0, 0), (0, 3), set(), 5, 5, MOVES)
    out.append(("astar_straight", p == ["ACTION4", "ACTION4", "ACTION4"]))

    # 2. A* around a wall
    blocked = {(0, 1), (1, 1), (2, 1)}  # vertical wall, gap at (3,1)
    p = astar((1, 0), (1, 2), blocked, 5, 5, MOVES)
    out.append(("astar_around_wall", p is not None and len(p) == 6
                and _valid(p, (1, 0), (1, 2), blocked, 5, 5)))

    # 3. A* unreachable -> None
    box = {(0, 1), (1, 1), (1, 0)}  # seals (0,0) in corner
    p = astar((0, 0), (4, 4), box, 5, 5, MOVES)
    out.append(("astar_unreachable_none", p is None))

    # 4. A* already at goal -> []
    out.append(("astar_at_goal", astar((2, 2), (2, 2), set(), 5, 5, MOVES) == []))

    # 5. avatar delta detection: color 2 shifts right by 1, color 8 static
    prev = {2: [(3, 3)], 8: [(0, 0), (0, 1)]}
    now = {2: [(3, 4)], 8: [(0, 0), (0, 1)]}
    d = detect_avatar_delta(prev, now)
    out.append(("avatar_delta", d == (2, (0, 1))))

    # 6. avatar delta ambiguous (two movers) -> picks a consistent one or None;
    #    here both move differently => first consistent single-vector color wins,
    #    but if both are single-vector it returns the first found; ensure nonzero.
    prev2 = {2: [(0, 0)], 3: [(5, 5)]}
    now2 = {2: [(0, 0)], 3: [(5, 6)]}  # only 3 moved
    d2 = detect_avatar_delta(prev2, now2)
    out.append(("avatar_delta_single", d2 == (3, (0, 1))))

    # 7. occupancy excludes bg/avatar/goal
    comps = [
        {"color": 0, "bbox": (0, 0, 4, 4)},      # bg
        {"color": 8, "bbox": (1, 1, 1, 1)},      # wall
        {"color": 2, "bbox": (3, 3, 3, 3)},      # avatar
        {"color": 4, "bbox": (0, 4, 0, 4)},      # goal
    ]
    blk = occupancy_from_components(comps, 0, 2, 4)
    out.append(("occupancy_walls_only", blk == {(1, 1)}))

    return out


def _valid(plan, start, goal, blocked, h, w):
    cur = start
    for a in plan:
        dr, dc = MOVES[a]
        nb = (cur[0] + dr, cur[1] + dc)
        if not (0 <= nb[0] < h and 0 <= nb[1] < w) or nb in blocked:
            return False
        cur = nb
    return cur == goal


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
