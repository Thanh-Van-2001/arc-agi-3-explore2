# Object-Abstraction + Planning Agent — Design (grounded in source RE)

Goal: beat the 15/25 frame-hashing ceiling. Reverse-engineered all 9 failing
games from their local source (D:\_kaggle_arc/environment_files). Grouping by the
*planner* each truly needs:

| Class | Planner | Games | Notes |
|-------|---------|-------|-------|
| **A** | Avatar→Goal A* (budget/deadline-aware) | bp35, g50t, sc25, sk48 | cheapest, biggest win |
| B | Multi-object sokoban + enemy rollout | wa30, su15 | su15 needs physics sim |
| C | Click-set vs hidden constraint | sb26, re86 | re86 needs pixel-canvas sim |
| D | Symbolic string search | tr87 | skip |

## Verified mechanics (from source, not guessed)
- **g50t** (TARGET #1): ACTION1-4 = move avatar ±1 cell (move() blocked by
  collision = walls). Win: avatar at (goal.x+1, goal.y+1). Lose: scrolling
  obstacle goes off-screen (deadline ~200-300 steps) OR avatar damage flag.
  Coupled "mirror" blocks move with the avatar (rloltuowth/kgvnkyaimw) — grid is
  not fully static, but first-cut A* on avatar pos still applies.

## The shared abstraction
State = typed object list from connected-components (explore2 already extracts):
`Obj(id, tag, role, cx, cy, w, h, color)`, role ∈ {AVATAR, GOAL, WALL, HAZARD,
MOVABLE, MANIPULATOR, DECOR}, plus `budget_remaining`, `movers`.
Role assignment (cheap, empirical — NO hardcoding per game):
- AVATAR = component whose centroid translates after a probe of ACTION1-4.
- WALL = cell where attempted move leaves avatar centroid unchanged.
- HAZARD = cell/event that triggers episode reset.
- GOAL = static non-wall component; try each as goal if ambiguous.

## Planner P1 — budget-aware grid A* (unlocks class A, the +2-3)
- Nodes = avatar cell. Edges = the 4 moves (learned cell_size from probe).
- Collision check against learned WALL cells.
- Goal test = avatar overlaps / reaches goal cell (g50t: goal+1 offset; others:
  adjacency). Verify by re-reading frame after the planned moves (don't trust a
  hardcoded goal cell — confirm via levels_completed increment).
- Deadline prune: reject path if len >= budget_remaining.
- Replan on surprise (hit an unknown wall / level changed).

## Build order (gated toggle on explore2 — worst case = parity 15/25)
- **Step 1 (g50t):** avatar-probe (4 actions) → detect avatar + cell_size;
  collect walls lazily; A* to each candidate goal; emit move plan; gate: only
  activate if exactly 1 avatar + ≥1 static goal + no manipulator needed, else
  fall back to blind explore. Expect +g50t, maybe sc25 spell-free levels.
- **Step 2 (sk48, more sc25):** variable cell_size, second controllable object
  via ACTION6, post-move win-verify by frame re-read.
- **Step 3 (wa30):** grab/drop (ACTION5) + held_flag + enemy rollout + greedy
  multi-goal→zone assignment.
- **Defer:** sb26 (P3 click-match), su15/re86 (need simulators), tr87 (symbolic).

Realistic ceiling with this design: **+2-3 games → 17-18/25** (g50t, sk48, sc25;
bp35 partial). NOT all 9 — re86/tr87/su15 need extra simulators beyond the
shared abstraction.

## Discipline (avoid world-model v1 trap)
- STANDALONE module (planner functions), explore2 calls it as a gated sub-policy
  — do NOT entangle with the parent's graph bookkeeping (that broke wm v1).
- Offline harness: build the avatar/A* on g50t source first; unit-test A* on a
  synthetic grid.
- Gate must prove preconditions before overriding blind explore → no regression.
- A/B every step on bee offline vs the 15/25 baseline; keep only on net gain.
- Fix `_bfs_to_frontier` O(n^2) is NOT needed here (planner replaces explore on
  these games; blind explore still capped at 15k elsewhere).
