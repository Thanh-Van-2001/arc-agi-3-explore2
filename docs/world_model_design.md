# World-Model Agent — Design (v1)

## Why graph-exploration plateaus at level 1

`explore_agent` hashes each raw frame into an opaque node and expands an
untried-action frontier. Two structural limits cap its depth:

1. **No transfer across levels.** When a level is cleared the game swaps in a
   *new layout*. Every new frame hashes to a brand-new node, so the explorer
   restarts from zero on level 2 — and the per-game HTTP budget (~15k steps,
   ~3 steps/s) is mostly gone by then. Result: 14/25 games reach level 1, only
   tu93/vc33 go deeper, and only by luck.
2. **Blind, goal-free search.** Untried-first expansion has no notion of *what
   an action does* or *what state is good*, so it cannot aim at the cell/object
   that advances a level — it can only stumble onto it.

## The lever: level-invariant dynamics + planning

The game *mechanics* (how the avatar moves, what blocks it, what interaction
does) are **constant across levels**; only the layout/goal changes. If we learn
those mechanics once as an **executable transition model** `T(state, action) →
state`, we can:

- **Plan in imagination for free.** The real bottleneck is the network, not CPU.
  One real HTTP step can be backed by thousands of *imagined* rollouts in `T`,
  so each real action is chosen by lookahead instead of guessed.
- **Transfer to new levels.** Movement deltas + blocking colors carry over, so
  level 2+ starts already knowing how to move and only has to *plan a path*, not
  re-learn physics. This is precisely what lets us go deeper.

This mirrors the SOTA "executable world model" (arxiv 2605.05138, 32.58%): a
deterministic, synthesized model you can *run*, not a neural net. No GPU, no
training infra — consistent with our no-LLM philosophy and the network-bound
setting.

## v1 components

1. **Object abstraction.** Top visual layer, border-masked (reuse
   `_mask_borders`). Background = modal color. Objects = connected components
   (reuse `_connected_components`).
2. **Avatar detection (online).** The object that *translates consistently*
   under movement actions ACTION1–4. After each real transition, diff prev/next
   grids; if one color's cells shift by a constant vector, tag it as the avatar
   and record the per-action delta.
3. **Dynamics `T` (level-invariant, learned online):**
   - `deltas[action] = (dr, dc)` — avatar translation per movement action.
   - `blocking_colors` / `passable_colors` — learned from move attempts
     (avatar tried to enter color X and didn't move → X blocks).
   - Imagined step: `next_pos = pos + delta` unless the target cell holds a
     blocking color (then stay). Interaction actions (5/6/7) are *unmodeled* in
     v1 → handled by fallback exploration, never simulated.
4. **Planning (model-predictive control).**
   - Abstract state = avatar position on the current layout.
   - Objective: A*/BFS in `T` toward (a) **goal-colored cells** if known, else
     (b) the nearest **unvisited reachable** cell (efficient exploration that
     still transfers). Emit the *first* action of the plan; re-plan each call.
   - **Reward learning:** when `levels_completed` increases, record the color
     the avatar stepped onto / the action that did it as a *goal pattern*;
     prioritize it on later levels.
5. **Fallback.** If avatar can't be identified yet, or movement exploration is
   exhausted, fall back to the proven `explore` policy (untried-first +
   interaction/click) so we never regress below the 14/25 baseline.

## Discipline

Same as the backtest rule: **prove it beats the baseline before trusting it.**
- Offline: synthetic gridworld sim (`tests/sim_gridworld.py`) verifies avatar
  learning, planning, and *cross-level transfer* with no network.
- Live: short head-to-head vs `explore` on deep games (tu93, vc33) + control
  games, comparing max level reached, before any full bee run.
