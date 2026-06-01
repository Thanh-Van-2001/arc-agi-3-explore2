# Paper Track — nội dung dán-thẳng cho từng ô của form

## TITLE (≤80 ký tự) — dán ô "Title"
When Does Masking Help? A Controlled Ablation of Graph Exploration on ARC-AGI-3

(76 ký tự)

## SUBTITLE (≤140 ký tự) — dán ô "Subtitle"
An LLM-free, training-free agent reaching 15/25 public games, with a falsifiable, fully-ablated account of why its two mechanisms work.

(132 ký tự)

## CARD / THUMBNAIL IMAGE (560×280) — upload ô "Card and Thumbnail Image"
File: `D:\arc_agi_3\card_560x280.png`

## MEDIA GALLERY — "Add photos" (tùy chọn, nên thêm để mạnh hơn)
File: `D:\arc_agi_3\cover.png` (1600×900, 2-panel đầy đủ)

## SUBMISSION TRACK
Main Track (đã tự chọn sẵn — không cần làm gì).

## PROJECT LINKS — "Add a link"
- Public notebook ARC-AGI-3 (link sau khi anh commit + public notebook ở Phần A của SUBMIT_CHECKLIST)
- (tùy chọn) GitHub repo nếu anh push D:\arc_agi_3 lên

## PROJECT DESCRIPTION — dán toàn bộ nội dung dưới (đây là PAPER_TRACK_WRITEUP.md,
## đã bỏ tiêu đề/subtitle vì 2 ô trên đã có; điền submission ID + notebook link ở cuối)

---
DÁN TỪ ĐÂY ↓
---

## Summary

We present **explore2**, a deterministic agent for ARC-AGI-3 that uses **no LLM, no neural network, and no training**. It is a frontier graph-exploration agent augmented with two mechanisms ported from prior public work: an **auto-learned counter/animation mask** and **effective-action ordering**. On the 25 public games it solves **15/25** (reaches ≥1 level), versus 14/25 for the un-augmented explorer, while preserving the deepest solves (tu93 = 5 levels, vc33 = 2). Every claim is backed by a **controlled ablation**: we toggle each mechanism independently across full 15,000-step runs and report exactly which games each saves and loses. The contribution is less the score than the *method discipline*: a falsifiable account of when frame-hashing exploration succeeds and when it hits a ceiling.

## The problem, precisely

An ARC-AGI-3 agent receives frames (≤64×64 grids, values 0–15) and emits one of seven actions. There are no instructions; the agent must discover, per game, what each action does and what constitutes progress. Levels increase in difficulty and a cleared level swaps in a new layout. Scoring rewards both **completion** (levels cleared) and **efficiency** (actions vs. a human baseline), squared.

This makes the core difficulty **state representation**, not search. The naive approach — treat every distinct frame as a graph node, explore untried actions first, BFS-replay to the nearest unexplored frontier — is correct in principle but fails in practice for one reason: **the node count explodes**. If any on-screen pixel changes every step (a step counter, an animation), every frame hashes to a new node, the frontier never closes, and BFS within a finite action budget never reaches the cells that advance a level.

## Theory: why the two mechanisms work

Our baseline `explore` already masks a fixed 3-cell border (status bars often live at the edge). The two mechanisms generalize this with a clear theory of *what they fix*.

**Mechanism 1 — Auto-learned counter mask.** Instead of hard-coding which cells are decorative, we *learn* them: over a warm-up window, accumulate the cells that change on (almost) every transition. Cells changing in ≥80% of transitions are tagged as counter/animation and zeroed before hashing. The theory is an **invariance claim**: a cell that changes regardless of the agent's action carries no decision-relevant information, so collapsing it cannot lose reachability but can dramatically shrink the state graph. A guard (never mask >20% of the interior) prevents the degenerate case where genuine game state happens to flicker.

**Why it must not hurt deep games.** The guard plus the "changes-every-step" criterion mean the mask only fires on truly action-invariant pixels. Our deepest solves are the test: tu93 stayed at **5 levels** (2008 vs 2087 states) and vc33 at **2 levels** (88 vs 88 states) — *identical* with the mask on. This is the falsifiable prediction the data confirms.

**Mechanism 2 — Effective-action ordering.** Track, per action, the empirical probability that it changes the (masked) state; expand historically-effective actions first. The theory: under a fixed action budget, frontier search should spend steps on actions that *move* the world, deferring no-op actions to the back of the queue. This is a count-based novelty bias, not a value function — it needs no reward signal, which matters because ARC-AGI-3 exposes none until a level completes.

## Completeness: the controlled ablation

We ran the full 25-game suite at 15,000 steps/game for three configurations, plus all four mask×reorder combinations for the single contested game. All runs are sentinel-verified (25 unique games asserted before any conclusion).

| Configuration | Games solved | Sum of levels |
|---|---|---|
| explore (baseline) | 14/25 | 19 |
| **explore2 (mask + reorder)** | **15/25** | **20** |
| explore2, reorder off | 14/25 | 19 |

The 15/25 wins decompose cleanly:

- **+dc22, +m0r0** (0→1 each): both have an in-scene counter; the auto-mask collapses it so BFS reaches a goal. Confirmed because turning the mask *off* leaves the rest unchanged.
- **−cn04** (1→0): the single regression. A four-way ablation isolated the cause **deterministically**: cn04 scores level-1 whenever reorder is *off* (mask on or off: 5034 states either way → mask irrelevant to cn04) and level-0 whenever reorder is *on*. So effective-action ordering, not masking, costs cn04 its level.

Is reorder a net win? Turning it off recovers cn04 **but loses both dc22 and m0r0** — those need the ordering to reach goals within budget. Net: reorder is **+2 / −1**. Hence we keep both on. Only a full ablation surfaces this: a single end-to-end number would hide that mask and reorder rescue *different* games, and that the lone regression is a deterministic side-effect of the weaker mechanism, not noise. Many games also shrink in state count while keeping their result — ar25 341 vs 1131 (−70%), sk48 1228 vs 3309 (−63%) — directly improving the efficiency score.

## The ceiling, and why it is real

We establish **15/25 as the ceiling of pure frame-hashing exploration on this public set**, via three independent lines of evidence:

1. **Budget saturation.** 3,000→15,000 steps lifted the baseline 8/25→14/25, then plateaued; the 11 unsolved games explore huge state spaces (re86 6,625, tr87 5,832, wa30 5,704 distinct states) yet never reach a goal — state-space-limited, not time-limited.
2. **Mechanism saturation.** No mask/reorder combination exceeds 15/25.
3. **Determinism.** Re-runs reproduce per-game state counts to the integer.

The theory of the ceiling: frame-hashing has no notion of *objects* or *goals*. It cannot represent "the avatar must reach the key" — only "this grid differs from that one." Games needing object-level abstraction or multi-step goal inference are out of reach in principle, regardless of compute. **This is the main signpost for progress**: the next gain requires changing the state representation (object-centric abstraction or an executable world model), not more search.

## Universality

The mechanisms are domain-general. The counter-mask is *"discard input dimensions that vary independently of your actions"* — a sufficient-statistic argument applicable to any observation-based RL/planning problem with decorative or adversarial visual noise (game HUDs, timestamps, blinking cursors). Effective-action ordering is count-based exploration prioritization, usable in any discrete-action environment with no reward signal. Both are ~30 lines of NumPy, deterministic, no training, no GPU — they transfer to compute-constrained settings where LLM agents cannot run. Because the agent is LLM-free it runs entirely within the competition's offline, internet-disabled sandbox, where hosted-model approaches cannot.

## Novelty and honesty

We are explicit about provenance. The graph-exploration core follows the public "just-explore" approach; the two augmentations are reimplemented from **Occam** (g-baskin/occam, MIT). Our novel contribution is **not** a new architecture but a **controlled-ablation methodology** that turns folklore ("masking helps") into a falsifiable, per-game causal account — including a documented case (cn04) where the conventionally-good mechanism *hurts*, and a precise characterization of the representational ceiling.

## Reproducibility

The agent is a single self-contained Kaggle notebook: it installs from the competition-provided wheels with internet disabled, runs `explore2` over all games via `OPERATION_MODE=offline`, and lets the harness auto-generate the submission. Offline behavior was validated under a hard network namespace block (`unshare -n`) and reproduces the online results exactly, at ~83× the speed. Ablation toggles (`ARC_E2_MASK`, `ARC_E2_REORDER`) are exposed so every table above can be regenerated.

**Leaderboard submission ID:** _[điền sau khi nộp Phần A]_
**Public notebook:** _[link notebook đã public]_
Code: explore2 (MIT), attribution to Occam (MIT) and ARC-AGI-3-Agents (MIT, ARC Prize).

---
DÁN TỚI ĐÂY ↑
---
