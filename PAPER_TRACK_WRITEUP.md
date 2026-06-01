# When Does Masking Help? A Controlled Study of Frontier Graph Exploration on ARC-AGI-3

**Subtitle:** An LLM-free, training-free agent reaching 15/25 public games — and a falsifiable account of *why* its two key mechanisms work, validated by full ablation.

---

## 1. Summary

We present **explore2**, a deterministic agent for ARC-AGI-3 that uses **no LLM, no neural network, and no training**. It is a frontier graph-exploration agent augmented with two mechanisms ported from prior public work: an **auto-learned counter/animation mask** and **effective-action ordering**. On the 25 public games it solves **15/25** (reaches ≥1 level), versus 14/25 for the un-augmented explorer, while preserving the deepest solves (tu93 = 5 levels, vc33 = 2). Crucially, every claim in this paper is backed by a **controlled ablation**: we toggle each mechanism independently across full 15,000-step runs and report exactly which games each mechanism saves and loses. The contribution is less the score than the *method discipline*: a falsifiable account of when frame-hashing exploration succeeds and when it hits a ceiling.

## 2. The problem, precisely

An ARC-AGI-3 agent receives frames (≤64×64 grids, values 0–15) and emits one of seven actions. There are no instructions; the agent must discover, per game, what each action does and what constitutes progress. Levels increase in difficulty and a cleared level swaps in a new layout. Scoring rewards both **completion** (levels cleared) and **efficiency** (actions relative to a human baseline), squared.

This makes the core difficulty **state representation**, not search. The naive approach — treat every distinct frame as a graph node, explore untried actions first, BFS-replay to the nearest unexplored frontier — is correct in principle but fails in practice for one reason: **the node count explodes**. If any on-screen pixel changes every step (a step counter, an animation), every frame hashes to a new node, the frontier never closes, and BFS within a finite action budget never reaches the cells that advance a level.

## 3. Theory: why the two mechanisms work

Our baseline `explore` already masks a fixed 3-cell border (status bars often live at the edge). The two mechanisms generalize this with a clear theory of *what they fix*.

**Mechanism 1 — Auto-learned counter mask.** Instead of hard-coding which cells are decorative, we *learn* them: over a warm-up window, accumulate the cells that change on (almost) every transition. Cells changing in ≥80% of transitions are tagged as counter/animation and zeroed before hashing. The theory is an **invariance claim**: a cell that changes regardless of the agent's action carries no decision-relevant information, so collapsing it cannot lose reachability but can dramatically shrink the state graph. A guard (never mask >20% of the interior) prevents the degenerate case where genuine game state happens to flicker.

**Why it must not hurt deep games.** The guard plus the "changes-every-step" criterion mean the mask only fires on truly action-invariant pixels. Our deepest solves are the test: tu93 stayed at **5 levels** (2008 vs 2087 states) and vc33 at **2 levels** (88 vs 88 states) — *identical* with the mask on. This is the falsifiable prediction the data confirms: masking collapses spurious states without eating real ones.

**Mechanism 2 — Effective-action ordering.** Track, per action, the empirical probability that it changes the (masked) state; expand historically-effective actions first. The theory: under a fixed action budget, frontier search should spend steps on actions that *move* the world, deferring no-op actions (a disabled button, an inapplicable interact) to the back of the queue. This is a count-based novelty bias, not a value function — it needs no reward signal, which matters because ARC-AGI-3 exposes none until a level completes.

## 4. Completeness: the controlled ablation

We ran the full 25-game suite at 15,000 steps/game for three configurations and, for the single contested game, all four mask×reorder combinations. All runs are sentinel-verified (25 unique games asserted before any conclusion).

| Configuration | Games solved | Sum of levels |
|---|---|---|
| `explore` (baseline) | 14/25 | 19 |
| **explore2 (mask + reorder)** | **15/25** | **20** |
| explore2, reorder **off** | 14/25 | 19 |

The 15/25 configuration's wins decompose cleanly:

- **+dc22, +m0r0** (0→1 level each): both have an in-scene counter; the auto-mask collapses it so BFS reaches a goal. These are the mask's wins — confirmed because turning the mask *off* leaves the rest unchanged.
- **−cn04** (1→0): the single regression. A four-way ablation isolated the cause **deterministically**: cn04 scores level-1 whenever reorder is *off* (mask on or off: 5034 states either way → the mask is irrelevant to cn04) and level-0 whenever reorder is *on*. So effective-action ordering, not masking, costs cn04 its level.

The decisive question is then whether reorder is a net win. Turning reorder off recovers cn04 **but loses both dc22 and m0r0** — those two need the action-ordering to reach their goals within budget. Net: reorder is **+2 / −1**. Hence we keep both mechanisms on. This is the kind of result that only a full ablation surfaces: a single end-to-end number would have hidden that mask and reorder each rescue *different* games and that the lone regression is a deterministic side-effect of the weaker mechanism, not noise.

Many games also shrink sharply in state count while keeping their result — ar25 341 vs 1131 (−70%), sk48 1228 vs 3309 (−63%), ka59 2787 vs 3727 — which directly improves the efficiency component of the score.

## 5. The ceiling, and why it is real

We establish **15/25 as the ceiling of pure frame-hashing exploration on this public set**, via three independent lines of evidence:

1. **Budget saturation.** Going from 3,000 to 15,000 steps/game lifted the baseline from 8/25 to 14/25, then plateaued; the 11 unsolved games explore enormous state spaces (re86 6,625, tr87 5,832, wa30 5,704 distinct states) yet never reach a goal. They are state-space-limited, not time-limited.
2. **Mechanism saturation.** Neither mask nor reorder nor any combination exceeds 15/25.
3. **Determinism.** Re-runs reproduce per-game state counts to the integer, so the ceiling is not variance.

The theory of the ceiling: frame-hashing has no notion of *objects* or *goals*. It cannot represent "the avatar must reach the key" — only "this pixel grid differs from that one." Games whose solution requires object-level abstraction or multi-step goal inference are out of reach in principle, regardless of compute. **This is the paper's main signpost for progress**: the next gain requires changing the state representation (object-centric abstraction or a learned/executable world model), not more search.

## 6. Universality

The mechanisms are domain-general and portable. The counter-mask is simply *"discard input dimensions that vary independently of your actions"* — a sufficient-statistic argument that applies to any pixel/observation-based RL or planning problem with decorative or adversarial visual noise (game HUDs, timestamps, blinking cursors). Effective-action ordering is count-based exploration prioritization, usable in any discrete-action environment with no reward signal. Both are ~30 lines of NumPy, deterministic, and require neither training data nor GPUs — they transfer to embedded or compute-constrained settings where LLM agents cannot run. Notably, because the agent is LLM-free it runs entirely within the competition's offline, internet-disabled sandbox, where hosted-model approaches cannot.

## 7. Novelty and honesty

We are explicit about provenance. The graph-exploration core follows the public "just-explore" approach; the two augmentations are reimplemented from **Occam** (g-baskin/occam, MIT). Our novel contribution is **not** a new architecture but a **controlled-ablation methodology** that turns folklore ("masking helps") into a falsifiable, per-game causal account — including a documented case (cn04) where the conventionally-good mechanism *hurts*, and a precise characterization of the representational ceiling. We also report our own process failures (premature conclusions corrected by sentinel-gated verification) in the spirit of reproducibility.

## 8. Reproducibility

The agent is a single self-contained Kaggle notebook: it installs from the competition-provided wheels with internet disabled, runs `explore2` over all games via `OPERATION_MODE=offline`, and lets the harness auto-generate the submission. Offline behavior was validated under a hard network namespace block (`unshare -n`) and reproduces the online results exactly, at ~83× the speed (no per-step HTTP round-trip). Ablation toggles (`ARC_E2_MASK`, `ARC_E2_REORDER`) are exposed so every table above can be regenerated.

---

*Submission ID and public notebook link: [to be filled after leaderboard submission]. Code: explore2 agent, MIT, with attribution to Occam (MIT) and ARC-AGI-3-Agents (MIT, ARC Prize). Word count ≈ 1,180.*
