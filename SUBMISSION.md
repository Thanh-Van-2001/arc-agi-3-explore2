# ARC-AGI-3 Kaggle Submission — explore2 agent

Agent: **explore2** (`agents/templates/explore2_agent.py`), a deterministic
graph-exploration agent. No LLM, no GPU, no training. CPU-only, fully offline
except for the ARC-AGI-3 game API itself.

Result on the 25 public games: **15/25 games reached ≥1 level** (best depth:
tu93=5 levels, vc33=2). Beats the un-modified explorer (14/25).

## How it runs locally
```bash
# .env holds ARC_API_KEY and ARC_BASE_URL=https://three.arcprize.org
.venv/Scripts/python longrun.py --agent explore2 --steps 15000   # full sweep
.venv/Scripts/python main.py --agent=explore2 --game=<game_id>    # single game
```

## Dependencies (minimal — `requirements-submission.txt`)
`arc-agi, arcengine, numpy, pydantic, requests, python-dotenv` only. The
langchain/openai/smolagents stack in `pyproject.toml` is for OTHER templates and
is **not** on the explore2 code path (verified: explore2 imports with those
libs blocked).

## Algorithm (open-source, all original or permissively-licensed)
1. Frontier graph exploration: node = hash(frame) with a 3-cell border mask;
   untried-action-first; BFS replay to the nearest unexplored frontier;
   reset-loop breaker.
2. **Auto-learned counter/animation mask** — interior pixels that change on
   (almost) every transition are treated as a step-counter/animation and zeroed
   before hashing (generalizes the fixed border mask). Ported from Occam.
3. **Effective-action ordering** — track P(action changes the masked state),
   expand historically-effective actions first. Ported from Occam.
   Toggles: `ARC_E2_MASK`, `ARC_E2_REORDER` (both default on; ablation showed
   the pair = 15/25, reorder-off = 14/25, so both stay on).

## Attribution / license
- This repo derives from `arcprize/ARC-AGI-3-Agents` (MIT, ARC Prize).
- Techniques 2 & 3 reimplemented from **Occam** (`g-baskin/occam`, MIT,
  Sean Donahoe) — see `docs/world_model_design.md` and commit history.
- **TODO before submitting:** the competition requires a CC0 or MIT-0 license on
  the submitter's own code. Current LICENSE is MIT (ARC Prize's). Need to add a
  CC0/MIT-0 license covering our contributions and confirm MIT-origin code is
  compatible (MIT is permissive; attribution must be preserved).

## Submission mechanics — CONFIRMED from competition rules
- Submit via a **Kaggle notebook**, must run in **< 12 hours**.
- **No internet during scoring** (sandboxed) → an LLM-free agent like ours is a
  natural fit; most competitors can't call hosted models.
- Runtime budget: up to **$10,000** (third-party compute like Modal/Lambda
  allowed). We need far less — CPU-only.
- Open-source required before private scores are issued.
- Milestones: **#1 June 30 2026**, **#2 Sept 30 2026** ($37.5K each).

## BLOCKER (needs the user)
The exact Kaggle notebook entrypoint/interface (how the harness hands the
notebook the eval games, what object/function it calls, whether the ARC API key
+ endpoint are injected) is on the **authenticated** Kaggle competition page,
which automated fetch can't read. **User must open the logged-in Kaggle
competition → Code/Rules tabs** and copy the submission template so the agent
can be wrapped to match it.
