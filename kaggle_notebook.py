"""ARC-AGI-3 Kaggle submission notebook (explore2 agent).

Convert each ``# %% [cell]`` block into a Kaggle notebook cell. Runs fully
OFFLINE (internet disabled during scoring): installs arc_agi + deps from the
competition-provided wheels, then runs the explore2 graph-exploration agent
(no LLM, no GPU) across every game via OPERATION_MODE=offline. The submission
file is generated automatically by the harness once the agent acts on the games.

Verified offline on bee (Python 3.12) from the competition wheels: arc_agi 0.9.8
LocalEnvironmentWrapper loads games from environment_files and runs ~200x faster
than the live HTTP API (no per-step network round-trip).

Paths below assume the Kaggle dataset mount layout; adjust DATA_ROOT to match the
actual competition input path shown in the notebook's right-hand Data panel.
"""

# %% [cell] 1 — install deps from local wheels (no internet)
import os, subprocess, sys, glob

# TODO(user): confirm the mounted input path on Kaggle (right panel → Data).
DATA_ROOT = "/kaggle/input/arc-prize-2026-arc-agi-3"
WHEELS = os.path.join(DATA_ROOT, "arc_agi_3_wheels")
ENV_DIR = os.path.join(DATA_ROOT, "environment_files")

wheels = sorted(glob.glob(os.path.join(WHEELS, "*.whl")))
if wheels:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index",
         "--find-links", WHEELS,
         "arc_agi", "arcengine", "numpy", "pydantic", "requests",
         "python-dotenv", "flask", "matplotlib"],
        check=True,
    )
else:
    print("WHEELS not found at", WHEELS, "— fix DATA_ROOT")

# %% [cell] 2 — bring in the explore2 agent code
# The agent lives in this repo's agents/ package. On Kaggle, attach it as a
# dataset/utility script, or paste agents/templates/explore2_agent.py +
# explore_agent.py + a minimal agents/__init__.py that registers explore2.
# (Only the no-LLM templates are needed; skip langgraph/openai/smolagents.)
sys.path.insert(0, "/kaggle/input/explore2-agent-code")  # TODO(user): adjust

# %% [cell] 3 — run explore2 offline across all games
os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = ENV_DIR
os.environ["ARC_API_KEY"] = ""  # offline needs no key

import time, json
from arc_agi import Arcade
from arcengine import GameState
from agents import AVAILABLE_AGENTS

MAX_STEPS = 15000          # offline is ~200x faster than HTTP; tune to the 9h cap
RESET_LOOP_BREAK = 20

arc = Arcade()
games = sorted(e.game_id for e in arc.get_environments())
print(f"discovered {len(games)} games offline")
cls = AVAILABLE_AGENTS["explore2"]
card = arc.open_scorecard(tags=["kaggle", "explore2"])

results = []
for gid in games:
    env = arc.make(gid, scorecard_id=card)
    ag = cls(card_id=card, game_id=gid, agent_name="kaggle",
             ROOT_URL="", record=False, arc_env=env)
    ag.timer = time.time()
    mx = steps = cons = 0
    final = "?"
    while steps < MAX_STEPS:
        latest = ag._convert_raw_frame_data(env.observation_space)
        if latest.state is GameState.WIN:
            final = "WIN"; break
        a = ag.choose_action(ag.frames, latest)
        cons = cons + 1 if a.name == "RESET" else 0
        fr = ag.take_action(a)
        if fr:
            ag.append_frame(fr)
            mx = max(mx, fr.levels_completed or 0)
            final = fr.state.name
        steps += 1
        if cons >= RESET_LOOP_BREAK:
            break
    results.append({"game": gid.split("-")[0], "levels": mx, "steps": steps,
                    "final": final, "sec": round(time.time() - ag.timer, 1)})
    print(json.dumps(results[-1]))

try:
    arc.close_scorecard(card)
except Exception:
    pass
solved = sum(1 for r in results if r["levels"] >= 1)
print(f"DONE games_with_level={solved}/{len(results)}")
# The competition harness writes the submission file automatically from the
# scorecard once the agent has acted on the games.
