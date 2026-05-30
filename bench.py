"""Benchmark harness for ARC-AGI-3 agents.

Drives an agent across several games via the Arcade API directly (bypassing the
Swarm threading/logging layer) so we get clean, structured per-game results
written to JSON. Captures: which actions each game exposes, max levels solved,
distinct states discovered, reset count, and final state.

Usage:
    .venv\\Scripts\\python bench.py [--agent explore] [--steps 200] [--games ls20,vc33,...]

Results -> bench_results.json (rewritten after every game so partial runs survive).
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from dotenv import load_dotenv

load_dotenv(".env.example")
load_dotenv(".env", override=True)

from arc_agi import Arcade  # noqa: E402
from arcengine import GameState  # noqa: E402

from agents import AVAILABLE_AGENTS  # noqa: E402

# Quiet the SDK's chatty INFO logging; we print our own progress.
logging.getLogger().setLevel(logging.WARNING)

DEFAULT_GAMES = ["ls20", "vc33", "ft09", "sp80", "cd82", "tn36"]


def run_game(arc, card, agent_cls, gid, max_steps):
    env = arc.make(gid, scorecard_id=card)
    agent = agent_cls(
        card_id=card,
        game_id=gid,
        agent_name="bench",
        ROOT_URL="https://three.arcprize.org",
        record=False,
        arc_env=env,
    )
    agent.timer = time.time()
    actions_seen = {}
    max_levels = 0
    resets = 0
    final = "?"
    steps = 0
    consecutive_resets = 0

    while steps < max_steps:
        latest = agent._convert_raw_frame_data(env.observation_space)
        if latest.state is GameState.WIN:
            final = "WIN"
            break
        action = agent.choose_action(agent.frames, latest)
        name = action.name
        actions_seen[name] = actions_seen.get(name, 0) + 1
        if name == "RESET":
            resets += 1
            consecutive_resets += 1
        else:
            consecutive_resets = 0
        frame = agent.take_action(action)
        if frame:
            agent.append_frame(frame)
            max_levels = max(max_levels, frame.levels_completed or 0)
            final = frame.state.name
        steps += 1
        # Escape hatch: if the agent can only RESET (graph fully explored, no
        # frontier), stop wasting the budget — this is itself a finding.
        if consecutive_resets >= 12:
            final = final + "+RESET_LOOP"
            break

    distinct_states = len(getattr(agent, "_nodes", {}) or {})
    return {
        "game": gid.split("-")[0],
        "game_id": gid,
        "steps": steps,
        "max_levels": max_levels,
        "resets": resets,
        "actions_seen": actions_seen,
        "distinct_states": distinct_states,
        "final_state": final,
        "seconds": round(time.time() - agent.timer, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="explore")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--games", default=",".join(DEFAULT_GAMES))
    args = ap.parse_args()

    agent_cls = AVAILABLE_AGENTS[args.agent]
    shorts = [g.strip() for g in args.games.split(",") if g.strip()]

    arc = Arcade()
    envs = {e.game_id.split("-")[0]: e.game_id for e in arc.get_environments()}
    card = arc.open_scorecard(tags=["bench", args.agent])
    print(f"scorecard={card}  agent={args.agent}  steps/game={args.steps}", flush=True)

    results = []
    for short in shorts:
        gid = envs.get(short)
        if not gid:
            print(f"  {short}: NOT AVAILABLE", flush=True)
            results.append({"game": short, "error": "not available"})
            continue
        print(f"  {short}: running...", flush=True)
        try:
            r = run_game(arc, card, agent_cls, gid, args.steps)
        except Exception as e:  # noqa: BLE001
            r = {"game": short, "game_id": gid, "error": repr(e)}
        results.append(r)
        print(
            f"  {short}: levels={r.get('max_levels')} states={r.get('distinct_states')} "
            f"actions={list((r.get('actions_seen') or {}).keys())} "
            f"final={r.get('final_state')} steps={r.get('steps')} {r.get('seconds')}s",
            flush=True,
        )
        json.dump(results, open("bench_results.json", "w"), indent=2)

    try:
        arc.close_scorecard(card)
    except Exception:  # noqa: BLE001
        pass
    json.dump(results, open("bench_results.json", "w"), indent=2)
    print(f"DONE scorecard_url=https://three.arcprize.org/scorecards/{card}", flush=True)


if __name__ == "__main__":
    main()
