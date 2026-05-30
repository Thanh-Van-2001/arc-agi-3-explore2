"""Long-run driver for ARC-AGI-3: sweep all games with a big per-game step
budget, checkpoint after every game, low-noise logging. Built for the bee
server (server-first). Run inside tmux:

    ./.venv/bin/python longrun.py --steps 3000

Results -> longrun_results.json (rewritten after each game so partial runs
survive a kill). Scorecard tracked online.
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

logging.getLogger().setLevel(logging.WARNING)
OUT = "longrun_results.json"  # overridable via --out


def run_game(arc, card, cls, gid, max_steps):
    env = arc.make(gid, scorecard_id=card)
    ag = cls(
        card_id=card,
        game_id=gid,
        agent_name="longrun",
        ROOT_URL="https://three.arcprize.org",
        record=False,
        arc_env=env,
    )
    ag.timer = time.time()
    max_levels = 0
    resets = 0
    steps = 0
    final = "?"
    cons = 0
    while steps < max_steps:
        latest = ag._convert_raw_frame_data(env.observation_space)
        if latest.state is GameState.WIN:
            final = "WIN"
            break
        a = ag.choose_action(ag.frames, latest)
        if a.name == "RESET":
            resets += 1
            cons += 1
        else:
            cons = 0
        fr = ag.take_action(a)
        if fr:
            ag.append_frame(fr)
            max_levels = max(max_levels, fr.levels_completed or 0)
            final = fr.state.name
        steps += 1
        if cons >= 20:
            final = final + "+RESET_LOOP"
            break
    return {
        "game": gid.split("-")[0],
        "game_id": gid,
        "steps": steps,
        "max_levels": max_levels,
        "resets": resets,
        "distinct_states": len(getattr(ag, "_nodes", {}) or {}),
        "final_state": final,
        "seconds": round(time.time() - ag.timer, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    global OUT
    ap.add_argument("--agent", default="explore")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--games", default="",
                    help="comma-separated game prefixes to restrict to (default: all)")
    ap.add_argument("--out", default="longrun_results.json", help="results json path")
    a = ap.parse_args()
    OUT = a.out
    cls = AVAILABLE_AGENTS[a.agent]
    arc = Arcade()
    games = sorted(e.game_id for e in arc.get_environments())
    if a.games.strip():
        prefixes = [p.strip() for p in a.games.split(",") if p.strip()]
        games = [g for g in games if any(g.startswith(p) for p in prefixes)]
    card = arc.open_scorecard(tags=["longrun", a.agent])
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print("START %s scorecard=%s games=%d steps/game=%d"
          % (started, card, len(games), a.steps), flush=True)
    print("scorecard_url=https://three.arcprize.org/scorecards/%s" % card, flush=True)
    results = []
    for i, gid in enumerate(games, 1):
        t = time.strftime("%H:%M:%S")
        print("[%s] (%d/%d) %s ..." % (t, i, len(games), gid.split("-")[0]), flush=True)
        try:
            r = run_game(arc, card, cls, gid, a.steps)
        except Exception as e:  # noqa: BLE001
            r = {"game": gid.split("-")[0], "game_id": gid, "error": repr(e)}
        results.append(r)
        print("   -> levels=%s states=%s final=%s %ss"
              % (r.get("max_levels"), r.get("distinct_states"),
                 r.get("final_state"), r.get("seconds")), flush=True)
        json.dump(results, open(OUT, "w"), indent=2)
    try:
        arc.close_scorecard(card)
    except Exception:  # noqa: BLE001
        pass
    solved = sum(1 for r in results if (r.get("max_levels") or 0) >= 1)
    print("DONE %s games_with_level=%d/%d"
          % (time.strftime("%Y-%m-%d %H:%M:%S"), solved, len(games)), flush=True)


if __name__ == "__main__":
    main()
