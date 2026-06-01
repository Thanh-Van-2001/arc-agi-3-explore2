"""Build a SELF-CONTAINED Kaggle notebook for the explore2 ARC-AGI-3 submission.

Embeds the minimal agent files (base64, so embedded docstrings/quotes can't
break parsing) and emits kaggle_submission_notebook.py — a single cell that
(1) writes the agent package to /kaggle/working, (2) installs deps from the
competition wheels offline, (3) runs explore2 across all games via
OPERATION_MODE=offline so the harness auto-generates the submission.

Run locally:  python build_kaggle_notebook.py
"""
import base64
import pathlib

BUNDLE = pathlib.Path("submission_bundle")
FILES = [
    "agents/__init__.py",
    "agents/agent.py",
    "agents/recorder.py",
    "agents/tracing.py",
    "agents/templates/__init__.py",
    "agents/templates/explore_agent.py",
    "agents/templates/explore2_agent.py",
]
DATA_ROOT = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"

enc = {rel: base64.b64encode((BUNDLE / rel).read_bytes()).decode("ascii")
       for rel in FILES}

lines = []
lines.append("# ARC-AGI-3 submission — explore2 (graph exploration, no LLM, no GPU, OFFLINE)")
lines.append("# Paste this whole file as ONE Kaggle code cell, add the competition")
lines.append("# dataset (right panel -> Add Input), then Save & Run (Commit).")
lines.append("# It writes the agent package, installs deps from the competition wheels")
lines.append("# with internet OFF, runs explore2 across all games; the harness auto-")
lines.append("# generates the submission. Agent = frontier graph explore + auto counter")
lines.append("# mask + effective-action ordering. 15/25 public games (verified offline,")
lines.append("# identical to online, ~83x faster). Techniques ported from Occam (MIT);")
lines.append("# base: ARC-AGI-3-Agents (MIT, ARC Prize).")
lines.append("import os, sys, glob, base64, subprocess, time, json")
lines.append("")
lines.append('DATA_ROOT = %r' % DATA_ROOT)
lines.append('WHEELS = os.path.join(DATA_ROOT, "arc_agi_3_wheels")')
lines.append('ENV_DIR = os.path.join(DATA_ROOT, "environment_files")')
lines.append('PKG_ROOT = "/kaggle/working/_agentpkg"')
lines.append("")
lines.append("# --- 1. agent package (base64-embedded so quotes/docstrings are safe) ---")
lines.append("_FILES = {")
for rel, b64 in enc.items():
    lines.append("    %r: %r," % (rel, b64))
lines.append("}")
lines.append("for _rel, _b64 in _FILES.items():")
lines.append("    _p = os.path.join(PKG_ROOT, _rel)")
lines.append("    os.makedirs(os.path.dirname(_p), exist_ok=True)")
lines.append("    with open(_p, 'wb') as _f:")
lines.append("        _f.write(base64.b64decode(_b64))")
lines.append("sys.path.insert(0, PKG_ROOT)")
lines.append("")
lines.append("# --- 2. install deps from competition wheels (offline) ---")
lines.append("if glob.glob(os.path.join(WHEELS, '*.whl')):")
lines.append("    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',")
lines.append("                    '--find-links', WHEELS, 'arc_agi', 'arcengine', 'numpy',")
lines.append("                    'pydantic', 'requests', 'python-dotenv', 'flask', 'matplotlib'],")
lines.append("                   check=True)")
lines.append("else:")
lines.append("    print('WARNING: wheels not found at', WHEELS)")
lines.append("")
lines.append("# --- 3. run explore2 across all games, offline ---")
lines.append("os.environ['OPERATION_MODE'] = 'offline'")
lines.append("os.environ['ENVIRONMENTS_DIR'] = ENV_DIR")
lines.append("os.environ['ARC_API_KEY'] = ''")
lines.append("")
lines.append("from arc_agi import Arcade")
lines.append("from arcengine import GameState")
lines.append("from agents import AVAILABLE_AGENTS")
lines.append("")
lines.append("MAX_STEPS = 15000")
lines.append("RESET_LOOP_BREAK = 20")
lines.append("")
lines.append("arc = Arcade()")
lines.append("games = sorted(e.game_id for e in arc.get_environments())")
lines.append("print('discovered', len(games), 'games offline')")
lines.append("cls = AVAILABLE_AGENTS['explore2']")
lines.append("card = arc.open_scorecard(tags=['kaggle', 'explore2'])")
lines.append("")
lines.append("results = []")
lines.append("for gid in games:")
lines.append("    env = arc.make(gid, scorecard_id=card)")
lines.append("    if env is None:")
lines.append("        print('skip (no env):', gid); continue")
lines.append("    ag = cls(card_id=card, game_id=gid, agent_name='kaggle',")
lines.append("             ROOT_URL='', record=False, arc_env=env)")
lines.append("    ag.timer = time.time()")
lines.append("    mx = steps = cons = 0")
lines.append("    final = '?'")
lines.append("    while steps < MAX_STEPS:")
lines.append("        latest = ag._convert_raw_frame_data(env.observation_space)")
lines.append("        if latest.state is GameState.WIN:")
lines.append("            final = 'WIN'; break")
lines.append("        a = ag.choose_action(ag.frames, latest)")
lines.append("        cons = cons + 1 if a.name == 'RESET' else 0")
lines.append("        fr = ag.take_action(a)")
lines.append("        if fr:")
lines.append("            ag.append_frame(fr)")
lines.append("            mx = max(mx, fr.levels_completed or 0)")
lines.append("            final = fr.state.name")
lines.append("        steps += 1")
lines.append("        if cons >= RESET_LOOP_BREAK:")
lines.append("            break")
lines.append("    results.append({'game': gid.split('-')[0], 'levels': mx, 'steps': steps,")
lines.append("                    'final': final, 'sec': round(time.time() - ag.timer, 1)})")
lines.append("    print(json.dumps(results[-1]), flush=True)")
lines.append("")
lines.append("try:")
lines.append("    arc.close_scorecard(card)")
lines.append("except Exception:")
lines.append("    pass")
lines.append("solved = sum(1 for r in results if r['levels'] >= 1)")
lines.append("print('DONE games_with_level=%d/%d' % (solved, len(results)))")

out = "\n".join(lines) + "\n"
pathlib.Path("kaggle_submission_notebook.py").write_text(out, encoding="utf-8")
print("wrote kaggle_submission_notebook.py (%d bytes, %d lines)"
      % (len(out), out.count(chr(10)) + 1))
