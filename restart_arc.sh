#!/bin/bash
# Clean restart of the ARC-AGI-3 long-run on bee (server-side, SSH-drop safe).
# Kills stray longrun instances until pgrep hits zero, THEN launches one in tmux.
# Run detached:  nohup bash ~/arc_agi_3/restart_arc.sh >/tmp/restart_arc.out 2>&1 &
cd ~/arc_agi_3
n=0
while [ "$(pgrep -fc longrun.py)" -gt 0 ] && [ $n -lt 30 ]; do
  pkill -9 -f longrun.py 2>/dev/null
  pkill -9 -f "python longrun" 2>/dev/null
  sleep 1
  n=$((n+1))
done
tmux kill-session -t arc 2>/dev/null
sleep 2
rm -f longrun.log longrun_results.json
tmux new-session -d -s arc "cd ~/arc_agi_3 && ./.venv/bin/python longrun.py --steps 3000 > longrun.log 2>&1"
