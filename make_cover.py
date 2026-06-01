"""Generate the Paper Track cover image from the real result files.

Two panels:
  (left)  games solved: explore 14 vs explore2 15 vs reorder-off 14
  (right) per-game ablation grid showing which mechanism saves/loses which game
Outputs cover.png (1600x900, suitable as a Kaggle Writeup cover).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

base = {r["game"]: r for r in json.load(open("longrun_full15k_2026-05-31.json"))}
e2 = {r["game"]: r for r in json.load(open("longrun_explore2_15k.json"))}
r0 = {r["game"]: r for r in json.load(open("longrun_explore2_r0_15k.json"))}

def wins(d):
    return sum(1 for v in d.values() if (v.get("max_levels") or 0) >= 1)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 9), width_ratios=[1, 1.45])
fig.suptitle("explore2: LLM-free graph exploration on ARC-AGI-3\n"
             "Controlled ablation of an auto-learned counter mask + effective-action ordering",
             fontsize=20, fontweight="bold")

# -- left: solved counts --
labels = ["explore\n(baseline)", "explore2\n(mask+reorder)", "explore2\n(reorder off)"]
vals = [wins(base), wins(e2), wins(r0)]
colors = ["#9aa7b4", "#2e7d32", "#c2783f"]
bars = axL.bar(labels, vals, color=colors, width=0.62)
axL.set_ylim(0, 16)
axL.set_ylabel("Public games solved (>=1 level)  /  25", fontsize=13)
axL.set_title("Best config = mask + reorder", fontsize=15)
for b, v in zip(bars, vals):
    axL.text(b.get_x() + b.get_width()/2, v + 0.2, str(v),
             ha="center", fontsize=18, fontweight="bold")
axL.axhline(wins(base), ls="--", lw=1, color="#9aa7b4")
axL.grid(axis="y", alpha=0.25)

# -- right: per-game level grid for the contested + deep games --
games = ["dc22", "m0r0", "cn04", "tu93", "vc33", "ar25", "sk48", "ka59"]
cfgs = [("explore", base), ("mask+reorder", e2), ("reorder off", r0)]
cell = []
for _, d in cfgs:
    cell.append([(d[g].get("max_levels") or 0) for g in games])

im = axR.imshow(cell, cmap="Greens", vmin=0, vmax=5, aspect="auto")
axR.set_xticks(range(len(games)))
axR.set_xticklabels(games, fontsize=12)
axR.set_yticks(range(len(cfgs)))
axR.set_yticklabels([c[0] for c in cfgs], fontsize=12)
axR.set_title("Levels per game — mask saves dc22/m0r0; reorder net +2/-1 (cn04)",
              fontsize=13)
for i in range(len(cfgs)):
    for j in range(len(games)):
        axR.text(j, i, str(cell[i][j]), ha="center", va="center",
                 color="black", fontsize=13, fontweight="bold")
cbar = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04)
cbar.set_label("levels completed", fontsize=11)

fig.text(0.5, 0.02,
         "No LLM - no GPU - no training - runs in the offline sandbox. "
         "15/25 verified, identical online/offline, ~83x faster than HTTP.",
         ha="center", fontsize=12, style="italic")
fig.tight_layout(rect=[0, 0.04, 1, 0.93])
fig.savefig("cover.png", dpi=100)
print("wrote cover.png", "wins:", vals)
