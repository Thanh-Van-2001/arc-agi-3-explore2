"""560x280 card/thumbnail for the Paper Track writeup (from real result files)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = {r["game"]: r for r in json.load(open("longrun_full15k_2026-05-31.json"))}
e2 = {r["game"]: r for r in json.load(open("longrun_explore2_15k.json"))}
r0 = {r["game"]: r for r in json.load(open("longrun_explore2_r0_15k.json"))}
w = lambda d: sum(1 for v in d.values() if (v.get("max_levels") or 0) >= 1)

fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
ax = fig.add_axes([0.13, 0.20, 0.83, 0.52])
labels = ["explore", "explore2\n(mask+reorder)", "reorder off"]
vals = [w(base), w(e2), w(r0)]
colors = ["#9aa7b4", "#2e7d32", "#c2783f"]
bars = ax.bar(labels, vals, color=colors, width=0.6)
ax.set_ylim(0, 17)
ax.set_ylabel("games solved / 25", fontsize=9)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.3, str(v), ha="center",
            fontsize=13, fontweight="bold")
ax.tick_params(labelsize=8)
ax.grid(axis="y", alpha=0.25)
fig.suptitle("explore2 on ARC-AGI-3: LLM-free, offline, 15/25",
             fontsize=12, fontweight="bold", y=0.95)
fig.text(0.5, 0.04,
         "auto counter-mask + effective-action ordering, fully ablated",
         ha="center", fontsize=8, style="italic")
fig.savefig("card_560x280.png")
print("wrote card_560x280.png", vals)
