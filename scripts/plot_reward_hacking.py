"""
PopcornBench L1-3 Reward Hacking — analysis and plots.

Original runs (reward-hacked):
  pop_l123_st_gpt      → Single Turn
  pop_l123_default_gpt → Default
  pop_l123_all_gpt     → All Tools

Stress-test runs (confirmation):
  pop_l123_default_gpt_stress_gpt
  pop_l123_all_gpt_stress_20260515_071035_476589996_gpt
"""

import sys, types, json, os, glob
if "IPython" not in sys.modules or not hasattr(sys.modules["IPython"], "get_ipython"):
    _ipy = types.ModuleType("IPython")
    _ipy.get_ipython = lambda: None
    sys.modules["IPython"] = _ipy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

# ── palette & style ─────────────────────────────────────────────────────────
MODEL_ORDER = ["gpt-5.5-priority", "FW-GLM-5-1", "grok-4.3"]
MODEL_LABELS = {
    "gpt-5.5-priority": "GPT-5.5",
    "FW-GLM-5-1":       "FW-GLM-5-1",
    "grok-4.3":         "Grok-4.3",
}
MODEL_COLORS = dict(zip(MODEL_ORDER, sns.color_palette("Set2", 3)))

HARNESS_ORDER = ["Single Turn", "Default", "All Tools"]
HARNESS_COLORS = {
    "Single Turn": "#4C72B0",
    "Default":     "#55A868",
    "All Tools":   "#DD8452",
}
HARNESS_MARKERS = {"Single Turn": "o", "Default": "D", "All Tools": "s"}

BASE = "/scratch/abaner/KernelBench/runs"
STUB = "popcorn"
OUTDIR = "/scratch/abaner/KernelBench/plots/reward_hacking"
os.makedirs(OUTDIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── data loading ─────────────────────────────────────────────────────────────
def load_run(run_dir, harness_label):
    records = []
    base = os.path.join(BASE, run_dir, STUB)
    if not os.path.exists(base):
        print(f"  WARNING: {base} not found")
        return records
    for model in sorted(os.listdir(base)):
        if model not in MODEL_ORDER:
            continue
        mdir = os.path.join(base, model)
        if not os.path.isdir(mdir):
            continue
        for tf in sorted(glob.glob(f"{mdir}/level_*_problem_*_trajectory.json")):
            with open(tf) as f:
                d = json.load(f)
            fr  = d.get("final_result", {}) or {}
            sol = fr.get("sol_stats",     {}) or {}
            roof = fr.get("roofline_stats", {}) or {}
            rt  = fr.get("runtime",     -1)
            ref = fr.get("ref_runtime", -1)
            outcome = d.get("outcome", "")
            rec = dict(
                harness         = harness_label,
                model           = model,
                model_label     = MODEL_LABELS.get(model, model),
                problem_id      = d["problem_id"],
                level           = d.get("level", -1),
                problem_name    = d.get("problem_name", ""),
                outcome         = outcome,
                correct         = bool(fr.get("correctness")),
                speedup         = ref / rt if rt > 0 and ref > 0 else None,
                sol_score       = sol.get("sol_score"),
                compute_util    = sol.get("compute_utilization_pct"),
                dram_util       = sol.get("dram_utilization_pct"),
                occupancy       = roof.get("occupancy_pct"),
            )
            # clip sol_score of -1 sentinel
            if rec["sol_score"] == -1:
                rec["sol_score"] = None
            records.append(rec)
    return records


# load originals
r_st = load_run("pop_l123_st_gpt",      "Single Turn")
r_d  = load_run("pop_l123_default_gpt", "Default")
r_at = load_run("pop_l123_all_gpt",     "All Tools")
df_orig = pd.DataFrame(r_st + r_d + r_at)

# load stress tests (only Default and All Tools)
r_ds = load_run("pop_l123_default_gpt_stress_gpt",
                "Default")
r_as = load_run("pop_l123_all_gpt_stress_20260515_071035_476589996_gpt",
                "All Tools")
df_stress = pd.DataFrame(r_ds + r_as)

# build stress lookup: (harness, model, problem_id, level) → stress_correct
stress_lookup = {}
for _, row in df_stress.iterrows():
    key = (row["harness"], row["model"], row["problem_id"], row["level"])
    stress_lookup[key] = row["correct"]

# annotate each original row with stress result
def stress_survived(row):
    key = (row["harness"], row["model"], row["problem_id"], row["level"])
    if key not in stress_lookup:
        return None   # no stress data (pending/missing)
    return stress_lookup[key]

df_orig["stress_correct"] = df_orig.apply(stress_survived, axis=1)


def save(fig, name):
    fig.savefig(f"{OUTDIR}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {name}")


# ════════════════════════════════════════════════════════════════════════════
# FIG A — Speedup vs SOL Score: the smoking gun  (1×2)
# ════════════════════════════════════════════════════════════════════════════
print("Figure A: Reward Hacking Scatter")

# work with originally-correct kernels that have sol and speedup data
plot_df = df_orig[
    df_orig["correct"] &
    df_orig["sol_score"].notna() &
    df_orig["speedup"].notna()
].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Left: all originally-correct, color=model, marker=harness ────────────────
ax = axes[0]
for model in MODEL_ORDER:
    for harness in HARNESS_ORDER:
        sub = plot_df[(plot_df["model"] == model) & (plot_df["harness"] == harness)]
        if len(sub) == 0:
            continue
        ax.scatter(sub["sol_score"], sub["speedup"],
                   color=MODEL_COLORS[model],
                   marker=HARNESS_MARKERS[harness],
                   s=55, alpha=0.65, edgecolors="w", linewidths=0.3,
                   zorder=3)

# Reference line at speedup = 1
ax.axhline(1.0, color="black", lw=1, ls="--", alpha=0.5)
ax.set_yscale("log")
ax.set_xlabel("SOL score (hardware efficiency, 0–1)")
ax.set_ylabel("Speedup vs. reference (log scale)")
ax.set_title("All 'correct' submissions")
ax.set_xlim(-0.02, 1.05)

# model + harness legend
m_handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
             for m in MODEL_ORDER]
h_handles = [mlines.Line2D([], [], color="grey", marker=HARNESS_MARKERS[h],
                            linestyle="None", markersize=7, label=h)
             for h in HARNESS_ORDER]
ax.legend(handles=m_handles + h_handles,
          bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)
sns.despine(ax=ax)

# ── Right: same data, filled=survived stress, hollow=failed ──────────────────
ax = axes[1]

# only plot rows where stress data is available (D and AT original)
stress_df = plot_df[
    plot_df["harness"].isin(["Default", "All Tools"]) &
    plot_df["stress_correct"].notna()
].copy()

for model in MODEL_ORDER:
    for harness in ["Default", "All Tools"]:
        sub = stress_df[(stress_df["model"] == model) & (stress_df["harness"] == harness)]
        survived = sub[sub["stress_correct"] == True]
        failed   = sub[sub["stress_correct"] == False]

        # survived: filled
        if len(survived):
            ax.scatter(survived["sol_score"], survived["speedup"],
                       color=MODEL_COLORS[model],
                       marker=HARNESS_MARKERS[harness],
                       s=60, alpha=0.85, edgecolors="w", linewidths=0.4,
                       zorder=4, label=f"{MODEL_LABELS[model]} / {harness} (survived)")
        # failed: hollow
        if len(failed):
            ax.scatter(failed["sol_score"], failed["speedup"],
                       facecolors="none",
                       edgecolors=MODEL_COLORS[model],
                       marker=HARNESS_MARKERS[harness],
                       s=60, alpha=0.65, linewidths=0.9,
                       zorder=3, label=f"_")

ax.axhline(1.0, color="black", lw=1, ls="--", alpha=0.5)
ax.set_yscale("log")
ax.set_xlabel("SOL score (hardware efficiency, 0–1)")
ax.set_ylabel("Speedup vs. reference (log scale)")
ax.set_title("Stress-test survival (Default + All Tools)")
ax.set_xlim(-0.02, 1.05)

survived_h = mpatches.Patch(color="grey",       label="Survived stress")
failed_h   = mpatches.Patch(facecolor="none",
                             edgecolor="grey",
                             linewidth=1.2,       label="Failed stress")
ax.legend(handles=m_handles + [survived_h, failed_h],
          bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)
sns.despine(ax=ax)

fig.suptitle(
    "Extreme Speedup + Near-Zero Efficiency: Hardware Metrics Reveal Reward Hacking",
    fontsize=12, y=1.02)
fig.tight_layout()
save(fig, "figA_reward_hack_scatter")


# ════════════════════════════════════════════════════════════════════════════
# FIG B — Correctness collapse under stress testing (bar chart)
# ════════════════════════════════════════════════════════════════════════════
print("Figure B: Correctness Collapse")

fig, ax = plt.subplots(figsize=(7, 5))

harnesses_with_stress = ["Default", "All Tools"]
x = np.arange(len(harnesses_with_stress))
w = 0.35

# Show stress-survival rate among originally-correct kernels (cleaner metric)
surv_rates, orig_n, surv_n = [], [], []
for h in harnesses_with_stress:
    orig_correct = df_orig[(df_orig["harness"] == h) & df_orig["correct"] &
                           df_orig["stress_correct"].notna()]
    survived = orig_correct["stress_correct"].sum()
    total    = len(orig_correct)
    surv_rates.append(100 * survived / total if total > 0 else 0)
    orig_n.append(total)
    surv_n.append(int(survived))

bars_o = ax.bar(x - w/2, [100] * len(harnesses_with_stress), w,
                label="Originally correct",
                color=[HARNESS_COLORS[h] for h in harnesses_with_stress])
bars_s = ax.bar(x + w/2, surv_rates, w,
                label="Survived stress test",
                color=[HARNESS_COLORS[h] for h in harnesses_with_stress],
                hatch="///", edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(harnesses_with_stress, fontsize=9)
ax.set_ylabel("% of originally-correct kernels", fontsize=9)
ax.set_title("Stress-test survival among originally-correct kernels", fontsize=10)
ax.set_ylim(0, 108)
ax.tick_params(axis="both", labelsize=8)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)
sns.despine(ax=ax)

fig.tight_layout()
save(fig, "figB_correctness_collapse")


# ════════════════════════════════════════════════════════════════════════════
# KEY NUMBERS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== KEY NUMBERS — ORIGINAL RUNS ===")
for h in HARNESS_ORDER:
    sub = df_orig[df_orig["harness"] == h]
    n   = len(sub)
    c   = sub["correct"].sum()
    spd = sub[sub["speedup"].notna() & sub["correct"]]["speedup"]
    sol = sub[sub["sol_score"].notna() & sub["correct"]]["sol_score"]
    print(f"\n{h} (n={n})")
    print(f"  Correct:     {c}/{n} = {100*c/n:.1f}%")
    if len(spd):
        print(f"  Speedup:     mean={spd.mean():.0f}x  median={spd.median():.1f}x  max={spd.max():.0f}x")
    if len(sol):
        print(f"  SOL score:   mean={sol.mean():.3f}")

print("\n=== STRESS TEST ===")
for h in ["Default", "All Tools"]:
    os_sub = df_orig[df_orig["harness"] == h]
    st_sub = df_stress[df_stress["harness"] == h]
    oc = os_sub["correct"].sum()
    sc = st_sub["correct"].sum()
    n  = len(os_sub)
    sn = len(st_sub)
    print(f"\n{h}:")
    print(f"  Original correct:     {oc}/{n} ({100*oc/n:.1f}%)")
    print(f"  Stress correct:       {sc}/{sn} ({100*sc/sn:.1f}%)")
    print(f"  Collapse:             −{oc-sc} kernels ({100*(oc-sc)/oc:.1f}% relative)")

# stress-test survival rate among originally-correct kernels
print("\n=== STRESS SURVIVAL AMONG ORIG-CORRECT ===")
for h in ["Default", "All Tools"]:
    sub = df_orig[(df_orig["harness"] == h) & df_orig["correct"] & df_orig["stress_correct"].notna()]
    surv = sub["stress_correct"].sum()
    total = len(sub)
    print(f"{h}: {surv}/{total} originally-correct kernels survived ({100*surv/total:.1f}%)")

# speedup distribution of survivors vs failures
print("\n=== SPEEDUP: SURVIVORS vs FAILURES ===")
for h in ["Default", "All Tools"]:
    sub = df_orig[(df_orig["harness"] == h) & df_orig["correct"] &
                  df_orig["stress_correct"].notna() & df_orig["speedup"].notna()]
    surv = sub[sub["stress_correct"] == True]["speedup"]
    fail = sub[sub["stress_correct"] == False]["speedup"]
    print(f"{h}:  survivors median={surv.median():.1f}x  failures median={fail.median():.0f}x")

print("\nDone. Plots saved to:", OUTDIR)
