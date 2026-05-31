"""
Single-GPU Hardware Translation (A100→H100) — analysis and plots.
Folders:
  Single Turn : l5_hw_translation_single_turn_final_gpt
  Default     : l5_hw_translation_default_final_gpt
  All Tools   : l5_hw_translation_all_final_gpt
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

OUTDIR = "/scratch/abaner/KernelBench/plots/hw_single_gpu"
os.makedirs(OUTDIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

# ── problem metadata ──────────────────────────────────────────────────────────
PROBLEM_NAMES = {
    1:  "Paged Attn v1",   2:  "Paged Attn v2",
    3:  "Fused RMSNorm",   4:  "SwiGLU",
    5:  "Rotary Embed",    6:  "Custom AllReduce",
    7:  "Marlin INT4 GEMM", 8:  "INT8 W8A8 GEMM",
    9:  "Flash Attn2 Fwd", 10: "Flash Attn2 Bwd",
    22: "TF32 TensorOp",   23: "Gather-Scatter",
    25: "Grouped GEMM",    26: "Permuted GEMM",
    27: "Batched GEMM",    28: "Turing TensorOp",
}
CUTLASS_PIDS     = {22, 23, 25, 26, 27, 28}
VLLM_MARLIN_PIDS = {1, 2, 3, 4, 5, 6, 7, 8}
FA_PIDS          = {9, 10}


def kernel_family(pid):
    if pid in CUTLASS_PIDS:     return "CUTLASS"
    if pid in FA_PIDS:          return "FlashAttn-2"
    return "vLLM/Marlin"


def model_harness_legend(ax, models=True, harnesses=True, **legend_kw):
    """Attach a combined model-color + harness-marker legend."""
    handles = []
    if models:
        handles += [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
                    for m in MODEL_ORDER]
    if harnesses:
        handles += [mlines.Line2D([], [], color="grey",
                                  marker=HARNESS_MARKERS[h], linestyle="None",
                                  markersize=7, label=h)
                    for h in HARNESS_ORDER]
    kw = dict(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=9)
    kw.update(legend_kw)
    ax.legend(handles=handles, **kw)


# ── data loading ─────────────────────────────────────────────────────────────
def load_run(run_dir, harness_label):
    records = []
    base = f"/scratch/abaner/KernelBench/runs/{run_dir}/hardware_translation_stub"
    if not os.path.exists(base):
        print(f"  WARNING: {base} not found")
        return records
    for model in sorted(os.listdir(base)):
        if model not in MODEL_ORDER:
            continue
        mdir = os.path.join(base, model)
        if not os.path.isdir(mdir):
            continue
        for traj_file in sorted(glob.glob(f"{mdir}/level_5_problem_*_trajectory.json")):
            with open(traj_file) as f:
                d = json.load(f)
            fr     = d.get("final_result", {}) or {}
            sol    = fr.get("sol_stats",     {}) or {}
            roof   = fr.get("roofline_stats", {}) or {}
            energy = fr.get("energy_stats",   {}) or {}
            pid    = d["problem_id"]
            sol_score = sol.get("sol_score")
            if sol_score == -1:
                sol_score = None
            tool_seq = []
            for turn in d.get("turns", []):
                for tc in turn.get("tool_calls", []):
                    tn = tc.get("tool_name", "")
                    if tn:
                        tool_seq.append(tn)
            outcome  = d.get("outcome", "")
            compiled = outcome not in ("compile_fail", "error", "in_progress", "")
            rt  = fr.get("runtime",     -1)
            ref = fr.get("ref_runtime", -1)
            rec = dict(
                harness              = harness_label,
                model                = model,
                model_label          = MODEL_LABELS.get(model, model),
                problem_id           = pid,
                problem_name         = PROBLEM_NAMES.get(pid, f"P{pid}"),
                kernel_family        = kernel_family(pid),
                outcome              = outcome,
                compiled             = compiled,
                correct              = bool(fr.get("correctness")),
                speedup_vs_ref       = (ref / rt) if rt > 0 and ref > 0 else None,
                turns                = d.get("total_turns",      0),
                tool_calls           = d.get("total_tool_calls", 0),
                input_tokens         = d.get("llm_input_tokens",  0),
                output_tokens        = d.get("llm_output_tokens", 0),
                total_tokens         = d.get("llm_total_tokens",  0),
                wall_clock_s         = d.get("agent_wall_clock_s", 0),
                sol_score            = sol_score,
                compute_util_pct     = sol.get("compute_utilization_pct"),
                dram_util_pct        = sol.get("dram_utilization_pct"),
                bottleneck           = sol.get("bottleneck"),
                dram_bw_gbs          = roof.get("dram_bandwidth_gbs"),
                occupancy_pct        = roof.get("occupancy_pct"),
                pipe_tensor_pct      = roof.get("pipe_tensor_pct"),
                pipe_fma_pct         = roof.get("pipe_fma_pct"),
                l2_hit_rate_pct      = roof.get("l2_hit_rate_pct"),
                stall_long           = (roof.get("warp_stalls") or {}).get("long_scoreboard"),
                stall_short          = (roof.get("warp_stalls") or {}).get("short_scoreboard"),
                n_compile            = tool_seq.count("compile_kernel"),
                n_correctness        = tool_seq.count("run_correctness"),
                n_profile            = tool_seq.count("profile_kernel"),
                n_disasm             = tool_seq.count("disassemble_kernel"),
                n_ert                = tool_seq.count("ert_roofline"),
                n_static             = tool_seq.count("static_check"),
                tool_seq             = tool_seq,
            )
            records.append(rec)
    return records


r_st = load_run("l5_hw_translation_single_turn_final_gpt", "Single Turn")
r_d  = load_run("l5_hw_translation_default_final_gpt",     "Default")
r_at = load_run("l5_hw_translation_all_final_gpt",         "All Tools")
df   = pd.DataFrame(r_st + r_d + r_at)

df["outcome_clean"] = df["outcome"].map({
    "correct":      "Correct",
    "incorrect":    "Incorrect",
    "compile_fail": "Compile Fail",
    "error":        "Error",
    "in_progress":  "Error",
}).fillna(df["outcome"])


def save(fig, name):
    fig.savefig(f"{OUTDIR}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {name}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — General Results  (2×2)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 1: General Results")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

bar_w = 0.22
x3    = np.arange(len(MODEL_ORDER))
off3  = np.linspace(-(len(HARNESS_ORDER)-1)/2*bar_w,
                     (len(HARNESS_ORDER)-1)/2*bar_w,
                     len(HARNESS_ORDER))

# ── 1a  Correctness rate by model × harness ─────────────────────────────────
ax = axes[0, 0]
for i, harness in enumerate(HARNESS_ORDER):
    sub  = df[df["harness"] == harness]
    vals = [sub[sub["model"] == m]["correct"].mean() * 100
            if len(sub[sub["model"] == m]) > 0 else 0.0
            for m in MODEL_ORDER]
    ax.bar(x3 + off3[i], vals, bar_w, label=harness, color=HARNESS_COLORS[harness])
ax.set_xticks(x3)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax.set_ylabel("Correctness rate (%)")
ax.set_title("Correctness by model")
ax.set_ylim(0, 75)
ax.legend(title="Harness", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 1b  Compile rate by model × harness ─────────────────────────────────────
ax = axes[0, 1]
for i, harness in enumerate(HARNESS_ORDER):
    sub  = df[df["harness"] == harness]
    vals = [sub[sub["model"] == m]["compiled"].mean() * 100
            if len(sub[sub["model"] == m]) > 0 else 0.0
            for m in MODEL_ORDER]
    ax.bar(x3 + off3[i], vals, bar_w, label=harness, color=HARNESS_COLORS[harness])
ax.set_xticks(x3)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax.set_ylabel("Compile rate (%)")
ax.set_title("Compilation success by model")
ax.set_ylim(0, 100)
ax.legend(title="Harness", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 1c  Outcome breakdown by harness (stacked bar) ──────────────────────────
ax = axes[1, 0]
out_order  = ["Correct", "Incorrect", "Compile Fail", "Error"]
out_colors = {"Correct": "#4CAF50", "Incorrect": "#FF9800",
              "Compile Fail": "#F44336", "Error": "#9C27B0"}
pivot = (
    df.groupby(["harness", "outcome_clean"]).size()
    .unstack(fill_value=0)
    .reindex(HARNESS_ORDER)
    .reindex(columns=[o for o in out_order if o in df["outcome_clean"].unique()], fill_value=0)
)
bottom = np.zeros(len(HARNESS_ORDER))
for outcome in pivot.columns:
    vals = pivot[outcome].values.astype(float)
    ax.bar(HARNESS_ORDER, vals, bottom=bottom,
           label=outcome, color=out_colors.get(outcome, "grey"))
    bottom += vals
ax.set_ylabel("Problem count")
ax.set_title("Outcome breakdown by harness")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 1d  Speedup per solved kernel — color=model, marker=harness ──────────────
ax = axes[1, 1]
correct_df = df[df["correct"]].copy()
pid_order  = sorted(correct_df["problem_id"].unique())
pid_labels = [PROBLEM_NAMES.get(p, f"P{p}") for p in pid_order]

for model in MODEL_ORDER:
    for harness in HARNESS_ORDER:
        sub = correct_df[(correct_df["model"] == model) & (correct_df["harness"] == harness)]
        for _, row in sub.iterrows():
            xi  = pid_order.index(row["problem_id"])
            spd = row["speedup_vs_ref"]
            kw  = dict(color=MODEL_COLORS[model], marker=HARNESS_MARKERS[harness],
                       s=80, zorder=3)
            if spd is not None and not np.isnan(spd):
                ax.scatter(xi, spd, alpha=0.85, **kw)
            else:
                ax.scatter(xi, 0.05, alpha=0.35, facecolors="none",
                           edgecolors=MODEL_COLORS[model],
                           marker=HARNESS_MARKERS[harness], s=80, zorder=2)

ax.axhline(1.0, color="black", lw=1.2, ls="--", zorder=1, label="H100 reference (1×)")
for harness in HARNESS_ORDER:
    spd_vals = correct_df[
        (correct_df["harness"] == harness) & correct_df["speedup_vs_ref"].notna()
    ]["speedup_vs_ref"]
    if len(spd_vals) >= 2:
        med = spd_vals.median()
        ax.axhline(med, color=HARNESS_COLORS[harness], lw=1.0, ls=":",
                   alpha=0.75, label=f"{harness} median ({med:.2f}×)")

ax.set_xticks(range(len(pid_order)))
ax.set_xticklabels(pid_labels, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Speedup vs. H100 reference")
ax.set_title("Speedup per solved kernel")
ax.set_ylim(0, None)
model_harness_legend(ax, fontsize=8)
sns.despine(ax=ax)

fig.suptitle("A100 → H100 Translation: General Results", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "fig1_general_results")


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — Pareto Frontiers  (2×2)  — color=model, marker=harness
# ════════════════════════════════════════════════════════════════════════════
print("Figure 2: Pareto Frontiers")
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

profiled = df[df["compute_util_pct"].notna() & df["dram_util_pct"].notna()].copy()

def scatter_mh(ax, data, xcol, ycol, **extra_kw):
    """Scatter with color=model, marker=harness."""
    for model in MODEL_ORDER:
        for harness in HARNESS_ORDER:
            sub = data[(data["model"] == model) & (data["harness"] == harness)]
            if len(sub) == 0:
                continue
            ax.scatter(sub[xcol], sub[ycol],
                       color=MODEL_COLORS[model],
                       marker=HARNESS_MARKERS[harness],
                       s=80, alpha=0.8, edgecolors="w", linewidths=0.5,
                       **extra_kw)

# ── 2a  Compute util vs DRAM util ───────────────────────────────────────────
ax = axes[0, 0]
scatter_mh(ax, profiled, "dram_util_pct", "compute_util_pct")
ax.axvline(50, color="grey", lw=0.8, ls=":")
ax.axhline(50, color="grey", lw=0.8, ls=":")
ax.set_xlim(0, 110);  ax.set_ylim(0, 110)
ax.set_xlabel("DRAM utilization (%)")
ax.set_ylabel("Compute utilization (%)")
ax.set_title("Compute vs. memory utilization")
model_harness_legend(ax)
sns.despine(ax=ax)

# ── 2b  SOL Score vs Speedup ────────────────────────────────────────────────
ax = axes[0, 1]
sol_speed = df[df["sol_score"].notna() & df["speedup_vs_ref"].notna()].copy()
scatter_mh(ax, sol_speed, "speedup_vs_ref", "sol_score")
ax.axvline(1.0, color="black", lw=1, ls="--")
ax.set_xlabel("Speedup vs. H100 reference")
ax.set_ylabel("SOL score (0–1)")
ax.set_title("Hardware efficiency vs. speedup")
ax.set_ylim(0, 1.1)
# add ref line to legend manually
ref_line = mlines.Line2D([], [], color="black", ls="--", label="H100 ref (1×)")
handles_base = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in MODEL_ORDER]
handles_base += [mlines.Line2D([], [], color="grey", marker=HARNESS_MARKERS[h],
                                linestyle="None", markersize=7, label=h)
                 for h in HARNESS_ORDER]
handles_base.append(ref_line)
ax.legend(handles=handles_base, bbox_to_anchor=(1.01, 1), loc="upper left",
          frameon=False, fontsize=9)
sns.despine(ax=ax)

# ── 2c  Occupancy vs Tensor-pipe utilization ─────────────────────────────────
ax = axes[1, 0]
occ_df = df[df["occupancy_pct"].notna() & df["pipe_tensor_pct"].notna()].copy()
occ_df = occ_df[occ_df["occupancy_pct"] <= 105]
scatter_mh(ax, occ_df, "occupancy_pct", "pipe_tensor_pct")
ax.set_xlabel("Occupancy (%)")
ax.set_ylabel("Tensor core utilization (%)")
ax.set_title("Occupancy vs. tensor core use")
ax.set_xlim(0, 105);  ax.set_ylim(0, None)
model_harness_legend(ax)
sns.despine(ax=ax)

# ── 2d  Warp stall decomposition ─────────────────────────────────────────────
ax = axes[1, 1]
stall_df = df[df["stall_long"].notna() & df["stall_short"].notna()].copy()
scatter_mh(ax, stall_df, "stall_long", "stall_short")
ax.set_xlabel("Long-scoreboard stalls (memory latency)")
ax.set_ylabel("Short-scoreboard stalls (dep. latency)")
ax.set_title("Warp stall decomposition")
model_harness_legend(ax)
sns.despine(ax=ax)

fig.suptitle("A100 → H100 Translation: Pareto Frontiers", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "fig2_pareto_frontiers")


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — Resource Efficiency  (2×2)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 3: Resource metrics")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

multi_turn  = df[df["harness"].isin(["Default", "All Tools"])].copy()
harness_dat = ["Default", "All Tools"]
x3r = np.arange(len(MODEL_ORDER))
w2  = 0.35
off2 = [-(w2/2), (w2/2)]

# ── 3a  Tokens per correct problem — by model × harness (bar) ────────────────
ax = axes[0, 0]
for i, harness in enumerate(harness_dat):
    rows = []
    for model in MODEL_ORDER:
        sub = multi_turn[(multi_turn["harness"] == harness) & (multi_turn["model"] == model)]
        n_correct = sub["correct"].sum()
        rows.append(sub["total_tokens"].sum() / n_correct / 1e3 if n_correct > 0 else np.nan)
    bars = ax.bar(x3r + off2[i], rows, w2, label=harness, color=HARNESS_COLORS[harness])
    for bar, model in zip(bars, MODEL_ORDER):
        sub = multi_turn[(multi_turn["harness"] == harness) & (multi_turn["model"] == model)]
        nc  = sub["correct"].sum()
        h   = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"n={nc}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x3r)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax.set_ylabel("Tokens per correct problem (thousands)")
ax.set_title("Token cost per solution")
ax.set_ylim(0, None)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 3b  Wall-clock per correct problem — by model × harness (bar) ────────────
ax = axes[0, 1]
for i, harness in enumerate(harness_dat):
    rows = []
    for model in MODEL_ORDER:
        sub = multi_turn[(multi_turn["harness"] == harness) & (multi_turn["model"] == model)]
        n_correct = sub["correct"].sum()
        rows.append(sub["wall_clock_s"].sum() / n_correct / 60 if n_correct > 0 else np.nan)
    bars = ax.bar(x3r + off2[i], rows, w2, label=harness, color=HARNESS_COLORS[harness])
    for bar, model in zip(bars, MODEL_ORDER):
        sub = multi_turn[(multi_turn["harness"] == harness) & (multi_turn["model"] == model)]
        nc  = sub["correct"].sum()
        h   = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"n={nc}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x3r)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax.set_ylabel("Wall-clock time per solution (min)")
ax.set_title("Time cost per solution")
ax.set_ylim(0, None)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 3c  Tokens vs speedup — color=model, marker=harness ─────────────────────
ax = axes[1, 0]
correct_spd = multi_turn[multi_turn["correct"] & multi_turn["speedup_vs_ref"].notna()].copy()
scatter_mh(ax, correct_spd, "total_tokens", "speedup_vs_ref")
# scale x manually since scatter_mh uses raw column values
ax.set_xscale("linear")
# redo with /1e3 scaling
ax.cla()
for model in MODEL_ORDER:
    for harness in harness_dat:
        sub = correct_spd[(correct_spd["model"] == model) & (correct_spd["harness"] == harness)]
        if len(sub) == 0:
            continue
        ax.scatter(sub["total_tokens"] / 1e3, sub["speedup_vs_ref"],
                   color=MODEL_COLORS[model], marker=HARNESS_MARKERS[harness],
                   s=90, alpha=0.85, edgecolors="w", linewidths=0.5)
ax.axhline(1.0, color="black", lw=1, ls="--")
ax.set_xlabel("Total tokens used (thousands)")
ax.set_ylabel("Speedup vs. H100 reference")
ax.set_title("Tokens spent vs. speedup")
ax.set_ylim(0, None)
model_harness_legend(ax, harnesses=False)   # only model colors needed here (2 harnesses, same markers)
# add harness markers too since both D and AT present
model_harness_legend(ax)
sns.despine(ax=ax)

# ── 3d  Turns to outcome — color=model, marker=correct/incorrect ─────────────
ax = axes[1, 1]
np.random.seed(42)
jitter = 0.10
harness_pos = {h: i for i, h in enumerate(harness_dat)}
outcome_markers = {True: "*", False: "X"}   # * = correct, X = incorrect
outcome_sizes   = {True: 100, False: 70}

for model in MODEL_ORDER:
    for flag, marker, size in [(True, "*", 110), (False, "X", 70)]:
        sub = multi_turn[(multi_turn["model"] == model) & (multi_turn["correct"] == flag) &
                         multi_turn["harness"].isin(harness_dat)]
        xs = [harness_pos[row["harness"]] +
              np.random.uniform(-jitter, jitter)
              for _, row in sub.iterrows()]
        ax.scatter(xs, sub["turns"].values,
                   color=MODEL_COLORS[model], marker=marker,
                   s=size, alpha=0.7, zorder=3,
                   label=f"{MODEL_LABELS[model]} ({'✓' if flag else '✗'})")

# per-harness mean bars for correct and incorrect (pooled across models)
for k, (flag, color) in enumerate([(True, "#4CAF50"), (False, "#F44336")]):
    for i, harness in enumerate(harness_dat):
        hsub = multi_turn[(multi_turn["harness"] == harness) & (multi_turn["correct"] == flag)]
        if len(hsub) > 0:
            xpos = i + (k - 0.5) * 0.35
            ax.hlines(hsub["turns"].mean(), xpos - 0.12, xpos + 0.12,
                      color=color, lw=2.5, zorder=4)

ax.set_xticks(range(len(harness_dat)))
ax.set_xticklabels(harness_dat)
ax.set_ylabel("Turns used")
ax.set_title("Turns to outcome")
ax.set_ylim(0, None)
# legend: model colors + shape for outcome
m_handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in MODEL_ORDER]
o_handles = [mlines.Line2D([], [], color="grey", marker="*", linestyle="None",
                            markersize=9, label="Correct"),
             mlines.Line2D([], [], color="grey", marker="X", linestyle="None",
                            markersize=7, label="Incorrect")]
ax.legend(handles=m_handles + o_handles,
          bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=9)
sns.despine(ax=ax)

fig.suptitle("A100 → H100 Translation: Resource Efficiency", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "fig3_resource_metrics")


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — Trajectory & Tool-Use Analysis  (2×2)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 4: Trajectory analysis")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

at_df = df[df["harness"] == "All Tools"].copy()

# ── 4a  Tool call frequency: correct vs incorrect (All Tools) ────────────────
ax = axes[0, 0]
tool_cols  = ["n_compile", "n_correctness", "n_profile", "n_disasm", "n_ert", "n_static"]
tool_names = ["compile", "run_correct", "profile", "disassemble", "ert_roofline", "static_check"]
correct_mean   = at_df[at_df["correct"]][tool_cols].mean()
incorrect_mean = at_df[~at_df["correct"]][tool_cols].mean()
xi = np.arange(len(tool_cols))
ax.bar(xi - 0.2, correct_mean,   0.35, label="Correct",   color="#4CAF50")
ax.bar(xi + 0.2, incorrect_mean, 0.35, label="Incorrect",  color="#F44336")
ax.set_xticks(xi)
ax.set_xticklabels(tool_names, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Mean calls per problem")
ax.set_title("Tool calls: correct vs. incorrect (AT)")
ax.set_ylim(0, None)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 4b  Compile iterations — model on y-axis, color=model, correct=star ──────
ax = axes[0, 1]
for model in MODEL_ORDER:
    sub = at_df[at_df["model"] == model]
    if len(sub) == 0:
        continue
    for flag, marker, size in [(True, "*", 110), (False, "o", 55)]:
        fsub = sub[sub["correct"] == flag]
        if len(fsub) == 0:
            continue
        ax.scatter(fsub["n_compile"],
                   [MODEL_LABELS[model]] * len(fsub),
                   color=MODEL_COLORS[model],
                   marker=marker, s=size, alpha=0.8, zorder=3)

m_handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in MODEL_ORDER]
o_handles = [mlines.Line2D([], [], color="grey", marker="*", linestyle="None",
                            markersize=9, label="Correct"),
             mlines.Line2D([], [], color="grey", marker="o", linestyle="None",
                            markersize=7, label="Incorrect")]
ax.legend(handles=m_handles + o_handles,
          bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=9)
ax.set_xlabel("Number of compile_kernel calls")
ax.set_title("Compile attempts per problem (AT)")
ax.axvline(at_df[at_df["correct"]]["n_compile"].mean(),   color="green", lw=1.2, ls="--")
ax.axvline(at_df[~at_df["correct"]]["n_compile"].mean(),  color="red",   lw=1.2, ls="--")
sns.despine(ax=ax)

# ── 4c  Kernel family correctness by harness ─────────────────────────────────
ax = axes[1, 0]
family_order = ["vLLM/Marlin", "FlashAttn-2", "CUTLASS"]
xf   = np.arange(len(family_order))
wf   = 0.22
offf = np.linspace(-(len(HARNESS_ORDER)-1)/2*wf,
                    (len(HARNESS_ORDER)-1)/2*wf,
                    len(HARNESS_ORDER))
for i, harness in enumerate(HARNESS_ORDER):
    sub  = df[df["harness"] == harness]
    vals, cnts = [], []
    for fam in family_order:
        fsub = sub[sub["kernel_family"] == fam]
        vals.append(fsub["correct"].mean() * 100 if len(fsub) > 0 else 0.0)
        cnts.append(len(fsub))
    bars = ax.bar(xf + offf[i], vals, wf, label=harness, color=HARNESS_COLORS[harness])
    for bar, cnt, val in zip(bars, cnts, vals):
        if cnt > 0 and val > 2:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                    f"n={cnt}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(xf)
ax.set_xticklabels(family_order)
ax.set_ylabel("Correctness rate (%)")
ax.set_title("Correctness by kernel family")
ax.set_ylim(0, 90)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 4d  Profiling-tool use vs correctness (All Tools) ────────────────────────
ax = axes[1, 1]
at_df_p = at_df.copy()
at_df_p["used_profile"] = at_df_p["n_profile"] > 0
at_df_p["used_disasm"]  = at_df_p["n_disasm"]  > 0
groups = {
    "No profiling\nor disasm": at_df_p[~at_df_p["used_profile"] & ~at_df_p["used_disasm"]],
    "profile_kernel\nonly":    at_df_p[ at_df_p["used_profile"] & ~at_df_p["used_disasm"]],
    "disasm\nonly":            at_df_p[~at_df_p["used_profile"] &  at_df_p["used_disasm"]],
    "profile +\ndisasm":       at_df_p[ at_df_p["used_profile"] &  at_df_p["used_disasm"]],
}
correct_rates = [g["correct"].mean() * 100 if len(g) > 0 else 0.0 for g in groups.values()]
counts        = [len(g) for g in groups.values()]
bar_colors_g  = ["#9E9E9E", "#2196F3", "#FF9800", "#9C27B0"]
bars = ax.bar(list(groups.keys()), correct_rates, color=bar_colors_g)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"n={cnt}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Correctness rate (%)")
ax.set_title("Profiling tools vs. correctness (AT)")
ax.set_ylim(0, 110)
sns.despine(ax=ax)

fig.suptitle("A100 → H100 Translation: Trajectory Analysis", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "fig4_trajectory_analysis")


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — Compute vs Memory: three panels  (1×3)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 5: Compute vs memory bias")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
bins = np.linspace(0, 100, 21)

# ── 5a  Compute utilization — one histogram per model ───────────────────────
ax = axes[0]
for model in MODEL_ORDER:
    sub = df[df["model"] == model]["compute_util_pct"].dropna()
    if len(sub):
        ax.hist(sub, bins=bins, alpha=0.55,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.axvline(sub.median(), color=MODEL_COLORS[model], lw=1.5, ls="--")
ax.set_xlabel("Compute utilization (%)")
ax.set_ylabel("Problem count")
ax.set_title("Compute utilization")
ax.set_xlim(0, 105);  ax.set_ylim(0, None)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 5b  DRAM utilization — one histogram per model ──────────────────────────
ax = axes[1]
for model in MODEL_ORDER:
    sub = df[df["model"] == model]["dram_util_pct"].dropna()
    if len(sub):
        ax.hist(sub, bins=bins, alpha=0.55,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.axvline(sub.median(), color=MODEL_COLORS[model], lw=1.5, ls="--")
ax.set_xlabel("DRAM utilization (%)")
ax.set_ylabel("Problem count")
ax.set_title("DRAM utilization")
ax.set_xlim(0, 105);  ax.set_ylim(0, None)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 5c  L2 hit rate: correct vs incorrect, scatter colored by model ──────────
ax = axes[2]
l2_df = df[df["l2_hit_rate_pct"].notna()].copy()
outcome_groups = {"Correct": l2_df[l2_df["correct"]], "Incorrect": l2_df[~l2_df["correct"]]}
out_colors_l2  = {"Correct": "#4CAF50", "Incorrect": "#F44336"}
data_l2   = [g["l2_hit_rate_pct"].values for g in outcome_groups.values() if len(g) > 0]
labels_l2 = [lbl for lbl, g in outcome_groups.items() if len(g) > 0]
bplot = ax.boxplot(data_l2, patch_artist=True, labels=labels_l2, widths=0.45,
                   medianprops=dict(color="black", lw=1.5))
for patch, lbl in zip(bplot["boxes"], labels_l2):
    patch.set_facecolor(out_colors_l2[lbl]);  patch.set_alpha(0.35)
# scatter overlay colored by model
np.random.seed(42)
for k, (lbl, g) in enumerate(outcome_groups.items()):
    if len(g) == 0:
        continue
    for model in MODEL_ORDER:
        msub = g[g["model"] == model]
        if len(msub) == 0:
            continue
        jx = np.random.uniform(-0.15, 0.15, size=len(msub))
        ax.scatter(np.full(len(msub), k + 1) + jx, msub["l2_hit_rate_pct"].values,
                   color=MODEL_COLORS[model], alpha=0.7, s=40, zorder=3)
ax.set_ylabel("L2 cache hit rate (%)")
ax.set_title("L2 reuse: correct vs. incorrect")
ax.set_ylim(0, 110)
ax.legend(handles=[mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
                   for m in MODEL_ORDER],
          bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

fig.suptitle("LLMs Neglect Memory: Compute Bias in Generated Kernels", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "fig5_compute_memory_bias")


# ════════════════════════════════════════════════════════════════════════════
# KEY NUMBERS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== KEY NUMBERS ===")
for harness in HARNESS_ORDER:
    sub = df[df["harness"] == harness]
    n   = len(sub)
    cor = sub["correct"].sum()
    com = sub["compiled"].sum()
    spd = sub[sub["speedup_vs_ref"].notna()]["speedup_vs_ref"]
    print(f"\n{harness}  (n={n})")
    print(f"  Correct:  {cor}/{n} = {100*cor/n:.1f}%")
    print(f"  Compiled: {com}/{n} = {100*com/n:.1f}%")
    if len(spd):
        print(f"  Speedup:  mean={spd.mean():.2f}x  median={spd.median():.2f}x  max={spd.max():.2f}x")
    for model in MODEL_ORDER:
        msub = sub[sub["model"] == model]
        mc, mn = msub["correct"].sum(), len(msub)
        print(f"    {MODEL_LABELS[model]}: {mc}/{mn} correct ({100*mc/mn:.0f}%), "
              f"{msub['compiled'].sum()}/{mn} compiled")

at   = df[df["harness"] == "All Tools"]
d_df = df[df["harness"] == "Default"]
print(f"\nAll Tools — avg turns (correct): {at[at['correct']]['turns'].mean():.1f}  "
      f"incorrect: {at[~at['correct']]['turns'].mean():.1f}")
print(f"Default   — avg turns (correct): {d_df[d_df['correct']]['turns'].mean():.1f}  "
      f"incorrect: {d_df[~d_df['correct']]['turns'].mean():.1f}")

prof_used   = at[at["n_profile"] > 0]
prof_unused = at[at["n_profile"] == 0]
print(f"\nAT profile_kernel used ({len(prof_used)}): {prof_used['correct'].mean()*100:.0f}% correct")
print(f"AT profile_kernel unused ({len(prof_unused)}): {prof_unused['correct'].mean()*100:.0f}% correct")

print("\nDone. Plots saved to:", OUTDIR)
