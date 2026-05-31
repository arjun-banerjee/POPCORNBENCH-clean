"""
8xH100 Hardware Translation — analysis and plots.
Folders:
  Single Turn : fl5_hw_translation_8xh100_single_turn_gpt
  All Tools   : sl5_hw_translation_8xh100_all_reval30m_gpt_FIXED_gpt
"""

import sys, types, json, os, glob, collections
# Patch broken IPython stub that confuses matplotlib's backend detection
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
import seaborn as sns

# ── palette & style ─────────────────────────────────────────────────────────
MODEL_ORDER  = ["gpt-5.5-priority", "FW-GLM-5-1", "grok-4.3"]
MODEL_LABELS = {"gpt-5.5-priority": "GPT-5.5", "FW-GLM-5-1": "FW-GLM-5-1", "grok-4.3": "Grok-4.3"}
MODEL_COLORS = dict(zip(MODEL_ORDER, sns.color_palette("Set2", 3)))
HARNESS_ORDER  = ["Single Turn", "All Tools"]
HARNESS_COLORS = {"Single Turn": "#4C72B0", "All Tools": "#DD8452"}
HARNESS_MARKERS = {"Single Turn": "o", "All Tools": "s"}

OUTDIR = "/scratch/abaner/KernelBench/plots/hw_8xh100"
os.makedirs(OUTDIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

# ── data loading ─────────────────────────────────────────────────────────────
PROBLEM_NAMES = {
    1: "Paged Attn v1", 2: "Paged Attn v2", 3: "Fused RMSNorm",
    4: "SwiGLU", 5: "Rotary Embed", 7: "Marlin INT4 GEMM",
    8: "INT8 W8A8 GEMM", 9: "Flash Attn2 Fwd", 10: "Flash Attn2 Bwd",
    22: "TF32 TensorOp GEMM", 23: "Gather-Scatter",
    25: "Grouped GEMM", 26: "Permuted GEMM",
    27: "Batched GEMM", 28: "Turing TensorOp GEMM",
}

def load_run(run_dir, harness_label):
    records = []
    base = f"/scratch/abaner/KernelBench/runs/{run_dir}/hardware_translation_stub"
    if not os.path.exists(base):
        return records
    for model in sorted(os.listdir(base)):
        mdir = os.path.join(base, model)
        if not os.path.isdir(mdir):
            continue
        for traj_file in sorted(glob.glob(f"{mdir}/level_5_problem_*_trajectory.json")):
            with open(traj_file) as f:
                d = json.load(f)
            fr  = d.get("final_result", {}) or {}
            sol  = fr.get("sol_stats",     {}) or {}
            roof = fr.get("roofline_stats", {}) or {}
            energy = fr.get("energy_stats", {}) or {}
            pid = d["problem_id"]
            sol_score = sol.get("sol_score")
            if sol_score == -1:
                sol_score = None

            # build tool sequence
            tool_seq = []
            for turn in d.get("turns", []):
                for tc in turn.get("tool_calls", []):
                    tn = tc.get("tool_name", "")
                    if tn:
                        tool_seq.append(tn)

            outcome = d.get("outcome", "")
            rec = dict(
                harness           = harness_label,
                model             = model,
                model_label       = MODEL_LABELS.get(model, model),
                problem_id        = pid,
                problem_name      = PROBLEM_NAMES.get(pid, f"P{pid}"),
                outcome           = outcome,
                # compiled = anything that isn't a compile_fail (correct and
                # incorrect both imply the kernel compiled; final_result.compiled
                # is unreliable for timeout-recovery cases).
                compiled          = outcome != "compile_fail",
                correct           = bool(fr.get("correctness")),
                runtime_ms        = fr.get("runtime",        -1),
                ref_runtime_ms    = fr.get("ref_runtime",    -1),
                source_runtime_ms = fr.get("source_runtime", -1),
                turns             = d.get("total_turns",     0),
                tool_calls        = d.get("total_tool_calls", 0),
                input_tokens      = d.get("llm_input_tokens",  0),
                output_tokens     = d.get("llm_output_tokens", 0),
                total_tokens      = d.get("llm_total_tokens",  0),
                wall_clock_s      = d.get("agent_wall_clock_s", 0),
                sol_score         = sol_score,
                compute_util_pct  = sol.get("compute_utilization_pct"),
                dram_util_pct     = sol.get("dram_utilization_pct"),
                bottleneck        = sol.get("bottleneck"),
                dominant_pipe     = sol.get("dominant_pipe"),
                dram_bw_gbs       = roof.get("dram_bandwidth_gbs"),
                occupancy_pct     = roof.get("occupancy_pct"),
                pipe_tensor_pct   = roof.get("pipe_tensor_pct"),
                pipe_fma_pct      = roof.get("pipe_fma_pct"),
                l2_hit_rate_pct   = roof.get("l2_hit_rate_pct"),
                registers_per_thread = roof.get("registers_per_thread"),
                shared_mem_per_block = roof.get("shared_mem_per_block"),
                power_w           = energy.get("avg_power_w"),
                energy_j          = energy.get("total_energy_j"),
                stall_long        = (roof.get("warp_stalls") or {}).get("long_scoreboard"),
                stall_short       = (roof.get("warp_stalls") or {}).get("short_scoreboard"),
                n_compile         = tool_seq.count("compile_kernel"),
                n_correctness     = tool_seq.count("run_correctness"),
                n_profile         = tool_seq.count("profile_kernel"),
                n_disasm          = tool_seq.count("disassemble_kernel"),
                n_ert             = tool_seq.count("ert_roofline"),
                n_static          = tool_seq.count("static_check"),
                tool_seq          = tool_seq,
            )
            rt  = fr.get("runtime",     -1)
            ref = fr.get("ref_runtime", -1)
            rec["speedup_vs_ref"] = (ref / rt) if rt > 0 and ref > 0 else None
            records.append(rec)
    return records


r_st = load_run("fl5_hw_translation_8xh100_single_turn_gpt",            "Single Turn")
r_at = load_run("sl5_hw_translation_8xh100_all_reval30m_gpt_FIXED_gpt", "All Tools")
df   = pd.DataFrame(r_st + r_at)

# ── helper ───────────────────────────────────────────────────────────────────
def save(fig, name):
    path = f"{OUTDIR}/{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    path2 = f"{OUTDIR}/{name}.png"
    fig.savefig(path2, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {name}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — General Results  (2×2)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 1: General Results")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ── 1a  Correctness rate by model × harness ─────────────────────────────────
ax = axes[0, 0]
plot_df = (
    df.groupby(["harness", "model"])
    .agg(correct_rate=("correct", "mean"),
         n=("correct", "count"))
    .reset_index()
)
plot_df["model_label"] = plot_df["model"].map(MODEL_LABELS)
x     = np.arange(len(MODEL_ORDER))
width = 0.35
for i, (harness, color) in enumerate(HARNESS_COLORS.items()):
    sub = plot_df[plot_df["harness"] == harness].set_index("model")
    vals = [sub.loc[m, "correct_rate"] * 100 if m in sub.index else 0.0
            for m in MODEL_ORDER]
    ax.bar(x + (i - 0.5) * width, vals, width, label=harness, color=color)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax.set_ylabel("Correctness rate (%)")
ax.set_title("Correctness by model")
ax.set_ylim(0, 55)
ax.legend(title="Harness", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 1b  Compile rate by model × harness ─────────────────────────────────────
ax = axes[0, 1]
plot_df2 = (
    df.groupby(["harness", "model"])
    .agg(compile_rate=("compiled", "mean"))
    .reset_index()
)
for i, (harness, color) in enumerate(HARNESS_COLORS.items()):
    sub = plot_df2[plot_df2["harness"] == harness].set_index("model")
    vals = [sub.loc[m, "compile_rate"] * 100 if m in sub.index else 0.0
            for m in MODEL_ORDER]
    ax.bar(x + (i - 0.5) * width, vals, width, label=harness, color=color)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax.set_ylabel("Compile rate (%)")
ax.set_title("Compilation success by model")
ax.set_ylim(0, 100)
ax.legend(title="Harness", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 1c  Outcome breakdown (stacked bar) — All Tools only ────────────────────
ax = axes[1, 0]
at_df = df[df["harness"] == "All Tools"].copy()
outcome_map = {
    "correct": "Correct",
    "incorrect": "Incorrect",
    "compile_fail": "Compile Fail",
    "timeout": "Timeout",
}
at_df["outcome_clean"] = at_df["outcome"].map(lambda x: outcome_map.get(x, x))
out_order = ["Correct", "Incorrect", "Compile Fail", "Timeout"]
out_colors = {"Correct": "#4CAF50", "Incorrect": "#FF9800", "Compile Fail": "#F44336", "Timeout": "#9E9E9E"}

pivot = (
    at_df.groupby(["model", "outcome_clean"])
    .size()
    .unstack(fill_value=0)
    .reindex(MODEL_ORDER)
    .reindex(columns=[o for o in out_order if o in at_df["outcome_clean"].unique()], fill_value=0)
)
bottom = np.zeros(len(MODEL_ORDER))
for outcome in pivot.columns:
    vals = pivot[outcome].values.astype(float)
    ax.bar([MODEL_LABELS[m] for m in MODEL_ORDER], vals,
           bottom=bottom, label=outcome, color=out_colors.get(outcome, "grey"))
    bottom += vals
ax.set_ylabel("Problem count")
ax.set_title("Outcome breakdown — All Tools")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 1d  Speedup per kernel — all correct problems, by problem name ──────────
# Show every solved kernel; timeout-missing speedups shown as hollow marker at y=0.
ax = axes[1, 1]
correct_df = df[df["correct"]].copy()
# Short problem names ordered by problem_id
pid_order = sorted(correct_df["problem_id"].unique())
pid_labels = [PROBLEM_NAMES.get(p, f"P{p}") for p in pid_order]

for model in MODEL_ORDER:
    for harness in HARNESS_ORDER:
        sub = correct_df[(correct_df["model"] == model) & (correct_df["harness"] == harness)]
        for _, row in sub.iterrows():
            xi = pid_order.index(row["problem_id"])
            spd = row["speedup_vs_ref"]
            if spd is not None and not np.isnan(spd):
                ax.scatter(xi, spd,
                           color=MODEL_COLORS[model],
                           marker=HARNESS_MARKERS[harness],
                           s=90, alpha=0.85, zorder=3,
                           label=f"{MODEL_LABELS[model]} ({harness})")
            else:
                # timeout — kernel correct but no timing; show as hollow at bottom
                ax.scatter(xi, 0.05,
                           color=MODEL_COLORS[model],
                           marker=HARNESS_MARKERS[harness],
                           s=90, alpha=0.4, zorder=2,
                           facecolors="none", edgecolors=MODEL_COLORS[model],
                           label="_nolegend_")

ax.axhline(1.0, color="black", lw=1, ls="--", zorder=1, label="H100 reference (1×)")
ax.set_xticks(range(len(pid_order)))
ax.set_xticklabels(pid_labels, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Speedup vs. H100 reference")
ax.set_title("Speedup per solved kernel")
ax.set_ylim(0, None)
# deduplicated legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(),
          bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)
sns.despine(ax=ax)

fig.suptitle("8×A100 → 8×H100 Translation: General Results", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "fig1_general_results")


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — Pareto Frontiers  (2×2)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 2: Pareto Frontiers")
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Use all records with compiled kernels (have nsight data) from single-turn
# (single-turn always profiles because it runs nsight as part of submit)
profiled = df[df["compute_util_pct"].notna() & df["dram_util_pct"].notna()].copy()

# ── 2a  Compute util vs DRAM util ───────────────────────────────────────────
ax = axes[0, 0]
if len(profiled) > 0:
    for model in MODEL_ORDER:
        sub = profiled[profiled["model"] == model]
        ax.scatter(
            sub["dram_util_pct"], sub["compute_util_pct"],
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
            s=80, alpha=0.8, edgecolors="w", linewidths=0.5,
            marker="^" if sub["harness"].iloc[0] == "All Tools" else "o"
            if len(sub) > 0 else "o",
        )
    # quadrant lines at 50%
    ax.axvline(50, color="grey", lw=0.8, ls=":")
    ax.axhline(50, color="grey", lw=0.8, ls=":")
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 110)
ax.set_xlabel("DRAM utilization (%)")
ax.set_ylabel("Compute utilization (%)")
ax.set_title("Compute vs. memory utilization")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 2b  SOL Score vs Speedup vs reference ───────────────────────────────────
ax = axes[0, 1]
sol_speed = df[df["sol_score"].notna() & df["speedup_vs_ref"].notna()].copy()
if len(sol_speed) > 0:
    for model in MODEL_ORDER:
        sub = sol_speed[sol_speed["model"] == model]
        ax.scatter(sub["speedup_vs_ref"], sub["sol_score"],
                   color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                   s=90, alpha=0.85, edgecolors="w", linewidths=0.5)
ref_line = ax.axvline(1.0, color="black", lw=1, ls="--", label="H100 ref (1×)")
ax.set_xlabel("Speedup vs. H100 reference")
ax.set_ylabel("SOL score (0–1)")
ax.set_title("Hardware efficiency vs. speedup")
ax.set_ylim(0, 1.1)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(),
          bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 2c  Occupancy vs Tensor-pipe utilization (fusion proxy) ─────────────────
ax = axes[1, 0]
occ_df = df[df["occupancy_pct"].notna() & df["pipe_tensor_pct"].notna()].copy()
# Clamp occupancy to 100 (>100 is a multi-kernel artifact)
occ_df = occ_df[occ_df["occupancy_pct"] <= 105].copy()
if len(occ_df) > 0:
    for model in MODEL_ORDER:
        sub = occ_df[occ_df["model"] == model]
        ax.scatter(
            sub["occupancy_pct"], sub["pipe_tensor_pct"],
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
            s=80, alpha=0.8, edgecolors="w", linewidths=0.5,
        )
ax.set_xlabel("Occupancy (%)")
ax.set_ylabel("Tensor core utilization (%)")
ax.set_title("Occupancy vs. tensor core use")
ax.set_xlim(0, 105)
ax.set_ylim(0, None)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 2d  Warp stall: long-scoreboard vs short-scoreboard ─────────────────────
ax = axes[1, 1]
stall_df = df[df["stall_long"].notna() & df["stall_short"].notna()].copy()
if len(stall_df) > 0:
    for model in MODEL_ORDER:
        sub = stall_df[stall_df["model"] == model]
        ax.scatter(
            sub["stall_long"], sub["stall_short"],
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
            s=80, alpha=0.8, edgecolors="w", linewidths=0.5,
        )
ax.set_xlabel("Long-scoreboard stall cycles (memory latency)")
ax.set_ylabel("Short-scoreboard stall cycles (dep. latency)")
ax.set_title("Warp stall decomposition")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

fig.suptitle("8×A100 → 8×H100 Translation: Pareto Frontiers", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "fig2_pareto_frontiers")


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — LLM Resource Efficiency  (2×2)
# All panels relate resource consumption to an outcome metric so that
# "more tokens" and "more time" can be judged against what was achieved.
# ════════════════════════════════════════════════════════════════════════════
print("Figure 3: Resource metrics")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

at_full = df[df["harness"] == "All Tools"].copy()

# ── 3a  Tokens per correct problem by model (All Tools) ─────────────────────
# = total tokens used by model / number of correct problems solved
ax = axes[0, 0]
rows = []
for model in MODEL_ORDER:
    sub = at_full[at_full["model"] == model]
    total_tok = sub["total_tokens"].sum()
    n_correct  = sub["correct"].sum()
    tok_per_correct = total_tok / n_correct if n_correct > 0 else np.nan
    rows.append({"model": model, "tok_per_correct": tok_per_correct / 1e3,
                 "n_correct": n_correct})
eff_df = pd.DataFrame(rows)
bar_vals = [eff_df[eff_df["model"] == m]["tok_per_correct"].values[0] for m in MODEL_ORDER]
bar_colors_m = [MODEL_COLORS[m] for m in MODEL_ORDER]
bars = ax.bar([MODEL_LABELS[m] for m in MODEL_ORDER], bar_vals,
              color=bar_colors_m)
for bar, row in zip(bars, [eff_df[eff_df["model"] == m].iloc[0] for m in MODEL_ORDER]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"n={int(row['n_correct'])}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Tokens per correct problem (thousands)")
ax.set_title("Token cost per solution — All Tools")
ax.set_ylim(0, None)
sns.despine(ax=ax)

# ── 3b  Wall-clock per correct problem by model (All Tools) ─────────────────
ax = axes[0, 1]
rows2 = []
for model in MODEL_ORDER:
    sub = at_full[at_full["model"] == model]
    total_wc  = sub["wall_clock_s"].sum()
    n_correct  = sub["correct"].sum()
    wc_per_correct = total_wc / n_correct if n_correct > 0 else np.nan
    rows2.append({"model": model, "wc_per_correct": wc_per_correct / 60,
                  "n_correct": n_correct})
eff2_df = pd.DataFrame(rows2)
bar_vals2 = [eff2_df[eff2_df["model"] == m]["wc_per_correct"].values[0] for m in MODEL_ORDER]
bars2 = ax.bar([MODEL_LABELS[m] for m in MODEL_ORDER], bar_vals2, color=bar_colors_m)
for bar, row in zip(bars2, [eff2_df[eff2_df["model"] == m].iloc[0] for m in MODEL_ORDER]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"n={int(row['n_correct'])}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Wall-clock time per solution (min)")
ax.set_title("Time cost per solution — All Tools")
ax.set_ylim(0, None)
sns.despine(ax=ax)

# ── 3c  Total tokens vs. speedup achieved (correct problems with speedup) ────
ax = axes[1, 0]
correct_spd = at_full[at_full["correct"] & at_full["speedup_vs_ref"].notna()].copy()
for model in MODEL_ORDER:
    sub = correct_spd[correct_spd["model"] == model]
    if len(sub) == 0:
        continue
    ax.scatter(sub["total_tokens"] / 1e3, sub["speedup_vs_ref"],
               color=MODEL_COLORS[model], label=MODEL_LABELS[model],
               s=90, alpha=0.85, edgecolors="w", linewidths=0.5)
ax.axhline(1.0, color="black", lw=1, ls="--", label="H100 ref (1×)")
ax.set_xlabel("Total tokens used (thousands)")
ax.set_ylabel("Speedup vs. H100 reference")
ax.set_title("Tokens spent vs. speedup — All Tools")
ax.set_ylim(0, None)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(),
          bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 3d  Turns distribution: correct vs incorrect — strip + mean bar ──────────
ax = axes[1, 1]
jitter = 0.12
for k, (correct_flag, color, label) in enumerate(
        [(True, "#4CAF50", "Correct"), (False, "#F44336", "Incorrect")]):
    sub = at_full[at_full["correct"] == correct_flag]
    model_positions = {m: i for i, m in enumerate(MODEL_ORDER)}
    xs = [model_positions[row["model"]] + (k - 0.5) * 0.3 +
          np.random.uniform(-jitter, jitter)
          for _, row in sub.iterrows()]
    ax.scatter(xs, sub["turns"].values, color=color, alpha=0.6, s=55,
               label=label, zorder=3)
    # mean line per model
    for i, model in enumerate(MODEL_ORDER):
        msub = sub[sub["model"] == model]
        if len(msub) > 0:
            ax.hlines(msub["turns"].mean(),
                      i + (k - 0.5) * 0.3 - 0.1,
                      i + (k - 0.5) * 0.3 + 0.1,
                      color=color, lw=2.5, zorder=4)

ax.set_xticks(range(len(MODEL_ORDER)))
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax.set_ylabel("Turns used")
ax.set_title("Turns to outcome — All Tools")
ax.set_ylim(0, None)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

fig.suptitle("8×A100 → 8×H100 Translation: Resource Efficiency", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "fig3_resource_metrics")


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — Trajectory & Tool-Use Analysis  (2×2)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 4: Trajectory analysis")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

at_df = df[df["harness"] == "All Tools"].copy()

# ── 4a  Tool call frequency: correct vs incorrect ────────────────────────────
ax = axes[0, 0]
tool_cols = ["n_compile", "n_correctness", "n_profile", "n_disasm", "n_ert", "n_static"]
tool_names = ["compile", "run_correctness", "profile", "disassemble", "ert_roofline", "static_check"]
correct_mean   = at_df[at_df["correct"]][tool_cols].mean()
incorrect_mean = at_df[~at_df["correct"]][tool_cols].mean()
xi = np.arange(len(tool_cols))
ax.bar(xi - 0.2, correct_mean,   0.35, label="Correct",   color="#4CAF50")
ax.bar(xi + 0.2, incorrect_mean, 0.35, label="Incorrect",  color="#F44336")
ax.set_xticks(xi)
ax.set_xticklabels(tool_names, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Mean calls per problem")
ax.set_title("Tool calls: correct vs. incorrect")
ax.set_ylim(0, None)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 4b  Compile iterations needed ───────────────────────────────────────────
ax = axes[0, 1]
for j, model in enumerate(MODEL_ORDER):
    sub = at_df[at_df["model"] == model]
    ax.scatter(
        sub["n_compile"],
        [MODEL_LABELS[model]] * len(sub),
        c=sub["correct"].map({True: "#4CAF50", False: "#F44336"}),
        s=70, alpha=0.75, zorder=3,
    )
h1 = mpatches.Patch(color="#4CAF50", label="Correct")
h2 = mpatches.Patch(color="#F44336", label="Incorrect")
ax.legend(handles=[h1, h2], bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
ax.set_xlabel("Number of compile_kernel calls")
ax.set_title("Compile attempts per problem")
ax.axvline(at_df[at_df["correct"]]["n_compile"].mean(), color="green", lw=1.2, ls="--")
ax.axvline(at_df[~at_df["correct"]]["n_compile"].mean(), color="red",   lw=1.2, ls="--")
sns.despine(ax=ax)

# ── 4c  Problem solve heatmap ────────────────────────────────────────────────
ax = axes[1, 0]
all_pids = sorted(df["problem_id"].unique())
heatmap_data = {}
for model in MODEL_ORDER:
    row = []
    for harness in HARNESS_ORDER:
        for pid in all_pids:
            sub = df[(df["model"] == model) & (df["harness"] == harness) & (df["problem_id"] == pid)]
            row.append(1 if (len(sub) > 0 and sub.iloc[0]["correct"]) else 0)
    heatmap_data[MODEL_LABELS[model]] = row

col_labels = [f"ST·P{p}" if i < len(all_pids) else f"AT·P{p}"
              for i, p in enumerate(all_pids + all_pids)]

hm_df = pd.DataFrame(heatmap_data).T
hm_df.columns = [f"ST·{PROBLEM_NAMES.get(p,p)}" for p in all_pids] + \
                [f"AT·{PROBLEM_NAMES.get(p,p)}" for p in all_pids]

# Only show problems that were attempted in both harnesses
sns.heatmap(
    hm_df, ax=ax, cmap=["#F44336", "#4CAF50"], vmin=0, vmax=1,
    linewidths=0.5, linecolor="white", cbar=False,
    xticklabels=True, yticklabels=True,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6.5)
ax.set_title("Correctness heatmap by problem")

# ── 4d  Profile tool use: is it a success predictor? ────────────────────────
ax = axes[1, 1]
# Only All Tools; bin by n_profile
at_df_p = at_df.copy()
at_df_p["used_profile"] = at_df_p["n_profile"] > 0
at_df_p["used_disasm"]  = at_df_p["n_disasm"]  > 0
groups = {
    "No profiling\nor disasm":    at_df_p[~at_df_p["used_profile"] & ~at_df_p["used_disasm"]],
    "profile_kernel\nonly":       at_df_p[ at_df_p["used_profile"] & ~at_df_p["used_disasm"]],
    "disasm\nonly":               at_df_p[~at_df_p["used_profile"] &  at_df_p["used_disasm"]],
    "profile +\ndisasm":          at_df_p[ at_df_p["used_profile"] &  at_df_p["used_disasm"]],
}
group_labels = list(groups.keys())
correct_rates = [g["correct"].mean() * 100 for g in groups.values()]
counts        = [len(g) for g in groups.values()]
bar_colors    = ["#9E9E9E", "#2196F3", "#FF9800", "#9C27B0"]
bars = ax.bar(group_labels, correct_rates, color=bar_colors)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"n={cnt}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Correctness rate (%)")
ax.set_title("Profiling tools vs. correctness")
ax.set_ylim(0, 110)
sns.despine(ax=ax)

fig.suptitle("8×A100 → 8×H100 Translation: Trajectory Analysis", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "fig4_trajectory_analysis")


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — Compute vs Memory: three clean panels  (1×3)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 5: Compute vs memory bias")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
prof = df[df["compute_util_pct"].notna()].copy()
bins = np.linspace(0, 100, 21)

# ── 5a  Compute utilization distribution ────────────────────────────────────
ax = axes[0]
for model in MODEL_ORDER:
    sub = prof[prof["model"] == model]["compute_util_pct"].dropna()
    if len(sub):
        ax.hist(sub, bins=bins, alpha=0.6, color=MODEL_COLORS[model],
                label=MODEL_LABELS[model])
    med = sub.median() if len(sub) else None
    if med is not None:
        ax.axvline(med, color=MODEL_COLORS[model], lw=1.5, ls="--")
ax.set_xlabel("Compute utilization (%)")
ax.set_ylabel("Problem count")
ax.set_title("Compute utilization")
ax.set_xlim(0, 105)
ax.set_ylim(0, None)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 5b  DRAM utilization distribution ───────────────────────────────────────
ax = axes[1]
prof2 = df[df["dram_util_pct"].notna()].copy()
for model in MODEL_ORDER:
    sub = prof2[prof2["model"] == model]["dram_util_pct"].dropna()
    if len(sub):
        ax.hist(sub, bins=bins, alpha=0.6, color=MODEL_COLORS[model],
                label=MODEL_LABELS[model])
    med = sub.median() if len(sub) else None
    if med is not None:
        ax.axvline(med, color=MODEL_COLORS[model], lw=1.5, ls="--")
ax.set_xlabel("DRAM utilization (%)")
ax.set_ylabel("Problem count")
ax.set_title("DRAM utilization")
ax.set_xlim(0, 105)
ax.set_ylim(0, None)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.despine(ax=ax)

# ── 5c  L2 hit rate: correct vs incorrect ───────────────────────────────────
ax = axes[2]
l2_df = df[df["l2_hit_rate_pct"].notna()].copy()
outcome_groups = {"Correct": l2_df[l2_df["correct"]], "Incorrect": l2_df[~l2_df["correct"]]}
out_colors_l2  = {"Correct": "#4CAF50", "Incorrect": "#F44336"}
data   = [g["l2_hit_rate_pct"].values for g in outcome_groups.values() if len(g) > 0]
labels = [lbl for lbl, g in outcome_groups.items() if len(g) > 0]
bplot = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.45,
                   medianprops=dict(color="black", lw=1.5))
for patch, lbl in zip(bplot["boxes"], labels):
    patch.set_facecolor(out_colors_l2[lbl])
    patch.set_alpha(0.7)
# overlay individual points
for k, (lbl, g) in enumerate(outcome_groups.items()):
    if len(g) == 0:
        continue
    jx = np.random.uniform(-0.15, 0.15, size=len(g))
    ax.scatter(np.full(len(g), k + 1) + jx, g["l2_hit_rate_pct"].values,
               color=out_colors_l2[lbl], alpha=0.5, s=30, zorder=3)
ax.set_ylabel("L2 cache hit rate (%)")
ax.set_title("L2 reuse: correct vs. incorrect")
ax.set_ylim(0, 110)
sns.despine(ax=ax)

fig.suptitle("LLMs Neglect Memory: Compute Bias in Generated Kernels", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "fig5_compute_memory_bias")


# ════════════════════════════════════════════════════════════════════════════
# Print key numbers for the results section
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
        mc   = msub["correct"].sum()
        mn   = len(msub)
        print(f"    {MODEL_LABELS[model]}: {mc}/{mn} correct ({100*mc/mn:.0f}%)")

at = df[df["harness"] == "All Tools"]
print(f"\nAll Tools — avg turns (correct):   {at[at['correct']]['turns'].mean():.1f}")
print(f"All Tools — avg turns (incorrect): {at[~at['correct']]['turns'].mean():.1f}")
print(f"All Tools — avg tokens (correct):  {at[at['correct']]['total_tokens'].mean():.0f}")
print(f"All Tools — avg tokens (incorrect):{at[~at['correct']]['total_tokens'].mean():.0f}")

prof_used  = at[at["n_profile"] > 0]
prof_unused = at[at["n_profile"] == 0]
print(f"\nAll Tools — profile_kernel used: {len(prof_used)} problems, {prof_used['correct'].mean()*100:.0f}% correct")
print(f"All Tools — profile_kernel unused:{len(prof_unused)} problems, {prof_unused['correct'].mean()*100:.0f}% correct")

if len(profiled) > 0:
    print(f"\nCompute util: mean={profiled['compute_util_pct'].mean():.1f}%  median={profiled['compute_util_pct'].median():.1f}%")
    print(f"DRAM util:    mean={profiled['dram_util_pct'].mean():.1f}%  median={profiled['dram_util_pct'].median():.1f}%")

print("\nDone. Plots saved to:", OUTDIR)
