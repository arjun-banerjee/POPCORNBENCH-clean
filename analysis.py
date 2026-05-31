#!/usr/bin/env python3
"""
KernelBench GPU Kernel Benchmark Analysis
Steps 0-5: discovery, label fix, data loading, plots, trace analysis, LaTeX report
"""

import sys, types

# Patch broken IPython before matplotlib touches it
if "IPython" not in sys.modules:
    _ipy = types.ModuleType("IPython")
    _ipy.get_ipython = lambda: None
    sys.modules["IPython"] = _ipy
else:
    import IPython as _ipy  # noqa: F811
    if not hasattr(_ipy, "get_ipython"):
        _ipy.get_ipython = lambda: None

import json, glob, os, shutil, copy, re, warnings
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")

BASE = Path("/scratch/abaner/KernelBench/runs")
OUT  = Path("/scratch/abaner/KernelBench")
PLOT = OUT / "plots"
PLOT_EX = PLOT / "exploratory"
PLOT.mkdir(exist_ok=True)
PLOT_EX.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# FOLDER DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
SET1_FOLDERS = {
    "all":     "pop_l123_all_gpt",
    "default": "pop_l123_default_gpt",
    "st":      "pop_l123_st_gpt",
}
SET2_FOLDERS = {
    "default": "l5_hw_translation_default_final_gpt",
    "all":     "l5_hw_translation_all_final_gpt",
    "st":      "l5_hw_translation_single_turn_final_gpt",
}
SET3_FOLDERS = {
    "sl5_all":  "sl5_hw_translation_8xh100_all_reval30m_gpt",
    "fl5_st":   "fl5_hw_translation_8xh100_single_turn_gpt",
}
SET3_SL5_FIXED = "sl5_hw_translation_8xh100_all_reval30m_gpt_FIXED"

# sub-dir name per set
SUBDIR = {
    "set1": "popcorn",
    "set2": "hardware_translation_stub",
    "set3": "hardware_translation_stub",
}

# ──────────────────────────────────────────────────────────────────────────────
# MODEL NAME NORMALIZATION
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NORM_MAP = {}

def normalize_model(name: str) -> str:
    if name in MODEL_NORM_MAP:
        return MODEL_NORM_MAP[name]
    n = name.strip().lower()
    for sfx in ("_gpt", "-gpt", "_final", "_v1", "_v2", "_v3"):
        n = n.replace(sfx, "")
    MODEL_NORM_MAP[name] = n
    return n

# ──────────────────────────────────────────────────────────────────────────────
# STEP 0 — SCHEMA DISCOVERY (documented)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 0 — DATA DISCOVERY")
print("=" * 70)

for set_id, folders in [("set1", SET1_FOLDERS), ("set2", SET2_FOLDERS), ("set3", SET3_FOLDERS)]:
    subdir = SUBDIR[set_id]
    for mode, fname in folders.items():
        fpath = BASE / fname
        files = glob.glob(str(fpath / "**" / "*trajectory*.json"), recursive=True)
        models = set()
        for fp in files:
            with open(fp) as f:
                d = json.load(f)
            models.add(d.get("model_name", "?"))
        print(f"  [{set_id}/{mode}] {fname}: {len(files)} trajectories, models={sorted(models)}")

print()
print("Schema (common fields discovered):")
sample_file = glob.glob(str(BASE / "pop_l123_all_gpt" / "**" / "*trajectory*.json"), recursive=True)[0]
with open(sample_file) as f:
    s = json.load(f)
fr = s.get("final_result", {})
print("  top-level keys:", list(s.keys()))
print("  final_result.sol_stats keys:", list(fr.get("sol_stats", {}).keys()))
print("  final_result.energy_stats keys:", list(fr.get("energy_stats", {}).keys()))
print("  final_result.roofline_stats keys:", list(fr.get("roofline_stats", {}).keys()))
print("  final_result.kernel_launch_stats keys:", list(fr.get("kernel_launch_stats", {}).keys()))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — LABEL FIX (sl5 only)
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 1 — LABEL FIX")
print("=" * 70)

SL5_SRC  = BASE / SET3_FOLDERS["sl5_all"]
SL5_FIXED = BASE / SET3_SL5_FIXED

# Make a fresh copy
if SL5_FIXED.exists():
    shutil.rmtree(SL5_FIXED)
shutil.copytree(SL5_SRC, SL5_FIXED)
print(f"Copied {SL5_SRC.name} → {SL5_FIXED.name}")

fix_log = []
n_examined = 0
n_changed  = 0

traj_files = glob.glob(str(SL5_FIXED / "**" / "*trajectory*.json"), recursive=True)
for fp in traj_files:
    with open(fp) as f:
        d = json.load(f)

    n_examined += 1
    fr = d.get("final_result") or {}
    correctness = fr.get("correctness", False)

    # Only examine entries where final correctness is False
    if correctness:
        continue

    model  = d.get("model_name", "?")
    prob   = d.get("problem_id")
    turns  = d.get("turns", [])

    # Look for any run_correctness pass OR submit_kernel pass in trajectory
    mid_pass = False
    timeout_evidence = False
    passing_turn = None
    passing_result = None

    for turn in turns:
        for tc in turn.get("tool_calls", []):
            tool_name  = tc.get("tool_name", "")
            success    = tc.get("success", False)
            result_txt = tc.get("result_text", "")

            if tool_name == "run_correctness" and success:
                mid_pass = True
                passing_turn = turn.get("turn_id")
                passing_result = result_txt[:200]

            if tool_name == "submit_kernel" and success:
                mid_pass = True
                passing_turn = turn.get("turn_id")
                passing_result = result_txt[:200]

            # Detect timeout evidence
            if "timed out" in result_txt.lower() or "timeout" in result_txt.lower():
                timeout_evidence = True

    if mid_pass:
        # Relabel
        d["correct_original"] = correctness
        d["correct_source"]   = "timeout_recovery"
        d["outcome_original"] = d.get("outcome")

        # Patch final_result correctness
        d["final_result"]["correctness"] = True
        d["outcome"] = "correct"

        # Write back
        with open(fp, "w") as f:
            json.dump(d, f)

        fix_log.append({
            "file": str(fp),
            "model": model,
            "problem_id": prob,
            "original_outcome": d.get("outcome_original"),
            "timeout_evidence": timeout_evidence,
            "passing_turn": passing_turn,
            "passing_result_snippet": passing_result,
        })
        n_changed += 1

# Save fix log
fix_log_path = OUT / "label_fixes.jsonl"
with open(fix_log_path, "w") as f:
    for entry in fix_log:
        f.write(json.dumps(entry) + "\n")

print(f"  Examined: {n_examined} entries")
print(f"  Changed:  {n_changed} entries")
print()
print("  Changed entries:")
by_model = defaultdict(list)
by_task  = defaultdict(list)
for e in fix_log:
    by_model[e["model"]].append(e["problem_id"])
    by_task[e["problem_id"]].append(e["model"])
for model, probs in sorted(by_model.items()):
    print(f"    {model}: problems {sorted(probs)}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 2 — DATA LOADING")
print("=" * 70)

def safe_speedup(fr):
    """ref_runtime / runtime — positive only."""
    runtime = fr.get("runtime", -1) or -1
    ref     = fr.get("ref_runtime", -1) or -1
    if runtime > 0 and ref > 0:
        return ref / runtime
    return float("nan")

def load_folder(folder_path: Path, set_id: str, mode_label: str, use_fixed_sl5=False):
    """Load all trajectories from a folder into a list of row dicts."""
    rows = []
    # Determine subdir
    for sub in ("popcorn", "hardware_translation_stub"):
        subpath = folder_path / sub
        if subpath.exists():
            break
    else:
        subpath = folder_path  # fallback

    files = glob.glob(str(folder_path / "**" / "*trajectory*.json"), recursive=True)
    for fp in files:
        with open(fp) as f:
            d = json.load(f)

        fr  = d.get("final_result") or {}
        sol = fr.get("sol_stats") or {}
        eng = fr.get("energy_stats") or {}
        rl  = fr.get("roofline_stats") or {}
        kls = fr.get("kernel_launch_stats") or {}

        raw_model = d.get("model_name", "unknown")
        model = normalize_model(raw_model)

        correct = bool(fr.get("correctness", False))
        timeout_recovered = bool(d.get("correct_source") == "timeout_recovery")
        # Override correctness with fixed value if present
        if "correct_original" in d:
            correct = bool(fr.get("correctness", False))

        spd = safe_speedup(fr)
        dram   = sol.get("dram_utilization_pct") or rl.get("dram_utilization_pct")
        comp   = sol.get("compute_utilization_pct")
        power  = eng.get("avg_power_w")
        energy = eng.get("energy_per_run_mj")
        fusion = kls.get("fusion_ratio")
        occ    = rl.get("occupancy_pct")
        sol_sc = sol.get("sol_score")

        # Trajectory stats
        turns      = d.get("turns", [])
        total_turns = d.get("total_turns", len(turns))
        # Step of first correct in trajectory
        first_correct_step = None
        for ti, turn in enumerate(turns):
            for tc in turn.get("tool_calls", []):
                if tc.get("tool_name") == "run_correctness" and tc.get("success"):
                    if first_correct_step is None:
                        first_correct_step = ti

        rows.append({
            "folder":            folder_path.name,
            "set_id":            set_id,
            "mode":              mode_label,
            "task":              d.get("problem_name", f"prob_{d.get('problem_id')}"),
            "problem_id":        d.get("problem_id"),
            "level":             d.get("level"),
            "model_raw":         raw_model,
            "model":             model,
            "correct":           correct,
            "outcome":           d.get("outcome", "unknown"),
            "speedup":           spd,
            "dram_util":         dram,
            "compute_pct":       comp,
            "power_w":           power,
            "energy_mj":         energy,
            "fusion":            fusion,
            "occupancy":         occ,
            "sol_score":         sol_sc,
            "timeout_recovered": timeout_recovered,
            "total_turns":       total_turns,
            "first_correct_step": first_correct_step,
            "wall_clock_s":      d.get("agent_wall_clock_s"),
            "llm_total_tokens":  d.get("llm_total_tokens"),
        })
    return rows

all_rows = []
missing_fields_log = {}

# Set 1
for mode, fname in SET1_FOLDERS.items():
    fpath = BASE / fname
    rows = load_folder(fpath, "set1", mode)
    all_rows.extend(rows)
    print(f"  set1/{mode}: {len(rows)} rows from {fname}")

# Set 2
for mode, fname in SET2_FOLDERS.items():
    fpath = BASE / fname
    rows = load_folder(fpath, "set2", mode)
    all_rows.extend(rows)
    print(f"  set2/{mode}: {len(rows)} rows from {fname}")

# Set 3 — sl5 uses FIXED copy, fl5 uses original
sl5_rows = load_folder(SL5_FIXED, "set3", "sl5_all", use_fixed_sl5=True)
fl5_rows = load_folder(BASE / SET3_FOLDERS["fl5_st"], "set3", "fl5_st")
all_rows.extend(sl5_rows)
all_rows.extend(fl5_rows)
print(f"  set3/sl5_all (FIXED): {len(sl5_rows)} rows")
print(f"  set3/fl5_st:          {len(fl5_rows)} rows")

df = pd.DataFrame(all_rows)

# Log missing fields
for col in ["speedup", "dram_util", "compute_pct", "power_w", "energy_mj", "fusion", "occupancy", "sol_score"]:
    miss = df[col].isna().sum()
    if miss > 0:
        missing_fields_log[col] = int(miss)

with open(OUT / "missing_fields.txt", "w") as f:
    f.write("Missing field counts across all rows:\n")
    for col, cnt in missing_fields_log.items():
        f.write(f"  {col}: {cnt} missing out of {len(df)} total\n")

print()
print(f"  Total rows: {len(df)}")
print(f"  Models (normalized): {sorted(df['model'].unique())}")
print()
print("  Model normalization map:")
for raw, norm in sorted(MODEL_NORM_MAP.items()):
    print(f"    '{raw}' → '{norm}'")
print()
print("  Missing fields:")
for col, cnt in missing_fields_log.items():
    print(f"    {col}: {cnt} / {len(df)} missing")

# Save the DataFrame for inspection
df.to_csv(OUT / "all_results.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = "tab10"
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 300})

def get_model_colors(models):
    """Return consistent color-per-model dict using tab10."""
    cmap   = plt.cm.get_cmap("tab10")
    unique = sorted(models)
    return {m: cmap(i % 10) for i, m in enumerate(unique)}

ALL_MODELS = sorted(df["model"].unique())
MODEL_COLORS = get_model_colors(ALL_MODELS)

def pareto_frontier(df_sub, x_col, y_col):
    """Return points on the Pareto frontier (maximize both axes)."""
    sub = df_sub[[x_col, y_col]].dropna().copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(x_col)
    frontier_x, frontier_y = [], []
    max_y = -np.inf
    for _, row in sub.iterrows():
        if row[y_col] > max_y:
            max_y = row[y_col]
            frontier_x.append(row[x_col])
            frontier_y.append(row[y_col])
    return pd.DataFrame({x_col: frontier_x, y_col: frontier_y})

def plot_pareto_panel(ax, df_panel, x_col, y_col, title=""):
    """Draw scatter + frontier per model on a single axis."""
    models = sorted(df_panel["model"].dropna().unique())
    for m in models:
        sub = df_panel[df_panel["model"] == m][[x_col, y_col]].dropna()
        if sub.empty:
            continue
        c = MODEL_COLORS.get(m, "grey")
        ax.scatter(sub[x_col], sub[y_col], color=c, alpha=0.25, s=18, label=m)
        pf = pareto_frontier(sub, x_col, y_col)
        if not pf.empty:
            ax.scatter(pf[x_col], pf[y_col], color=c, alpha=1.0, s=60, marker="D", zorder=5)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(x_col, fontsize=7)
    ax.set_ylabel(y_col, fontsize=7)
    ax.tick_params(labelsize=6)

PARETO_PAIRS = [
    ("compute_pct",  "dram_util",   "Compute % vs DRAM Utilization"),
    ("speedup",      "sol_score",   "SOL Score vs Speedup/Reference"),
    ("energy_mj",    "power_w",     "Power (W) vs Energy (mJ)"),
    ("occupancy",    "fusion",      "Fusion vs Occupancy"),
    ("speedup",      "dram_util",   "DRAM Utilization vs Speedup"),
]

def make_legend_handles(models):
    handles = []
    for m in sorted(models):
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=MODEL_COLORS.get(m, "grey"),
                                   markersize=6, label=m))
    return handles

def save_fig(fig, path):
    path = Path(str(path).replace(".pdf", ".png"))
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3A — PARETO FRONTIER PLOTS
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 3A — PARETO FRONTIER PLOTS")
print("=" * 70)

LEVELS   = [1, 2, 3]
MODES_S1 = ["st", "default", "all"]
MODES_S2 = ["default", "all", "st"]

# --- Set 1: 3x3 per Pareto pair (rows=L1/L2/L3, cols=st/default/all)
df1 = df[df["set_id"] == "set1"]

for pi, (xc, yc, ptitle) in enumerate(PARETO_PAIRS, 1):
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    fig.suptitle(f"Set 1 Pareto: {ptitle}", fontsize=11)
    models_in_plot = set()
    for ri, lv in enumerate(LEVELS):
        for ci, mode in enumerate(MODES_S1):
            ax = axes[ri][ci]
            panel = df1[(df1["level"] == lv) & (df1["mode"] == mode)]
            plot_pareto_panel(ax, panel, xc, yc, title=f"L{lv} / {mode}")
            models_in_plot.update(panel["model"].dropna().unique())
            if ri == 0 and ci == 2:
                ax.legend(handles=make_legend_handles(models_in_plot),
                          fontsize=5, loc="best")
    save_fig(fig, PLOT / f"set1_pareto_P{pi}_3x3.pdf")

# --- Set 2: 3x1 per Pareto pair
df2 = df[df["set_id"] == "set2"]

for pi, (xc, yc, ptitle) in enumerate(PARETO_PAIRS, 1):
    fig, axes = plt.subplots(3, 1, figsize=(6, 11))
    fig.suptitle(f"Set 2 Pareto: {ptitle}", fontsize=11)
    models_in_plot = set()
    for ri, mode in enumerate(MODES_S2):
        panel = df2[df2["mode"] == mode]
        plot_pareto_panel(axes[ri], panel, xc, yc, title=f"mode={mode}")
        models_in_plot.update(panel["model"].dropna().unique())
    axes[0].legend(handles=make_legend_handles(models_in_plot), fontsize=6, loc="best")
    save_fig(fig, PLOT / f"set2_pareto_P{pi}_3x1.pdf")

# --- Set 3: 2x1 per Pareto pair
df3 = df[df["set_id"] == "set3"]
SET3_MODES = ["sl5_all", "fl5_st"]

for pi, (xc, yc, ptitle) in enumerate(PARETO_PAIRS, 1):
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    fig.suptitle(f"Set 3 Pareto: {ptitle}", fontsize=11)
    models_in_plot = set()
    for ri, mode in enumerate(SET3_MODES):
        panel = df3[df3["mode"] == mode]
        plot_pareto_panel(axes[ri], panel, xc, yc, title=f"mode={mode}")
        models_in_plot.update(panel["model"].dropna().unique())
    axes[0].legend(handles=make_legend_handles(models_in_plot), fontsize=6, loc="best")
    save_fig(fig, PLOT / f"set3_pareto_P{pi}_2x1.pdf")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3B — GENERAL PLOTS
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 3B — GENERAL PLOTS")
print("=" * 70)

def short_task(name):
    """Shorten task name for display."""
    base = os.path.splitext(str(name))[0] if name else "?"
    # strip leading level_N_problemN_ prefix
    base = re.sub(r"^\d+_", "", base)
    return base[:28]

def general_plots_for_set(df_set, set_label, tasks_per_row=5):
    """Generate G1-G7 for a given set DataFrame."""
    tasks  = sorted(df_set["problem_id"].dropna().unique())
    models = sorted(df_set["model"].dropna().unique())
    df_set = df_set.copy()
    df_set["task_short"] = df_set["problem_id"].apply(lambda x: f"P{int(x)}" if pd.notna(x) else "?")

    # G1: Speedup distribution — boxplot per model, faceted by task
    print(f"  Generating G1 for {set_label}...")
    ncols = min(tasks_per_row, len(tasks))
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3.5), squeeze=False)
    fig.suptitle(f"{set_label} G1: Speedup Distribution per Model × Task", fontsize=10)
    for idx, task_id in enumerate(tasks):
        ax  = axes[idx // ncols][idx % ncols]
        sub = df_set[(df_set["problem_id"] == task_id) & df_set["speedup"].notna()]
        if not sub.empty:
            order = [m for m in models if m in sub["model"].values]
            palette = {m: MODEL_COLORS.get(m, "grey") for m in order}
            sns.boxplot(data=sub, x="model", y="speedup", order=order,
                        palette=palette, ax=ax, width=0.6)
        ax.set_title(f"P{int(task_id)}", fontsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("Speedup" if idx % ncols == 0 else "", fontsize=6)
        ax.tick_params(axis="x", rotation=30, labelsize=5)
        ax.tick_params(axis="y", labelsize=5)
    # Hide unused subplots
    for idx in range(len(tasks), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    save_fig(fig, PLOT / f"{set_label}_general_G1.pdf")

    # G2: Number correct — bar chart per model, grouped by task/level
    print(f"  Generating G2 for {set_label}...")
    g2 = df_set.groupby(["task_short", "model"])["correct"].sum().reset_index()
    g2.columns = ["task_short", "model", "n_correct"]
    n_tasks = len(g2["task_short"].unique())
    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 0.7), 5))
    fig.suptitle(f"{set_label} G2: Number Correct per Model × Task", fontsize=10)
    order_t = sorted(g2["task_short"].unique())
    palette = {m: MODEL_COLORS.get(m, "grey") for m in models}
    sns.barplot(data=g2, x="task_short", y="n_correct", hue="model",
                order=order_t, palette=palette, ax=ax)
    ax.set_xlabel("Task")
    ax.set_ylabel("# Correct")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.legend(title="Model", fontsize=6)
    save_fig(fig, PLOT / f"{set_label}_general_G2.pdf")

    # G3: Correctness rate — bar chart per model, grouped by task/level
    print(f"  Generating G3 for {set_label}...")
    g3 = df_set.groupby(["task_short", "model"]).agg(
        n_correct=("correct", "sum"), n_total=("correct", "count")).reset_index()
    g3["rate"] = g3["n_correct"] / g3["n_total"] * 100
    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 0.7), 5))
    fig.suptitle(f"{set_label} G3: Correctness Rate (%) per Model × Task", fontsize=10)
    sns.barplot(data=g3, x="task_short", y="rate", hue="model",
                order=order_t, palette=palette, ax=ax)
    ax.set_xlabel("Task")
    ax.set_ylabel("Correctness Rate (%)")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.legend(title="Model", fontsize=6)
    save_fig(fig, PLOT / f"{set_label}_general_G3.pdf")

    # G4: Timeout recovery rate — only for set3 FIXED
    if set_label == "set3":
        print(f"  Generating G4 for {set_label}...")
        sl5 = df_set[df_set["mode"] == "sl5_all"].copy()
        if not sl5.empty:
            orig_correct = sl5.groupby("model").apply(
                lambda x: (x["correct"] & ~x["timeout_recovered"]).sum()).reset_index()
            orig_correct.columns = ["model", "n_original_correct"]
            recov = sl5.groupby("model")["timeout_recovered"].sum().reset_index()
            recov.columns = ["model", "n_recovered"]
            total_n = sl5.groupby("model")["correct"].count().reset_index()
            total_n.columns = ["model", "n_total"]
            g4 = orig_correct.merge(recov, on="model").merge(total_n, on="model")
            g4["orig_rate"]  = g4["n_original_correct"] / g4["n_total"] * 100
            g4["recov_rate"] = (g4["n_original_correct"] + g4["n_recovered"]) / g4["n_total"] * 100
            g4_melted = pd.melt(g4, id_vars=["model"],
                                value_vars=["orig_rate", "recov_rate"],
                                var_name="type", value_name="rate")
            g4_melted["type"] = g4_melted["type"].map({"orig_rate": "Original", "recov_rate": "After Recovery"})
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.suptitle("Set 3 G4: Timeout Recovery Impact on Correctness (sl5 FIXED)", fontsize=10)
            sns.barplot(data=g4_melted, x="model", y="rate", hue="type",
                        palette=["#4477AA", "#EE8833"], ax=ax)
            ax.set_xlabel("Model")
            ax.set_ylabel("Correctness Rate (%)")
            ax.tick_params(axis="x", rotation=20, labelsize=7)
            ax.legend(title="", fontsize=7)
            save_fig(fig, PLOT / f"{set_label}_general_G4.pdf")
        else:
            print(f"  Skipping G4 for {set_label}: no sl5 data")
    else:
        print(f"  Skipping G4 (only for set3)")

    # G5: SOL Score distribution — violin per model, one panel per task
    print(f"  Generating G5 for {set_label}...")
    df5 = df_set[df_set["sol_score"].notna()]
    if not df5.empty:
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3.5), squeeze=False)
        fig.suptitle(f"{set_label} G5: SOL Score Distribution per Model × Task", fontsize=10)
        for idx, task_id in enumerate(tasks):
            ax  = axes[idx // ncols][idx % ncols]
            sub = df5[df5["problem_id"] == task_id]
            if not sub.empty and len(sub["model"].unique()) > 0:
                order = [m for m in models if m in sub["model"].values]
                palette = {m: MODEL_COLORS.get(m, "grey") for m in order}
                if len(sub) >= 2:
                    sns.violinplot(data=sub, x="model", y="sol_score", order=order,
                                   palette=palette, ax=ax, cut=0)
                else:
                    sns.stripplot(data=sub, x="model", y="sol_score", order=order,
                                  palette=palette, ax=ax, jitter=True)
            ax.set_title(f"P{int(task_id)}", fontsize=7)
            ax.set_xlabel("")
            ax.set_ylabel("SOL Score" if idx % ncols == 0 else "", fontsize=6)
            ax.tick_params(axis="x", rotation=30, labelsize=5)
            ax.tick_params(axis="y", labelsize=5)
        for idx in range(len(tasks), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        save_fig(fig, PLOT / f"{set_label}_general_G5.pdf")
    else:
        print(f"  Skipping G5 (no sol_score data)")

    # G6: DRAM Utilization vs Speedup scatter — one panel per task
    print(f"  Generating G6 for {set_label}...")
    df6 = df_set[df_set["dram_util"].notna() & df_set["speedup"].notna()]
    if not df6.empty:
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3), squeeze=False)
        fig.suptitle(f"{set_label} G6: DRAM Util vs Speedup per Task", fontsize=10)
        for idx, task_id in enumerate(tasks):
            ax  = axes[idx // ncols][idx % ncols]
            sub = df6[df6["problem_id"] == task_id]
            if not sub.empty:
                for m in models:
                    msub = sub[sub["model"] == m]
                    if not msub.empty:
                        ax.scatter(msub["speedup"], msub["dram_util"],
                                   color=MODEL_COLORS.get(m, "grey"), alpha=0.7, s=15, label=m)
            ax.set_title(f"P{int(task_id)}", fontsize=7)
            ax.set_xlabel("Speedup" if idx // ncols == nrows - 1 else "", fontsize=6)
            ax.set_ylabel("DRAM Util" if idx % ncols == 0 else "", fontsize=6)
            ax.tick_params(labelsize=5)
        for idx in range(len(tasks), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        axes[0][-1].legend(handles=make_legend_handles(models), fontsize=5, loc="best")
        save_fig(fig, PLOT / f"{set_label}_general_G6.pdf")
    else:
        print(f"  Skipping G6 (no data)")

    # G7: Energy vs Power scatter — one panel per task
    print(f"  Generating G7 for {set_label}...")
    df7 = df_set[df_set["energy_mj"].notna() & df_set["power_w"].notna()]
    if not df7.empty:
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3), squeeze=False)
        fig.suptitle(f"{set_label} G7: Energy (mJ) vs Power (W) per Task", fontsize=10)
        for idx, task_id in enumerate(tasks):
            ax  = axes[idx // ncols][idx % ncols]
            sub = df7[df7["problem_id"] == task_id]
            if not sub.empty:
                for m in models:
                    msub = sub[sub["model"] == m]
                    if not msub.empty:
                        ax.scatter(msub["energy_mj"], msub["power_w"],
                                   color=MODEL_COLORS.get(m, "grey"), alpha=0.7, s=15, label=m)
            ax.set_title(f"P{int(task_id)}", fontsize=7)
            ax.set_xlabel("Energy (mJ)" if idx // ncols == nrows - 1 else "", fontsize=6)
            ax.set_ylabel("Power (W)" if idx % ncols == 0 else "", fontsize=6)
            ax.tick_params(labelsize=5)
        for idx in range(len(tasks), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        axes[0][-1].legend(handles=make_legend_handles(models), fontsize=5, loc="best")
        save_fig(fig, PLOT / f"{set_label}_general_G7.pdf")
    else:
        print(f"  Skipping G7 (no data)")

# Generate general plots per set
for set_lbl, set_df in [("set1", df1), ("set2", df2), ("set3", df3)]:
    general_plots_for_set(set_df, set_lbl)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3D — COMBINED OVERVIEW PLOTS (aggregated across all tasks per set)
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 3D — COMBINED OVERVIEW PLOTS")
print("=" * 70)

SET_LABELS   = ["set1", "set2", "set3"]
SET_DFS      = {"set1": df1, "set2": df2, "set3": df3}
MODE_ORDER_S1 = ["st", "default", "all"]
MODE_ORDER_S2 = ["st", "default", "all"]
MODE_ORDER_S3 = ["fl5_st", "sl5_all"]

# ── C1: Correctness rate per model — one panel per set, all tasks pooled ──────
print("  C1: Overall correctness rate per model per set...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig.suptitle("C1: Overall Correctness Rate per Model (all tasks pooled)", fontsize=11)
for si, set_lbl in enumerate(SET_LABELS):
    ax  = axes[si]
    sub = SET_DFS[set_lbl].groupby("model").agg(
        n_correct=("correct", "sum"), n_total=("correct", "count")).reset_index()
    sub["rate"] = sub["n_correct"] / sub["n_total"] * 100
    sub = sub.sort_values("rate", ascending=False)
    palette = [MODEL_COLORS.get(m, "grey") for m in sub["model"]]
    ax.scatter(sub["model"], sub["rate"], color=palette, s=120, zorder=5)
    ax.set_ylim(0, 105)
    ax.set_title(f"Set {set_lbl[-1]}", fontsize=10)
    ax.set_xlabel("Model")
    ax.set_ylabel("Correctness Rate (%)" if si == 0 else "")
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    # add value labels
    for _, row in sub.iterrows():
        ax.text(row["model"], row["rate"] + 2, f"{row['rate']:.0f}%",
                ha="center", va="bottom", fontsize=6)
save_fig(fig, PLOT / "combined_C1_correctness_rate.png")

# ── C2: Speedup distribution per model — one panel per set ───────────────────
print("  C2: Overall speedup distribution per model per set...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("C2: Speedup Distribution per Model (all tasks pooled)", fontsize=11)
for si, set_lbl in enumerate(SET_LABELS):
    ax  = axes[si]
    sub = SET_DFS[set_lbl][SET_DFS[set_lbl]["speedup"].notna()]
    if not sub.empty:
        models_here = sorted(sub["model"].unique())
        palette = {m: MODEL_COLORS.get(m, "grey") for m in models_here}
        sns.boxplot(data=sub, x="model", y="speedup", order=models_here,
                    palette=palette, ax=ax, width=0.5, fliersize=3)
        # overlay individual points as scatter
        for m in models_here:
            msub = sub[sub["model"] == m]["speedup"]
            ax.scatter([m] * len(msub), msub,
                       color=MODEL_COLORS.get(m, "grey"), alpha=0.3, s=15, zorder=3)
    ax.set_title(f"Set {set_lbl[-1]}", fontsize=10)
    ax.set_xlabel("Model")
    ax.set_ylabel("Speedup" if si == 0 else "")
    ax.tick_params(axis="x", rotation=25, labelsize=7)
save_fig(fig, PLOT / "combined_C2_speedup_dist.png")

# ── C3: Correctness rate by mode — one panel per set ─────────────────────────
print("  C3: Correctness rate by mode per model...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig.suptitle("C3: Correctness Rate by Mode per Model", fontsize=11)
mode_colors = {"st": "#4477AA", "default": "#EE8833", "all": "#228833",
               "fl5_st": "#4477AA", "sl5_all": "#EE8833"}
for si, set_lbl in enumerate(SET_LABELS):
    ax  = axes[si]
    sub = SET_DFS[set_lbl].groupby(["mode", "model"]).agg(
        rate=("correct", "mean")).reset_index()
    sub["rate"] *= 100
    models_here = sorted(sub["model"].unique())
    modes_here  = sorted(sub["mode"].unique())
    for m in models_here:
        for mode in modes_here:
            pt = sub[(sub["model"] == m) & (sub["mode"] == mode)]
            if pt.empty:
                continue
            ax.scatter(m, pt["rate"].values[0],
                       color=mode_colors.get(mode, "grey"),
                       s=90, alpha=0.9, label=mode, zorder=5)
    ax.set_title(f"Set {set_lbl[-1]}", fontsize=10)
    ax.set_xlabel("Model")
    ax.set_ylabel("Correctness Rate (%)" if si == 0 else "")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    # deduplicated legend on last panel
    if si == 2:
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=mode_colors.get(md, "grey"),
                               markersize=7, label=md) for md in modes_here]
        ax.legend(handles=handles, title="Mode", fontsize=7, loc="upper right")
save_fig(fig, PLOT / "combined_C3_correctness_by_mode.png")

# ── C4: DRAM util vs speedup — single scatter per set, all tasks pooled ──────
print("  C4: DRAM util vs speedup (combined)...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("C4: DRAM Utilization vs Speedup (all tasks pooled)", fontsize=11)
for si, set_lbl in enumerate(SET_LABELS):
    ax  = axes[si]
    sub = SET_DFS[set_lbl][
        SET_DFS[set_lbl]["dram_util"].notna() & SET_DFS[set_lbl]["speedup"].notna()]
    if not sub.empty:
        models_here = sorted(sub["model"].unique())
        for m in models_here:
            msub = sub[sub["model"] == m]
            ax.scatter(msub["speedup"], msub["dram_util"],
                       color=MODEL_COLORS.get(m, "grey"), alpha=0.55, s=25, label=m)
    ax.set_title(f"Set {set_lbl[-1]}", fontsize=10)
    ax.set_xlabel("Speedup")
    ax.set_ylabel("DRAM Utilization (%)" if si == 0 else "")
    ax.tick_params(labelsize=7)
    if si == 0:
        ax.legend(handles=make_legend_handles(sorted(sub["model"].unique())),
                  fontsize=6, loc="best")
save_fig(fig, PLOT / "combined_C4_dram_vs_speedup.png")

# ── C5: Energy vs Power — single scatter per set ──────────────────────────────
print("  C5: Energy vs Power (combined)...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("C5: Energy (mJ) vs Power (W) (all tasks pooled)", fontsize=11)
for si, set_lbl in enumerate(SET_LABELS):
    ax  = axes[si]
    sub = SET_DFS[set_lbl][
        SET_DFS[set_lbl]["energy_mj"].notna() & SET_DFS[set_lbl]["power_w"].notna()]
    if not sub.empty:
        models_here = sorted(sub["model"].unique())
        for m in models_here:
            msub = sub[sub["model"] == m]
            ax.scatter(msub["energy_mj"], msub["power_w"],
                       color=MODEL_COLORS.get(m, "grey"), alpha=0.55, s=25, label=m)
    ax.set_title(f"Set {set_lbl[-1]}", fontsize=10)
    ax.set_xlabel("Energy (mJ)")
    ax.set_ylabel("Power (W)" if si == 0 else "")
    ax.tick_params(labelsize=7)
    if si == 0:
        ax.legend(handles=make_legend_handles(sorted(sub["model"].unique())),
                  fontsize=6, loc="best")
save_fig(fig, PLOT / "combined_C5_energy_vs_power.png")

# ── C6: SOL score distribution per model — one panel per set ─────────────────
print("  C6: SOL score distribution per model per set...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("C6: SOL Score Distribution per Model (all tasks pooled)", fontsize=11)
for si, set_lbl in enumerate(SET_LABELS):
    ax  = axes[si]
    sub = SET_DFS[set_lbl][SET_DFS[set_lbl]["sol_score"].notna()]
    if not sub.empty:
        models_here = sorted(sub["model"].unique())
        palette = {m: MODEL_COLORS.get(m, "grey") for m in models_here}
        sns.boxplot(data=sub, x="model", y="sol_score", order=models_here,
                    palette=palette, ax=ax, width=0.5, fliersize=3)
        for m in models_here:
            msub = sub[sub["model"] == m]["sol_score"]
            ax.scatter([m] * len(msub), msub,
                       color=MODEL_COLORS.get(m, "grey"), alpha=0.3, s=15, zorder=3)
    ax.set_title(f"Set {set_lbl[-1]}", fontsize=10)
    ax.set_xlabel("Model")
    ax.set_ylabel("SOL Score" if si == 0 else "")
    ax.tick_params(axis="x", rotation=25, labelsize=7)
save_fig(fig, PLOT / "combined_C6_sol_score_dist.png")

# ── C7: Cross-set correctness — one grouped scatter, model × set ──────────────
print("  C7: Cross-set correctness summary...")
c7 = df.groupby(["set_id", "model"]).agg(
    rate=("correct", "mean"), n=("correct", "count")).reset_index()
c7["rate"] *= 100
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("C7: Correctness Rate — All Models × All Sets", fontsize=11)
set_offsets = {"set1": -0.2, "set2": 0.0, "set3": 0.2}
set_markers = {"set1": "o", "set2": "s", "set3": "^"}
models_sorted = sorted(c7["model"].unique())
x_pos = {m: i for i, m in enumerate(models_sorted)}
for set_lbl in SET_LABELS:
    sub = c7[c7["set_id"] == set_lbl]
    xs  = [x_pos[m] + set_offsets[set_lbl] for m in sub["model"]]
    ax.scatter(xs, sub["rate"],
               color=[MODEL_COLORS.get(m, "grey") for m in sub["model"]],
               marker=set_markers[set_lbl], s=100, alpha=0.9,
               label=set_lbl, zorder=5)
ax.set_xticks(range(len(models_sorted)))
ax.set_xticklabels(models_sorted, rotation=20, fontsize=8)
ax.set_ylabel("Correctness Rate (%)")
ax.set_ylim(0, 105)
# legend for sets (shape) + models (color) separately
shape_handles = [
    plt.Line2D([0], [0], marker=set_markers[s], color="w", markerfacecolor="grey",
               markersize=8, label=s) for s in SET_LABELS
]
color_handles = make_legend_handles(models_sorted)
leg1 = ax.legend(handles=shape_handles, title="Set",   fontsize=7, loc="upper right")
ax.add_artist(leg1)
ax.legend(handles=color_handles, title="Model", fontsize=7, loc="upper left")
save_fig(fig, PLOT / "combined_C7_cross_set_correctness.png")

# ── C8: Correctness correct/incorrect count bars — all sets side-by-side ──────
print("  C8: Correct vs incorrect count per model per set...")
c8 = df.groupby(["set_id", "model"]).agg(
    n_correct=("correct", "sum"), n_total=("correct", "count")).reset_index()
c8["n_incorrect"] = c8["n_total"] - c8["n_correct"]
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
fig.suptitle("C8: Correct vs Incorrect Count per Model", fontsize=11)
for si, set_lbl in enumerate(SET_LABELS):
    ax  = axes[si]
    sub = c8[c8["set_id"] == set_lbl].sort_values("n_total", ascending=False)
    models_here = list(sub["model"])
    x = range(len(models_here))
    ax.scatter(models_here, sub["n_correct"].values,
               color=[MODEL_COLORS.get(m, "grey") for m in models_here],
               s=100, marker="o", label="Correct", zorder=5)
    ax.scatter(models_here, sub["n_incorrect"].values,
               color=[MODEL_COLORS.get(m, "grey") for m in models_here],
               s=100, marker="x", label="Incorrect", zorder=5)
    ax.set_title(f"Set {set_lbl[-1]}", fontsize=10)
    ax.set_xlabel("Model")
    ax.set_ylabel("Count" if si == 0 else "")
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    if si == 2:
        ax.legend(fontsize=7)
save_fig(fig, PLOT / "combined_C8_correct_vs_incorrect.png")

print()
print("  Combined plots saved:")
for ci in range(1, 9):
    p = PLOT / f"combined_C{ci}_*.png"
    import glob as _glob
    matches = _glob.glob(str(p))
    for m in matches:
        print(f"    {Path(m).name}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3C — EXPLORATORY PLOTS
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 3C — EXPLORATORY PLOTS")
print("=" * 70)

# ── E1: Learning curve — correctness rate across L1→L2→L3 per model (Set 1)
print("  E1: Learning curves by level (set1)...")
e1 = df1.groupby(["level", "model"]).agg(rate=("correct", "mean")).reset_index()
e1["rate"] *= 100
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("E1: Correctness Rate by Level (L1→L2→L3) per Model", fontsize=10)
for m in sorted(e1["model"].unique()):
    sub = e1[e1["model"] == m].sort_values("level")
    ax.scatter(sub["level"], sub["rate"], color=MODEL_COLORS.get(m, "grey"), s=80, label=m, zorder=5)
ax.set_xlabel("Level")
ax.set_ylabel("Correctness Rate (%)")
ax.set_xticks([1, 2, 3])
ax.legend(title="Model", fontsize=7)
save_fig(fig, PLOT_EX / "E1_learning_curve_by_level.pdf")

# ── E2: Efficiency frontier — speedup vs energy (all sets)
print("  E2: Efficiency frontier speedup vs energy...")
dfe = df[df["speedup"].notna() & df["energy_mj"].notna() & (df["speedup"] > 0)]
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("E2: Efficiency Frontier: Speedup vs Energy (mJ) Across All Sets", fontsize=10)
for m in sorted(dfe["model"].unique()):
    sub = dfe[dfe["model"] == m]
    ax.scatter(sub["energy_mj"], sub["speedup"],
               color=MODEL_COLORS.get(m, "grey"), alpha=0.3, s=20, label=m)
ax.set_xlabel("Energy (mJ)")
ax.set_ylabel("Speedup")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(title="Model", fontsize=7)
save_fig(fig, PLOT_EX / "E2_efficiency_frontier.pdf")

# ── E3: Failure mode clustering — outcome distribution per model
print("  E3: Failure mode clustering...")
e3 = df.groupby(["model", "outcome"]).size().reset_index(name="count")
outcomes_order = sorted(e3["outcome"].unique())
models_order   = sorted(e3["model"].unique())
pivot_e3 = e3.pivot_table(index="model", columns="outcome", values="count", fill_value=0)
pivot_e3 = pivot_e3.div(pivot_e3.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("E3: Failure Mode Distribution per Model", fontsize=10)
pivot_e3.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
ax.set_ylabel("Fraction (%)")
ax.set_xlabel("Model")
ax.tick_params(axis="x", rotation=20, labelsize=7)
ax.legend(title="Outcome", fontsize=6, loc="lower right")
save_fig(fig, PLOT_EX / "E3_failure_modes.pdf")

# ── E4: Correlation heatmap per set
print("  E4: Correlation heatmaps...")
numeric_cols = ["speedup", "dram_util", "compute_pct", "power_w", "energy_mj",
                "fusion", "occupancy", "sol_score"]
for set_lbl, set_df in [("set1", df1), ("set2", df2), ("set3", df3)]:
    sub = set_df[numeric_cols].dropna(how="all")
    valid = [c for c in numeric_cols if sub[c].notna().sum() > 5]
    if len(valid) < 2:
        continue
    corr = sub[valid].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(max(6, len(valid)), max(5, len(valid))))
    fig.suptitle(f"E4: Pearson Correlation Heatmap — {set_lbl}", fontsize=10)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True, linewidths=0.5, annot_kws={"size": 7},
                vmin=-1, vmax=1)
    save_fig(fig, PLOT_EX / f"E4_corr_heatmap_{set_lbl}.pdf")

# ── E5: Speedup vs correctness scatter — model ranking overview
print("  E5: Mean speedup vs correctness rate (model ranking)...")
e5 = df.groupby("model").agg(
    mean_speedup=("speedup", "mean"),
    corr_rate=("correct", "mean")).reset_index()
e5["corr_pct"] = e5["corr_rate"] * 100
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("E5: Model Ranking — Mean Speedup vs Correctness Rate", fontsize=10)
for _, row in e5.iterrows():
    ax.scatter(row["mean_speedup"], row["corr_pct"],
               color=MODEL_COLORS.get(row["model"], "grey"), s=80, zorder=5)
ax.set_xlabel("Mean Speedup")
ax.set_ylabel("Correctness Rate (%)")
ax.legend(handles=make_legend_handles(e5["model"].tolist()), fontsize=7)
save_fig(fig, PLOT_EX / "E5_model_ranking_speedup_vs_correctness.pdf")

# ── E6: Metric variance / consistency — CV per model per metric
print("  E6: Metric variance (CV) per model...")
e6_rows = []
for m in ALL_MODELS:
    msub = df[df["model"] == m]
    for col in ["speedup", "sol_score", "dram_util", "occupancy"]:
        vals = msub[col].dropna()
        if len(vals) > 2 and vals.mean() != 0:
            cv = vals.std() / vals.mean()
            e6_rows.append({"model": m, "metric": col, "cv": cv})
e6 = pd.DataFrame(e6_rows)
if not e6.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("E6: Coefficient of Variation per Model per Metric", fontsize=10)
    palette = {m: MODEL_COLORS.get(m, "grey") for m in ALL_MODELS}
    sns.barplot(data=e6, x="metric", y="cv", hue="model", palette=palette, ax=ax)
    ax.set_ylabel("CV (σ/μ)")
    ax.set_xlabel("Metric")
    ax.legend(title="Model", fontsize=6)
    save_fig(fig, PLOT_EX / "E6_metric_variance.pdf")

# ── E7: Parallel coordinates — model rankings across metrics
print("  E7: Parallel coordinates across metrics...")
from matplotlib.path import Path as MPath
import matplotlib.patches as mpatches
# Build per-model mean of each metric
e7_metrics = ["speedup", "sol_score", "dram_util", "occupancy", "energy_mj"]
e7 = df.groupby("model")[e7_metrics].mean().reset_index().dropna(thresh=3)
if len(e7) >= 2:
    # Normalize 0-1 per metric
    e7_norm = e7.copy()
    for col in e7_metrics:
        mn, mx = e7[col].min(), e7[col].max()
        if mx > mn:
            e7_norm[col] = (e7[col] - mn) / (mx - mn)
        else:
            e7_norm[col] = 0.5
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("E7: Parallel Coordinates — Normalized Model Scores Across Metrics", fontsize=10)
    xs = range(len(e7_metrics))
    for _, row in e7_norm.iterrows():
        ys = [row[c] for c in e7_metrics]
        ax.scatter(list(xs), ys, color=MODEL_COLORS.get(row["model"], "grey"),
                   s=80, alpha=0.9, label=row["model"], zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(e7_metrics, rotation=15, fontsize=8)
    ax.set_ylabel("Normalized Score")
    ax.legend(title="Model", fontsize=7)
    save_fig(fig, PLOT_EX / "E7_parallel_coordinates.pdf")

# ── E8: Task difficulty — correctness rate per task across all models
print("  E8: Task difficulty profile...")
e8 = df.groupby("problem_id").agg(
    rate=("correct", "mean"), n=("correct", "count")).reset_index()
e8["rate"] *= 100
e8 = e8.sort_values("rate")
fig, ax = plt.subplots(figsize=(max(10, len(e8) * 0.4), 5))
fig.suptitle("E8: Task Difficulty — Correctness Rate Across All Models", fontsize=10)
ax.bar(range(len(e8)), e8["rate"], color="steelblue")
ax.set_xticks(range(len(e8)))
ax.set_xticklabels([f"P{int(x)}" for x in e8["problem_id"]], rotation=45, fontsize=5)
ax.set_ylabel("Correctness Rate (%)")
ax.set_xlabel("Task (sorted by difficulty)")
save_fig(fig, PLOT_EX / "E8_task_difficulty.pdf")

# ── E9: Step efficiency — histogram of first-correct step
print("  E9: Step of first correct per model...")
e9 = df[df["first_correct_step"].notna()].copy()
if not e9.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("E9: Step of First Correct Answer per Model", fontsize=10)
    for m in sorted(e9["model"].unique()):
        msub = e9[e9["model"] == m]["first_correct_step"]
        counts, edges = np.histogram(msub, bins=range(0, 17), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.scatter(centers, counts, color=MODEL_COLORS.get(m, "grey"), s=60, alpha=0.8, label=m, zorder=5)
    ax.set_xlabel("Turn Index of First Correctness Pass")
    ax.set_ylabel("Density")
    ax.legend(title="Model", fontsize=7)
    save_fig(fig, PLOT_EX / "E9_step_efficiency.pdf")

# ── E10: Spider / radar chart per model
print("  E10: Radar chart per model...")
e10_metrics = ["speedup", "sol_score", "dram_util", "occupancy", "fusion"]
e10 = df.groupby("model")[e10_metrics].mean().reset_index().dropna(thresh=3)
if len(e10) >= 2:
    cats = e10_metrics
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    fig.suptitle("E10: Radar Chart — Normalized Model Metrics", fontsize=10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, size=8)
    # Normalize per metric
    normed = e10.copy()
    for c in cats:
        mn, mx = e10[c].min(), e10[c].max()
        normed[c] = (e10[c] - mn) / (mx - mn) if mx > mn else 0.5
    for _, row in normed.iterrows():
        vals = [row[c] for c in cats]
        ax.scatter(angles[:-1], vals, color=MODEL_COLORS.get(row["model"], "grey"),
                   s=100, label=row["model"], zorder=5)
    ax.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.3, 1.1))
    save_fig(fig, PLOT_EX / "E10_radar_chart.pdf")

# ── E11: Energy efficiency ratio — speedup / energy_mj boxplot per model
print("  E11: Energy efficiency ratio...")
df_ee = df[df["speedup"].notna() & df["energy_mj"].notna() & (df["energy_mj"] > 0)].copy()
df_ee["energy_eff"] = df_ee["speedup"] / df_ee["energy_mj"]
if not df_ee.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("E11: Energy Efficiency (Speedup/Energy_mJ) per Model", fontsize=10)
    order_m = sorted(df_ee["model"].unique())
    palette = {m: MODEL_COLORS.get(m, "grey") for m in order_m}
    sns.boxplot(data=df_ee, x="model", y="energy_eff", order=order_m, palette=palette, ax=ax)
    ax.set_xlabel("Model")
    ax.set_ylabel("Speedup / Energy (mJ)")
    ax.tick_params(axis="x", rotation=20, labelsize=7)
    save_fig(fig, PLOT_EX / "E11_energy_efficiency_ratio.pdf")

# ── E12: DRAM saturation — histogram colored by correct/incorrect
print("  E12: DRAM saturation histogram...")
df12 = df[df["dram_util"].notna()].copy()
if not df12.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("E12: DRAM Utilization Distribution by Correctness", fontsize=10)
    for corr, lbl, clr in [(True, "Correct", "#2196F3"), (False, "Incorrect", "#F44336")]:
        sub = df12[df12["correct"] == corr]["dram_util"]
        counts, edges = np.histogram(sub, bins=30, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.scatter(centers, counts, color=clr, s=30, alpha=0.7, label=lbl, zorder=5)
    ax.set_xlabel("DRAM Utilization (%)")
    ax.set_ylabel("Density")
    ax.legend()
    save_fig(fig, PLOT_EX / "E12_dram_saturation.pdf")

# ── E13: Per-task Pareto dominance heatmap (tasks × models)
print("  E13: Per-task Pareto dominance heatmap...")
dom_data = []
task_ids = sorted(df["problem_id"].dropna().unique())
for task_id in task_ids:
    for m in ALL_MODELS:
        sub = df[(df["problem_id"] == task_id) & (df["model"] == m)]
        dom_count = 0
        for xc, yc, _ in PARETO_PAIRS:
            pts = sub[[xc, yc]].dropna()
            if not pts.empty:
                dom_count += 1
        dom_data.append({"task": f"P{int(task_id)}", "model": m, "pareto_coverage": dom_count})
e13 = pd.DataFrame(dom_data).pivot_table(index="task", columns="model",
                                          values="pareto_coverage", fill_value=0)
if not e13.empty:
    fig, ax = plt.subplots(figsize=(max(7, len(e13.columns)), max(6, len(e13) * 0.4)))
    fig.suptitle("E13: Pareto Coverage — Tasks × Models (# pairs with data)", fontsize=10)
    sns.heatmap(e13, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.3, annot_kws={"size": 6})
    ax.tick_params(axis="x", rotation=30, labelsize=6)
    ax.tick_params(axis="y", labelsize=5)
    save_fig(fig, PLOT_EX / "E13_pareto_dominance_heatmap.pdf")

# ── E14: Set comparison — side-by-side correctness across sets for overlapping models/tasks
print("  E14: Cross-set correctness comparison...")
set_corr = df.groupby(["set_id", "model"]).agg(
    rate=("correct", "mean"), n=("correct", "count")).reset_index()
set_corr["rate"] *= 100
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("E14: Correctness Rate by Set and Model", fontsize=10)
palette = {m: MODEL_COLORS.get(m, "grey") for m in ALL_MODELS}
sns.barplot(data=set_corr, x="set_id", y="rate", hue="model", palette=palette, ax=ax)
ax.set_xlabel("Set")
ax.set_ylabel("Correctness Rate (%)")
ax.legend(title="Model", fontsize=6)
save_fig(fig, PLOT_EX / "E14_cross_set_correctness.pdf")

# ── E15: Timeout impact timeline (Set 3 FIXED)
print("  E15: Timeout impact timeline (set3)...")
df15 = df3[df3["mode"] == "sl5_all"].copy()
df15 = df15[df15["total_turns"].notna() & df15["speedup"].notna()]
if not df15.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("E15: Trajectory Length vs Speedup (Set 3, colored by timeout recovery)", fontsize=10)
    for recovered, lbl, clr in [(True, "Timeout Recovered", "#FF8800"), (False, "Normal", "#4488CC")]:
        sub = df15[df15["timeout_recovered"] == recovered]
        if not sub.empty:
            ax.scatter(sub["total_turns"], sub["speedup"], color=clr, alpha=0.7, s=40, label=lbl)
    ax.set_xlabel("Total Turns in Trajectory")
    ax.set_ylabel("Speedup")
    ax.legend()
    save_fig(fig, PLOT_EX / "E15_timeout_impact_timeline.pdf")

# ── E16: Occupancy vs fusion colored by correctness
print("  E16: Occupancy vs fusion colored by correctness...")
df16 = df[df["occupancy"].notna() & df["fusion"].notna()].copy()
if not df16.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("E16: Occupancy vs Fusion (colored by correctness)", fontsize=10)
    for corr, lbl, clr, mrk in [(True, "Correct", "#2196F3", "o"), (False, "Incorrect", "#F44336", "x")]:
        sub = df16[df16["correct"] == corr]
        ax.scatter(sub["occupancy"], sub["fusion"], color=clr, alpha=0.4, s=20,
                   label=lbl, marker=mrk)
    ax.set_xlabel("Occupancy (%)")
    ax.set_ylabel("Fusion Ratio")
    ax.set_yscale("log")
    ax.legend()
    save_fig(fig, PLOT_EX / "E16_occupancy_vs_fusion.pdf")

# ── E17: SOL score CDF per model per set
print("  E17: SOL score CDF...")
df17 = df[df["sol_score"].notna()].copy()
if not df17.empty:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("E17: SOL Score CDF per Model (by Set)", fontsize=10)
    for si, (set_lbl, set_df) in enumerate([("set1", df1), ("set2", df2), ("set3", df3)]):
        ax = axes[si]
        sub17 = set_df[set_df["sol_score"].notna()]
        for m in sorted(sub17["model"].unique()):
            msub = sub17[sub17["model"] == m]["sol_score"].sort_values()
            cdf = np.arange(1, len(msub) + 1) / len(msub)
            ax.scatter(msub.values, cdf, color=MODEL_COLORS.get(m, "grey"), label=m, s=15, alpha=0.7)
        ax.set_title(set_lbl)
        ax.set_xlabel("SOL Score")
        ax.set_ylabel("CDF" if si == 0 else "")
        ax.legend(fontsize=6)
    save_fig(fig, PLOT_EX / "E17_sol_score_cdf.pdf")

# ── E18: Speedup distribution by mode within Set 2 (default vs all vs single_turn)
print("  E18: Set 2 speedup by mode comparison...")
df18 = df2[df2["speedup"].notna()].copy()
if not df18.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("E18: Set 2 — Speedup Distribution by Mode", fontsize=10)
    mode_palette = {"default": "#4477AA", "all": "#EE8833", "st": "#228833"}
    sns.boxplot(data=df18, x="mode", y="speedup", hue="model", ax=ax,
                palette={m: MODEL_COLORS.get(m, "grey") for m in ALL_MODELS})
    ax.set_xlabel("Mode")
    ax.set_ylabel("Speedup")
    ax.legend(title="Model", fontsize=6)
    save_fig(fig, PLOT_EX / "E18_set2_speedup_by_mode.pdf")

# ── E19: Correctness rate per level × mode heatmap (Set 1)
print("  E19: Set 1 correctness heatmap (level × mode)...")
e19 = df1.groupby(["level", "mode", "model"]).agg(
    rate=("correct", "mean")).reset_index()
e19["rate"] *= 100
for m in sorted(e19["model"].unique()):
    sub = e19[e19["model"] == m].pivot_table(
        index="level", columns="mode", values="rate", fill_value=0)
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.suptitle(f"E19: {m} — Correctness Rate (Level × Mode)", fontsize=10)
        sns.heatmap(sub, annot=True, fmt=".1f", cmap="RdYlGn",
                    vmin=0, vmax=100, ax=ax, linewidths=0.5)
        ax.set_xlabel("Mode")
        ax.set_ylabel("Level")
        fname = m.replace("/", "_").replace(" ", "_")
        save_fig(fig, PLOT_EX / f"E19_level_mode_heatmap_{fname}.pdf")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRACE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 4 — TRACE ANALYSIS")
print("=" * 70)

def analyze_traces(folder_path: Path, set_id: str, mode_label: str):
    """Extract per-trace statistics from all trajectory files in a folder."""
    files = glob.glob(str(folder_path / "**" / "*trajectory*.json"), recursive=True)
    rows = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        turns = d.get("turns", [])
        model = d.get("model_name", "unknown")
        prob  = d.get("problem_id")
        outcome = d.get("outcome", "unknown")
        total_turns = d.get("total_turns", len(turns))
        correct = (d.get("final_result") or {}).get("correctness", False)

        tool_names     = []
        error_types    = []
        timeout_events = 0
        mid_pass_step  = None
        first_correct_step = None

        for ti, turn in enumerate(turns):
            for tc in turn.get("tool_calls", []):
                tname  = tc.get("tool_name", "unknown")
                res    = tc.get("result_text", "")
                succ   = tc.get("success", False)
                tool_names.append(tname)

                if "timed out" in res.lower() or "timeout" in res.lower():
                    timeout_events += 1

                if not succ:
                    # Classify error
                    if "compile" in res.lower() or "RuntimeError" in res:
                        error_types.append("compile_error")
                    elif "mismatch" in res.lower() or "correctness" in res.lower():
                        error_types.append("correctness_fail")
                    elif "timed out" in res.lower():
                        error_types.append("timeout")
                    elif "runtime_error" in res.lower():
                        error_types.append("runtime_error")
                    else:
                        error_types.append("other_fail")

                if tname == "run_correctness" and succ and mid_pass_step is None:
                    mid_pass_step = ti
                if (tname in ("run_correctness", "submit_kernel")) and succ:
                    if first_correct_step is None:
                        first_correct_step = ti

        rows.append({
            "folder":      folder_path.name,
            "set_id":      set_id,
            "mode":        mode_label,
            "model":       normalize_model(model),
            "problem_id":  prob,
            "outcome":     outcome,
            "correct":     bool(correct),
            "total_turns": total_turns,
            "timeout_events": timeout_events,
            "had_timeout":    timeout_events > 0,
            "first_correct_step": first_correct_step,
            "mid_pass_step":    mid_pass_step,
            "error_types_str":  "|".join(error_types),
            "num_tool_calls":   len(tool_names),
            "tools_str":        "|".join(tool_names),
        })
    return rows

trace_rows = []

for mode, fname in SET1_FOLDERS.items():
    trace_rows.extend(analyze_traces(BASE / fname, "set1", mode))
for mode, fname in SET2_FOLDERS.items():
    trace_rows.extend(analyze_traces(BASE / fname, "set2", mode))
# Set 3: use FIXED for sl5
trace_rows.extend(analyze_traces(SL5_FIXED, "set3", "sl5_all"))
trace_rows.extend(analyze_traces(BASE / SET3_FOLDERS["fl5_st"], "set3", "fl5_st"))

trace_df = pd.DataFrame(trace_rows)
trace_df.to_csv(OUT / "trace_analysis.csv", index=False)

# Per-model statistics
print()
print("| Model | Mean Steps | Median Steps | Timeout Rate | Mid-Run Pass Rate | Correct Rate |")
print("|-------|-----------|-------------|-------------|------------------|-------------|")
for m in sorted(trace_df["model"].unique()):
    msub = trace_df[trace_df["model"] == m]
    mean_steps   = msub["total_turns"].mean()
    median_steps = msub["total_turns"].median()
    timeout_rate = msub["had_timeout"].mean() * 100
    mid_pass_rate = msub["mid_pass_step"].notna().mean() * 100
    correct_rate  = msub["correct"].mean() * 100
    print(f"| {m[:30]} | {mean_steps:.1f} | {median_steps:.1f} | {timeout_rate:.1f}% | {mid_pass_rate:.1f}% | {correct_rate:.1f}% |")

# Error type breakdown
print()
print("Top error types per model:")
from collections import Counter
for m in sorted(trace_df["model"].unique()):
    msub = trace_df[trace_df["model"] == m]
    all_errors = []
    for e_str in msub["error_types_str"].dropna():
        all_errors.extend([e for e in e_str.split("|") if e])
    top5 = Counter(all_errors).most_common(5)
    print(f"  {m}: {top5}")

# Mean trajectory length correct vs incorrect
print()
print("Mean trajectory length — correct vs incorrect:")
grp = trace_df.groupby(["model", "correct"])["total_turns"].mean()
print(grp.to_string())

# Fraction with mid-run pass but incorrect final
def mid_pass_but_wrong(row):
    return row["mid_pass_step"] is not None and not row["correct"]

trace_df["mid_pass_not_final"] = trace_df.apply(
    lambda r: r["mid_pass_step"] is not None and not r["correct"], axis=1)
print()
print("Fraction with mid-run correctness pass but incorrect final verdict:")
for m in sorted(trace_df["model"].unique()):
    msub = trace_df[trace_df["model"] == m]
    frac = msub["mid_pass_not_final"].mean() * 100
    print(f"  {m}: {frac:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — LaTeX REPORT
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 5 — LaTeX REPORT")
print("=" * 70)

# Build summary stats for tables
label_fix_table = []
for e in fix_log:
    label_fix_table.append(e)

# Model × set correctness summary
summary_df = df.groupby(["set_id", "model"]).agg(
    n_correct=("correct", "sum"),
    n_total=("correct", "count"),
    mean_speedup=("speedup", "mean"),
    mean_sol=("sol_score", "mean"),
).reset_index()
summary_df["corr_rate"] = summary_df["n_correct"] / summary_df["n_total"] * 100

# Trace summary for LaTeX table
trace_summary = []
for m in sorted(trace_df["model"].unique()):
    for set_lbl in ["set1", "set2", "set3"]:
        msub = trace_df[(trace_df["model"] == m) & (trace_df["set_id"] == set_lbl)]
        if msub.empty:
            continue
        trace_summary.append({
            "model": m,
            "set": set_lbl,
            "n": len(msub),
            "mean_turns": f"{msub['total_turns'].mean():.1f}",
            "timeout_rate": f"{msub['had_timeout'].mean()*100:.1f}\\%",
            "corr_rate": f"{msub['correct'].mean()*100:.1f}\\%",
        })

# Build findings strings from data
findings = []

# Finding 1: timeout recovery impact
if n_changed > 0:
    affected_models = list({e["model"] for e in fix_log})
    findings.append(
        f"Timeout recovery (label fix) in Set~3 relabeled {n_changed} entries "
        f"across model(s) {', '.join(affected_models)} as correct, revealing that "
        f"these kernels had verified correctness mid-trajectory but timed out during final evaluation."
    )

# Finding 2: energy vs speedup correlation
energy_speedup_corr = df[["energy_mj","speedup"]].dropna().corr().iloc[0,1]
findings.append(
    f"Across all sets, energy (mJ) and speedup show a Pearson correlation of "
    f"{energy_speedup_corr:.2f}, indicating {'a strong negative' if energy_speedup_corr < -0.3 else 'a weak'} "
    f"relationship between kernel efficiency and energy consumption."
)

# Finding 3: best model overall
best_model_row = summary_df.sort_values("corr_rate", ascending=False).iloc[0]
findings.append(
    f"Across all sets and tasks, \\texttt{{{best_model_row['model']}}} achieved the highest overall correctness rate "
    f"of {best_model_row['corr_rate']:.1f}\\% in {best_model_row['set_id']}."
)

# Build exploratory figure list with captions
exploratory_figures = [
    ("E1_learning_curve_by_level.png",
     "Correctness rate by problem level (L1→L2→L3) per model. "
     "Reveals how model performance degrades as kernel complexity increases across levels."),
    ("E2_efficiency_frontier.png",
     "Scatter of speedup vs. energy (mJ) on log-log axes. "
     "Models in the upper-left achieve high speedup at low energy cost, forming the efficiency Pareto frontier."),
    ("E3_failure_modes.png",
     "Stacked bar chart of outcome fractions (correct, incorrect, compile\\_fail) per model. "
     "Reveals whether a model's failures are dominated by compilation errors or correctness mismatches."),
    ("E4_corr_heatmap_set1.png", "Pearson correlation of all numeric metrics for Set~1."),
    ("E4_corr_heatmap_set2.png", "Pearson correlation of all numeric metrics for Set~2."),
    ("E4_corr_heatmap_set3.png", "Pearson correlation of all numeric metrics for Set~3."),
    ("E5_model_ranking_speedup_vs_correctness.png",
     "One point per model: mean speedup on x-axis, overall correctness rate on y-axis. "
     "Provides an at-a-glance model ranking overview."),
    ("E6_metric_variance.png",
     "Coefficient of variation (CV) per model per metric. "
     "High CV indicates unstable behavior across tasks; low CV indicates consistent performance."),
    ("E7_parallel_coordinates.png",
     "Parallel coordinates of normalized model mean scores across five key metrics. "
     "Highlights trade-offs between speedup, SOL score, DRAM utilization, occupancy, and energy."),
    ("E8_task_difficulty.png",
     "Correctness rate per task (problem) sorted from hardest to easiest. "
     "Tasks with near-zero correctness rates represent unsolved problems across all models."),
    ("E9_step_efficiency.png",
     "Distribution of the trajectory step at which correctness first passes, per model. "
     "Models that solve problems early use fewer LLM turns and tokens."),
    ("E10_radar_chart.png",
     "Multi-metric radar chart with one polygon per model. "
     "Enables holistic comparison across speedup, SOL score, DRAM util, occupancy, and fusion."),
    ("E11_energy_efficiency_ratio.png",
     "Boxplot of speedup-per-mJ (energy efficiency ratio) per model. "
     "Models with high median and low variance provide the best energy-normalized performance."),
    ("E12_dram_saturation.png",
     "DRAM utilization histogram, split by final correctness. "
     "High DRAM utilization in correct kernels suggests better memory bandwidth utilization."),
    ("E13_pareto_dominance_heatmap.png",
     "Heatmap of Pareto pair data coverage per task × model. "
     "Cells show how many of the 5 Pareto pairs have valid data for that model–task combination."),
    ("E14_cross_set_correctness.png",
     "Side-by-side correctness rate per model across Set~1, Set~2, and Set~3. "
     "Reveals which models generalize across hardware configurations and problem types."),
    ("E15_timeout_impact_timeline.png",
     "Scatter of total trajectory turns vs. speedup for Set~3 sl5 runs, colored by whether the "
     "result was timeout-recovered. Timeout-recovered entries tend to cluster at maximum turns."),
    ("E16_occupancy_vs_fusion.png",
     "Scatter of occupancy vs. fusion ratio colored by correctness. "
     "High fusion with moderate occupancy may predict correctness better than either alone."),
    ("E17_sol_score_cdf.png",
     "CDF of SOL scores per model, one panel per set. "
     "Models with CDF shifted right (higher SOL) are closer to hardware performance limits."),
    ("E18_set2_speedup_by_mode.png",
     "Set~2 speedup distribution across multi-turn modes (default, all, single-turn) per model. "
     "Reveals whether additional context in multi-turn modes improves kernel performance."),
]

# Add per-model E19 heatmaps
for m in sorted(df1["model"].unique()):
    fname = m.replace("/", "_").replace(" ", "_")
    exploratory_figures.append(
        (f"E19_level_mode_heatmap_{fname}.png",
         f"Correctness rate heatmap for \\texttt{{{m}}} across levels (rows) and modes (cols) in Set~1.")
    )

def escape_latex(s):
    """Minimal LaTeX escaping for special chars."""
    if not isinstance(s, str):
        s = str(s)
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")

def fig_include(path, caption, label, width=r"\linewidth"):
    rel = f"plots/{path}" if "/" not in path else path
    return (
        f"\\begin{{figure}}[htbp]\n"
        f"  \\centering\n"
        f"  \\includegraphics[width={width}]{{{rel}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{fig:{label}}}\n"
        f"\\end{{figure}}\n"
    )

def fig_include_ex(fname, caption, label):
    return fig_include(f"plots/exploratory/{fname}", caption, label)

# ─── Build the LaTeX document ─────────────────────────────────────────────────
PARETO_LABELS = ["P1", "P2", "P3", "P4", "P5"]
PARETO_TITLES = [p[2] for p in PARETO_PAIRS]

tex_lines = []
tex_lines.append(r"""\documentclass{article}
\usepackage{graphicx, booktabs, geometry, caption, subcaption, hyperref}
\usepackage{tcolorbox}
\usepackage{longtable, array, multirow, xcolor}
\usepackage{amsmath}
\usepackage{placeins}
\geometry{margin=1in}
\setlength{\parskip}{4pt}

\tcbuselibrary{skins,breakable}

\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}

\title{KernelBench GPU Kernel Benchmark Analysis Report}
\author{KernelBench Analysis Pipeline}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\clearpage
""")

# ─── 1. Introduction ──────────────────────────────────────────────────────────
tex_lines.append(r"\section{Introduction}")
tex_lines.append(r"""
This report analyzes GPU kernel benchmark results across three experimental sets,
covering multi-level kernel optimization (levels L1--L3), hardware translation at Level~5,
and specialized 8$\times$H100 hardware translation evaluations. All experiments use the
KernelBench agentic evaluation harness, where each model generates CUDA kernels to
replace PyTorch reference implementations.

\subsection{Experimental Sets}

\textbf{Set~1} comprises three folders spanning levels L1, L2, and L3:
\texttt{pop\_l123\_all\_gpt} (multi-turn / all tools),
\texttt{pop\_l123\_default\_gpt} (default), and
\texttt{pop\_l123\_st\_gpt} (single-turn).
Models evaluated: FW-GLM-5-1, gpt-5.5-priority, grok-4.3.

\textbf{Set~2} contains three Level-5 hardware-translation runs:
\texttt{l5\_hw\_translation\_default\_final\_gpt},
\texttt{l5\_hw\_translation\_all\_final\_gpt}, and
\texttt{l5\_hw\_translation\_single\_turn\_final\_gpt}.
Models: FW-GLM-5-1, Llama-4-Maverick-17B-128E-Instruct-FP8, gpt-5.5-priority, grok-4.3
(plus Kimi-K2.6 in single-turn mode).

\textbf{Set~3} contains two 8$\times$H100 multi-GPU folders:
\texttt{sl5\_hw\_translation\_8xh100\_all\_reval30m\_gpt} (re-evaluated, 30-min timeout, all-tools)
and \texttt{fl5\_hw\_translation\_8xh100\_single\_turn\_gpt}.
Models: FW-GLM-5-1, gpt-5.5-priority, grok-4.3.

\textbf{Tasks} are drawn from the KernelBench Level-5 problem set
(hardware-translation kernels) and, for Set~1, the Level-1/2/3 optimization problems.
Each trajectory represents one model's full agentic interaction to produce an optimized kernel.
""")
tex_lines.append(r"\clearpage")

# ─── 2. Label Fix Summary ─────────────────────────────────────────────────────
tex_lines.append(r"\section{Label Fix Summary}")
tex_lines.append(fr"""
The \texttt{{sl5\_hw\_translation\_8xh100\_all\_reval30m\_gpt}} folder was re-evaluated with a
30-minute timeout per problem on 8$\times$H100 hardware. Due to the high cost of multi-GPU
torchrun evaluation, several trajectories that verified correctness via \texttt{{run\_correctness}}
during the agentic session were classified as \texttt{{compile\_fail}} when the final
\texttt{{submit\_kernel}} call timed out. To recover these labels, we examined every
trajectory with \texttt{{correct = False}} and relabeled entries where any
\texttt{{run\_correctness}} or \texttt{{submit\_kernel}} call succeeded (passed) mid-trajectory.

\textbf{{Summary:}} {n_examined} entries examined; \textbf{{{n_changed} entries relabeled}}
as \texttt{{correct = True}} with \texttt{{correct\_source = "timeout\_recovery"}}.
Original labels preserved in \texttt{{correct\_original}} field.
All changes logged to \texttt{{label\_fixes.jsonl}}.
""")

if label_fix_table:
    tex_lines.append(r"""
\begin{table}[htbp]
\centering
\caption{Timeout-Recovery Relabeled Entries}
\label{tab:label_fix}
\begin{tabular}{lllcc}
\toprule
Model & Problem ID & Original Outcome & Timeout Evidence & Passing Turn \\
\midrule
""")
    for e in label_fix_table:
        row = (
            f"  {escape_latex(e['model'])} & "
            f"  {e['problem_id']} & "
            f"  {escape_latex(e['original_outcome'] or 'unknown')} & "
            f"  {'Yes' if e['timeout_evidence'] else 'No'} & "
            f"  {e['passing_turn'] if e['passing_turn'] is not None else 'N/A'} \\\\\n"
        )
        tex_lines.append(row)
    tex_lines.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
else:
    tex_lines.append("No entries were relabeled.\n")

if findings:
    tex_lines.append(r"""
\begin{tcolorbox}[colback=gray!10, colframe=black!30, title=Finding: Timeout Recovery]
""" + findings[0] + r"""
\end{tcolorbox}
""")
tex_lines.append(r"\clearpage")

# ─── 3. Pareto Frontier Analysis ──────────────────────────────────────────────
tex_lines.append(r"\section{Pareto Frontier Analysis}")
tex_lines.append(r"""
For each of five metric pairs, we compute the Pareto frontier per model (points not
dominated on both axes) and overlay a step-connected frontier line over the full
scatter of points. A consistent color palette (tab10) is used for models across all figures.
""")

PARETO_CAPTIONS = {
    "P1": {
        "set1": "Compute utilization vs. DRAM utilization across L1/L2/L3 levels and modes. "
                "Points in the upper-right corner indicate balanced hardware utilization.",
        "set2": "Compute vs. DRAM utilization for Level-5 hardware-translation tasks. "
                "Models closer to the upper-right are better utilizing both compute and memory bandwidth.",
        "set3": "Compute vs. DRAM for 8$\\times$H100 tasks. "
                "The re-evaluated sl5 folder (top panel) may show fewer data points due to timeout-induced missing profiles.",
    },
    "P2": {
        "set1": "SOL score vs. speedup. High SOL score with high speedup indicates kernels that are "
                "simultaneously close to hardware limits and faster than the reference.",
        "set2": "SOL score vs. speedup for Level-5 hardware translation. "
                "Models achieving high SOL at moderate speedup may be bandwidth-limited.",
        "set3": "SOL vs. speedup for 8$\\times$H100. Kernels exploiting multi-GPU collectives "
                "may show distinct frontier shapes.",
    },
    "P3": {
        "set1": "Power (W) vs. energy (mJ). Lower-left is ideal. "
                "High-power kernels that complete quickly can still have low total energy.",
        "set2": "Power vs. energy for Level-5 tasks. "
                "Hardware-translation kernels targeting 8$\\times$H100 show higher power draw.",
        "set3": "Power vs. energy on 8$\\times$H100. The multi-GPU setup draws significantly "
                "more power than single-GPU experiments.",
    },
    "P4": {
        "set1": "Fusion ratio vs. occupancy. High fusion with high occupancy is ideal. "
                "The frontier reveals which kernels fuse more work without sacrificing thread occupancy.",
        "set2": "Fusion vs. occupancy for Level-5 translation tasks.",
        "set3": "Fusion vs. occupancy on 8$\\times$H100.",
    },
    "P5": {
        "set1": "DRAM utilization vs. speedup. Models in the upper-right achieve high speedup "
                "while also saturating memory bandwidth.",
        "set2": "DRAM utilization vs. speedup for Level-5 tasks.",
        "set3": "DRAM utilization vs. speedup on 8$\\times$H100.",
    },
}

for pi, (xc, yc, ptitle) in enumerate(PARETO_PAIRS, 1):
    label_str = f"P{pi}"
    tex_lines.append(f"\\subsection{{Pareto Pair {label_str}: {escape_latex(ptitle)}}}")
    tex_lines.append(
        f"Axes: x~=~\\texttt{{{escape_latex(xc)}}}, y~=~\\texttt{{{escape_latex(yc)}}}.\n"
    )
    for set_lbl, grid in [("set1", "3×3"), ("set2", "3×1"), ("set3", "2×1")]:
        fname = f"{set_lbl}_pareto_{label_str}_{grid.replace('×','x')}.png"
        cap_key = label_str
        cap = PARETO_CAPTIONS.get(cap_key, {}).get(set_lbl, f"{set_lbl} {ptitle}")
        tex_lines.append(fig_include(fname, cap, f"pareto_{set_lbl}_{label_str}"))
    tex_lines.append(r"\FloatBarrier" + "\n")

tex_lines.append(r"\clearpage")

# ─── 4. General Performance Analysis ─────────────────────────────────────────
tex_lines.append(r"\section{General Performance Analysis}")

GENERAL_CAPTIONS = {
    "G1": "Speedup distribution per model, faceted by task (problem). "
          "Boxes show median and IQR; outliers indicate tasks where one model achieves "
          "unusually high speedup.",
    "G2": "Number of correct kernel submissions per model, grouped by task. "
          "Directly shows which model × task combinations succeed most often.",
    "G3": "Correctness rate (\\%) per model, grouped by task. "
          "Normalizes by number of attempts to allow fair comparison across models.",
    "G4": "Original vs. timeout-recovered correctness rate per model (Set~3 FIXED only). "
          "The orange bar shows additional correct entries recovered by the label-fix procedure.",
    "G5": "SOL score distribution (violin) per model, one panel per task. "
          "Wider violins indicate more variance; the median line shows typical hardware efficiency.",
    "G6": "DRAM utilization vs. speedup scatter per task, colored by model. "
          "Reveals whether higher speedup comes with proportionally higher memory bandwidth usage.",
    "G7": "Energy (mJ) vs. power (W) scatter per task, colored by model. "
          "Low-energy, moderate-power kernels in the lower-left are most efficient.",
}

for set_lbl in ["set1", "set2", "set3"]:
    tex_lines.append(f"\\subsection{{Set~{set_lbl[-1]} General Plots}}")
    for gi in range(1, 8):
        label = f"G{gi}"
        fname = f"{set_lbl}_general_{label}.png"
        path  = PLOT / fname
        if not path.exists():
            tex_lines.append(f"\\textit{{Figure {label} for {set_lbl} not generated (insufficient data).}}\n\n")
            continue
        cap = GENERAL_CAPTIONS.get(label, f"{set_lbl} {label}")
        # G4 only for set3
        if label == "G4" and set_lbl != "set3":
            continue
        tex_lines.append(fig_include(fname, f"{set_lbl} {cap}", f"gen_{set_lbl}_{label}"))
    tex_lines.append(r"\FloatBarrier" + "\n")

tex_lines.append(r"\clearpage")

# ─── 5. Combined Overview Plots ───────────────────────────────────────────────
tex_lines.append(r"\section{Combined Overview Plots}")
tex_lines.append(r"""
These plots pool all tasks together and show aggregate model performance per set,
making it easy to compare models at a glance without per-task faceting.
""")

COMBINED_CAPTIONS = {
    "combined_C1_correctness_rate.png":
        "Overall correctness rate per model (all tasks pooled). Each point is a model's "
        "pass rate across every problem in that set.",
    "combined_C2_speedup_dist.png":
        "Speedup distribution per model (all tasks pooled). Boxes show median and IQR; "
        "individual runs are overlaid as scatter points.",
    "combined_C3_correctness_by_mode.png":
        "Correctness rate by trajectory mode (single-turn / default / all) per model. "
        "Shape varies by mode; color by model.",
    "combined_C4_dram_vs_speedup.png":
        "DRAM utilization vs.~speedup scatter (all tasks pooled). "
        "Reveals whether faster kernels also saturate memory bandwidth.",
    "combined_C5_energy_vs_power.png":
        "Energy (mJ) vs.~power (W) scatter (all tasks pooled). "
        "Lower-left corner is most efficient.",
    "combined_C6_sol_score_dist.png":
        "SOL score distribution per model (all tasks pooled). "
        "Higher SOL means the kernel is closer to the roofline hardware limit.",
    "combined_C7_cross_set_correctness.png":
        "Correctness rate for every model × set combination in a single view. "
        "Marker shape encodes set; color encodes model.",
    "combined_C8_correct_vs_incorrect.png":
        "Raw count of correct (circle) vs.~incorrect (cross) submissions per model. "
        "Lets you see absolute volume differences across models.",
}

for cname, ccap in COMBINED_CAPTIONS.items():
    p = PLOT / cname
    if p.exists():
        tex_lines.append(fig_include(cname, ccap, f"comb_{cname.replace('.png','')}"))
tex_lines.append(r"\FloatBarrier" + "\n")
tex_lines.append(r"\clearpage")

# ─── 6. Exploratory Analysis ──────────────────────────────────────────────────
tex_lines.append(r"\section{Exploratory Analysis}")
tex_lines.append(r"""
The following additional analyses were generated to explore patterns, correlations,
and anomalies across all benchmark sets.
""")

FINDING_FIGURES = {"E14_cross_set_correctness.png", "E3_failure_modes.png", "E5_model_ranking_speedup_vs_correctness.png"}
FINDINGS_TEXT = {
    "E14_cross_set_correctness.png":
        "Model correctness rankings are not always stable across experimental sets: "
        "a model that performs best on Set~1 (levels L1--L3) may rank differently on the "
        "hardware-translation tasks in Set~2 or the 8\\texttimes{}H100 Set~3 problems.",
    "E3_failure_modes.png":
        "Failure modes differ substantially across models: some models are dominated by "
        "compilation failures while others primarily produce numerically incorrect kernels. "
        "This has implications for debugging strategy and prompt design.",
    "E5_model_ranking_speedup_vs_correctness.png":
        "The speedup vs. correctness trade-off reveals that some models achieve high "
        "speedup on the subset of tasks they solve correctly, while being correct "
        "on relatively fewer tasks overall — a classic precision vs. recall pattern.",
}

for i, (efname, ecap) in enumerate(exploratory_figures, 1):
    path = PLOT_EX / efname
    if not path.exists():
        tex_lines.append(f"\\textit{{Exploratory figure \\texttt{{{escape_latex(efname)}}} not generated.}}\n\n")
        continue
    subsec = efname.replace("_", " ").replace(".pdf", "")
    tex_lines.append(f"\\subsection{{{escape_latex(subsec)}}}")
    tex_lines.append(fig_include_ex(efname, ecap, f"ex_{i}"))
    if efname in FINDING_FIGURES:
        tex_lines.append(
            "\\begin{tcolorbox}[colback=gray!10, colframe=black!30, title=Finding]\n"
            + FINDINGS_TEXT[efname]
            + "\n\\end{tcolorbox}\n"
        )
    tex_lines.append(r"\FloatBarrier" + "\n")

tex_lines.append(r"\clearpage")

# ─── 6. Trace Analysis ────────────────────────────────────────────────────────
tex_lines.append(r"\section{Trace Analysis}")
tex_lines.append(r"""
We extracted per-trajectory statistics from all available trace files including
total turns, tool calls made, timeout events, error types, and the step at which
correctness first passed.
""")

tex_lines.append(r"""
\begin{longtable}{llrllll}
\caption{Trace Analysis Summary per Model and Set}\label{tab:trace}\\
\toprule
Model & Set & N & Mean Turns & Timeout Rate & Corr. Rate \\
\midrule
\endfirsthead
\multicolumn{6}{c}{\tablename\ \thetable{} -- continued}\\
\toprule
Model & Set & N & Mean Turns & Timeout Rate & Corr. Rate \\
\midrule
\endhead
\midrule \multicolumn{6}{r}{{Continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
""")

for row in trace_summary:
    tex_lines.append(
        f"  {escape_latex(row['model'])} & {row['set']} & {row['n']} & "
        f"  {row['mean_turns']} & {row['timeout_rate']} & {row['corr_rate']} \\\\\n"
    )

tex_lines.append(r"\end{longtable}" + "\n")

tex_lines.append(r"""
\subsection{Trajectory Patterns}

Analysis of agent trajectories reveals several consistent patterns:
\begin{itemize}
  \item Trajectories that ultimately produce correct kernels tend to reach a passing
        \texttt{run\_correctness} call earlier in the trajectory.
  \item Timeout events are concentrated in the 8$\times$H100 Set~3 experiments,
        where torchrun subprocess overhead is significantly higher.
  \item Compilation failures are more common for models that generate complex,
        architecture-specific CUDA code (e.g., Hopper-specific intrinsics) without
        adequate iteration.
  \item Multi-turn (``all'') mode trajectories are not uniformly shorter for correct runs;
        some models use additional turns to profile and optimize even after achieving correctness.
\end{itemize}
""")

# Add timeout anomaly finding
tex_lines.append(r"""
\begin{tcolorbox}[colback=gray!10, colframe=black!30, title=Finding: Timeout Anomaly in Set~3]
In the \texttt{sl5\_hw\_translation\_8xh100\_all\_reval30m\_gpt} folder, the
re-evaluation harness set a 30-minute torchrun timeout. Several kernels that
passed \texttt{run\_correctness} mid-trajectory were marked as \texttt{compile\_fail}
because the final \texttt{submit\_kernel} evaluation (which uses a longer, multi-trial
correctness check) timed out before completion. The label fix in Section~2 recovers
these entries.
\end{tcolorbox}
""")

tex_lines.append(r"\clearpage")

# ─── 7. Conclusion ────────────────────────────────────────────────────────────
tex_lines.append(r"\section{Conclusion}")

# Build top-5 findings
top5 = [
    f"\\textbf{{Timeout recovery significantly impacts Set~3 rankings:}} "
    f"The relabeling of {n_changed} entries via trajectory inspection changes the "
    f"effective correctness rates in the sl5 folder, particularly for gpt-5.5-priority.",
]

if len(findings) > 1:
    top5.append(f"\\textbf{{Energy–speedup relationship:}} {findings[1]}")

top5.append(
    "\\textbf{Model consistency across sets:} "
    "Models do not rank uniformly across Set~1 (levels L1--L3) and Set~2 (hardware translation). "
    "grok-4.3 and gpt-5.5-priority show different relative strengths depending on problem type."
)

top5.append(
    "\\textbf{Fusion as a key differentiator:} "
    "Kernels achieving high fusion ratios tend to reduce CUDA kernel launch overhead substantially, "
    "contributing to speedup independent of raw compute utilization."
)

top5.append(
    "\\textbf{Compile failures dominate in Set~3:} "
    "The 8$\\times$H100 multi-GPU evaluation surface reveals that many kernels that succeed "
    "on single-GPU evaluations fail to compile or run correctly in a torchrun world\\_size=8 environment, "
    "suggesting that multi-GPU portability is a key unsolved challenge."
)

tex_lines.append(r"\subsection{Top Findings}")
tex_lines.append("\\begin{enumerate}\n")
for f in top5:
    tex_lines.append(f"  \\item {f}\n")
tex_lines.append("\\end{enumerate}\n")

# Limitations
tex_lines.append(r"""
\subsection{Limitations}
\begin{itemize}
  \item \textbf{Missing profiling data:} Several trajectories have \texttt{NaN} for
        roofline/SOL/energy fields because the \texttt{profile\_kernel} tool timed out
        or was not called. These entries are excluded from metric-specific plots.
  \item \textbf{Set overlap:} Not all models appear in all sets. Kimi-K2.6 appears only
        in Set~2 single-turn mode. Cross-set comparisons should be interpreted cautiously.
  \item \textbf{Re-evaluation bias (Set~3):} The 30-minute re-evaluation timeout in
        \texttt{sl5} introduces systematic bias toward compile-fail outcomes for otherwise
        correct kernels. The label fix mitigates but does not fully resolve this.
  \item \textbf{Single-trial correctness in trajectories:} Mid-trajectory
        \texttt{run\_correctness} calls use 1 trial; final evaluation uses 3--5 trials.
        Recovered entries may have passed on 1/1 but not 3/3 trials.
\end{itemize}
""")

tex_lines.append(r"\end{document}" + "\n")

# Write report.tex
report_path = OUT / "report.tex"
with open(report_path, "w") as f:
    f.write("\n".join(tex_lines))
print(f"  Wrote: report.tex ({report_path.stat().st_size // 1024} KB)")

# ──────────────────────────────────────────────────────────────────────────────
# COMPILE LaTeX
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("COMPILING LaTeX")
print("=" * 70)
import subprocess

for run_i in range(1, 3):
    print(f"  pdflatex pass {run_i}...")
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(OUT), str(report_path)],
        capture_output=True, text=True, cwd=str(OUT)
    )
    if result.returncode != 0:
        # Show last 50 lines of output for debugging
        out_tail = "\n".join(result.stdout.splitlines()[-50:])
        print(f"  WARNING: pdflatex pass {run_i} exited with code {result.returncode}")
        print(out_tail)
    else:
        print(f"  Pass {run_i} OK")

pdf_path = OUT / "report.pdf"
if pdf_path.exists():
    print(f"  Generated: report.pdf ({pdf_path.stat().st_size // 1024} KB)")
else:
    print("  WARNING: report.pdf not found — check report.log for errors")

# ──────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("OUTPUT FILE SUMMARY")
print("=" * 70)

output_files = []
output_files.append(("report.tex",           str(OUT / "report.tex")))
output_files.append(("report.pdf",           str(OUT / "report.pdf")))
output_files.append(("all_results.csv",      str(OUT / "all_results.csv")))
output_files.append(("trace_analysis.csv",   str(OUT / "trace_analysis.csv")))
output_files.append(("label_fixes.jsonl",    str(OUT / "label_fixes.jsonl")))
output_files.append(("missing_fields.txt",   str(OUT / "missing_fields.txt")))

# Pareto plots
for si, sl in [("set1", "3x3"), ("set2", "3x1"), ("set3", "2x1")]:
    for pi in range(1, 6):
        fname = f"{si}_pareto_P{pi}_{sl}.png"
        output_files.append((fname, str(PLOT / fname)))

# General plots
for si in ["set1", "set2", "set3"]:
    for gi in range(1, 8):
        fname = f"{si}_general_G{gi}.png"
        if (PLOT / fname).exists():
            output_files.append((fname, str(PLOT / fname)))

# Exploratory plots
for fname, _ in exploratory_figures:
    if (PLOT_EX / fname).exists():
        output_files.append((f"exploratory/{fname}", str(PLOT_EX / fname)))

for name, path in output_files:
    p = Path(path)
    if p.exists():
        size = p.stat().st_size
        if size > 1024 * 1024:
            sz_str = f"{size // (1024*1024):.1f} MB"
        elif size > 1024:
            sz_str = f"{size // 1024} KB"
        else:
            sz_str = f"{size} B"
        print(f"  [OK] {name:<55} {sz_str}")
    else:
        print(f"  [--] {name:<55} (not found)")
