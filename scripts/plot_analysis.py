#!/usr/bin/env python3
"""
plot_analysis.py — Seaborn analysis plots from KernelBench sweep trajectories.

GPT-5.5 only. Three axes of variation:
  bench  → KernelBench (original prompt) or PopcornBench (popcorn prompt)
  level  → L1, L2, L3
  tier   → single | default | all   (tool availability given to the agent)

Usage:
    uv run python scripts/plot_analysis.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
RUNS_DIR = REPO_ROOT / "runs"
OUTPUT_DIR = REPO_ROOT / "docs" / "plots"

TARGET_MODEL_RAW = "gpt-5.5-priority"

LEVEL_ORDER = ["1", "2", "3"]
TIER_ORDER  = ["single", "default", "all"]
BENCH_ORDER = ["KernelBench", "PopcornBench"]
BENCH_SHORT = {"KernelBench": "KB", "PopcornBench": "PB"}

C_ORANGE = "#E6550D"
C_GRAY   = "#BDBDBD"

# Primary color axis: tier/run type
TIER_COLOR = {
    "single":  "#4C72B0",
    "default": "#DD8452",
    "all":     "#55A467",
}

# Secondary visual axes for bench distinction
BENCH_MARKER = {"KernelBench": "o", "PopcornBench": "s"}
BENCH_HATCH  = {"KernelBench": "",  "PopcornBench": "///"}

# Kept for heatmap y-label coloring
BENCH_COLOR = {"KernelBench": "#2171B5", "PopcornBench": C_ORANGE}

OUTCOME_ORDER = ["correct", "incorrect", "compile_fail", "error"]
OUTCOME_COLOR = {
    "correct":      "#55A467",
    "incorrect":    C_GRAY,
    "compile_fail": C_ORANGE,
    "error":        "#888888",
}

WARP_STALL_COLS = [
    "stall_long_scoreboard", "stall_mio_throttle", "stall_not_selected",
    "stall_barrier", "stall_wait", "stall_short_scoreboard",
    "stall_math_pipe_throttle", "stall_lg_throttle", "stall_drain",
]
WARP_STALL_LABELS = {
    "stall_long_scoreboard":    "Long scoreboard (L2/DRAM wait)",
    "stall_mio_throttle":       "MIO throttle (mem inst queue full)",
    "stall_short_scoreboard":   "Short scoreboard (shared mem/L1)",
    "stall_lg_throttle":        "LG throttle (global store queue)",
    "stall_math_pipe_throttle": "Math pipe throttle",
    "stall_not_selected":       "Not selected (scheduler)",
    "stall_wait":               "Wait (fixed-function unit)",
    "stall_barrier":            "Barrier (syncthreads)",
    "stall_drain":              "Drain (pipeline flush)",
}

# Stall categories with color palettes (memory → blue, compute → orange,
# scheduler → purple, sync → green, other → gray)
WARP_STALL_CATEGORY = {
    "stall_long_scoreboard":    "memory",
    "stall_mio_throttle":       "memory",
    "stall_short_scoreboard":   "memory",
    "stall_lg_throttle":        "memory",
    "stall_math_pipe_throttle": "compute",
    "stall_not_selected":       "scheduler",
    "stall_wait":               "scheduler",
    "stall_barrier":            "sync",
    "stall_drain":              "other",
}
_CATEGORY_PALETTES = {
    "memory":    ["#1565C0", "#1976D2", "#42A5F5", "#90CAF9"],
    "compute":   ["#E65100", "#FB8C00"],
    "scheduler": ["#6A1B9A", "#AB47BC"],
    "sync":      ["#2E7D32", "#66BB6A"],
    "other":     ["#757575"],
}

def _stall_colors(stall_cols: list[str]) -> dict[str, str]:
    """Assign colors within each category palette, cycling if needed."""
    counts: dict[str, int] = {}
    result = {}
    for col in stall_cols:
        cat = WARP_STALL_CATEGORY.get(col, "other")
        idx = counts.get(cat, 0)
        palette = _CATEGORY_PALETTES[cat]
        result[col] = palette[idx % len(palette)]
        counts[cat] = idx + 1
    return result

sns.set_theme(style="white", font="sans-serif", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titlelocation": "left",
})


def _hgrid(ax, alpha=0.18):
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#999", alpha=alpha)
    ax.set_axisbelow(True)


def _max_ticks(ax, n=6, axis="y"):
    loc = mticker.MaxNLocator(n, prune="both")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_locator(loc)


def _tier_legend(ax, **kw):
    elems = [Patch(facecolor=TIER_COLOR[t], label=t) for t in TIER_ORDER]
    ax.legend(handles=elems, fontsize=8, frameon=False,
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, **kw)


def _tier_bench_legend(ax, **kw):
    elems = [Patch(facecolor=TIER_COLOR[t], label=t) for t in TIER_ORDER]
    elems += [
        mlines.Line2D([], [], marker=BENCH_MARKER[b], color="#555",
                      linestyle="None", markersize=6, label=BENCH_SHORT[b])
        for b in BENCH_ORDER
    ]
    ax.legend(handles=elems, fontsize=8, frameon=False,
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
              title="tier  /  bench", title_fontsize=8, **kw)


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------

def infer_domain(name: str) -> str:
    n = name.lower()
    if "attention" in n:                                               return "Attention"
    if "conv" in n:                                                    return "Convolution"
    if any(x in n for x in ["matmul", "matrix", "gemm"]):             return "Linear Algebra"
    if any(x in n for x in ["batchnorm", "layernorm", "instancenorm",
                             "groupnorm", "_norm", "norm_"]):          return "Normalization"
    if "pool" in n:                                                    return "Pooling"
    if any(x in n for x in ["relu", "gelu", "sigmoid", "mish", "leaky",
                             "tanh", "softmax", "logsumexp"]):         return "Activation"
    if any(x in n for x in ["mlp", "ffn", "feedforward", "linear"]):  return "MLP/Linear"
    if any(x in n for x in ["sparse", "coo", "scatter"]):             return "Sparse"
    if any(x in n for x in ["all_reduce", "broadcast", "all_gather",
                             "reduce_scatter", "all_to_all", "pipeline",
                             "permut", "bitmask", "virtual", "p2p"]):  return "Distributed"
    return "Other"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _get_tier(run_name: str) -> str:
    if "_single"  in run_name: return "single"
    if "_default" in run_name: return "default"
    if "_all"     in run_name: return "all"
    return "unknown"


def load_dataframe() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RUNS_DIR / "**" / "*_trajectory.json"), recursive=True)):
        with open(f) as fh:
            d = json.load(fh)
        if d.get("model_name") != TARGET_MODEL_RAW:
            continue
        if d.get("outcome") == "in_progress":
            continue

        fp = str(f)
        variant = "popcorn" if "/popcorn/" in fp else "original" if "/original/" in fp else "default"
        bench = "KernelBench" if variant == "original" else "PopcornBench"
        level = str(d.get("level", "?"))
        tier  = _get_tier(d.get("run_name", ""))
        if tier == "unknown":
            # Fall back to the run-directory name in the path
            try:
                run_dir = Path(fp).relative_to(RUNS_DIR).parts[0]
                tier = _get_tier(run_dir)
            except ValueError:
                pass

        if level not in LEVEL_ORDER or tier not in TIER_ORDER:
            continue

        outcome = d.get("outcome", "unknown")
        fr  = d.get("final_result") or {}
        kls = fr.get("kernel_launch_stats") or {}
        sol = fr.get("sol_stats") or {}
        rs  = fr.get("roofline_stats") or {}
        ws  = rs.get("warp_stalls") or {}

        rt, ref_rt = fr.get("runtime"), fr.get("ref_runtime")
        speedup = (ref_rt / rt) if (rt and ref_rt and rt > 0 and ref_rt > 0) else None

        row: dict = {
            "bench": bench, "level": level, "tier": tier,
            "problem_id": d.get("problem_id"), "problem_name": d.get("problem_name", ""),
            "outcome": outcome,
            "compiled": fr.get("compiled", False),
            "speedup": speedup,
            "num_kernels":     kls.get("num_kernels"),
            "ref_num_kernels": kls.get("ref_num_kernels"),
            "fusion_ratio":    kls.get("fusion_ratio"),
            "sol_score":    sol.get("sol_score"),
            "bottleneck":   sol.get("bottleneck"),
            "dominant_pipe": sol.get("dominant_pipe"),
            "occupancy_pct": rs.get("occupancy_pct"),
            "dram_util_pct": rs.get("dram_utilization_pct"),
        }
        for col in WARP_STALL_COLS:
            row[col] = ws.get(col.replace("stall_", ""))
        row["domain"] = infer_domain(row["problem_name"])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Safety dedup: keep best result per (bench, tier, level, problem_id)
    outcome_rank = {"correct": 0, "incorrect": 1, "compile_fail": 2, "error": 3}
    df["_or"] = df["outcome"].map(outcome_rank).fillna(9)
    df["_ns"] = df["speedup"].fillna(0).mul(-1)
    df = (
        df.sort_values(["_or", "_ns"])
        .groupby(["bench", "tier", "level", "problem_id"], sort=False)
        .first().reset_index()
        .drop(columns=["_or", "_ns"])
    )

    df["bench"] = pd.Categorical(df["bench"], categories=BENCH_ORDER, ordered=True)
    df["tier"]  = pd.Categorical(df["tier"],  categories=TIER_ORDER,  ordered=True)
    df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)
    return df


# ---------------------------------------------------------------------------
# Shared: level-facet figure factory
# ---------------------------------------------------------------------------

def _level_fig(ncols=3, height=5.0, width_per=5.5, sharey=False):
    return plt.subplots(1, ncols, figsize=(width_per * ncols, height),
                        sharey=sharey, squeeze=False)


# ---------------------------------------------------------------------------
# Plot 1 — Outcome breakdown (two versions: with / without errors)
# ---------------------------------------------------------------------------

def plot1_outcomes(df: pd.DataFrame, include_errors: bool = True) -> None:
    suffix = "" if include_errors else "_no_errors"
    plot_df = df if include_errors else df[df["outcome"] != "error"].copy()
    outcomes_shown = OUTCOME_ORDER if include_errors else [o for o in OUTCOME_ORDER if o != "error"]

    bench_vals = [b for b in BENCH_ORDER if b in plot_df["bench"].values]

    bar_w     = 0.33
    group_gap = 0.18

    fig, axes = _level_fig(height=5.0)

    for ax, lvl in zip(axes[0], LEVEL_ORDER):
        lvl_df = plot_df[plot_df["level"] == lvl]
        tier_vals = [t for t in TIER_ORDER if t in lvl_df["tier"].values]

        for ti, tier in enumerate(tier_vals):
            tier_x0 = ti * (len(bench_vals) * bar_w + group_gap)
            for bi, bench in enumerate(bench_vals):
                sub = lvl_df[(lvl_df["tier"] == tier) & (lvl_df["bench"] == bench)]
                if sub.empty:
                    continue
                total = len(sub)
                xpos  = tier_x0 + bi * bar_w
                bottom = 0.0
                for outcome in outcomes_shown:
                    n    = (sub["outcome"] == outcome).sum()
                    frac = n / total
                    if frac == 0:
                        continue
                    hatch = BENCH_HATCH[bench]
                    ec    = "#333" if hatch else "white"
                    ax.bar(xpos, frac, bottom=bottom, width=bar_w,
                           color=OUTCOME_COLOR[outcome],
                           hatch=hatch, edgecolor=ec, linewidth=0.4)
                    if frac > 0.09:
                        ax.text(xpos, bottom + frac / 2, f"{frac:.0%}",
                                ha="center", va="center", fontsize=7,
                                color="white", fontweight="bold")
                    bottom += frac
                ax.text(xpos, 1.03, f"n={total}", ha="center", va="bottom",
                        fontsize=6.5, color="#777")

            tier_center = tier_x0 + (len(bench_vals) - 1) * bar_w / 2
            ax.annotate(tier, xy=(tier_center, -0.04), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=9, fontweight="bold")

        ax.set_xticks([])
        ax.set_ylim(0, 1.13)
        ax.set_title(f"Level {lvl}")
        ax.set_ylabel("Fraction of problems" if ax is axes[0][0] else "")
        sns.despine(ax=ax)

    # Single unified legend: outcomes + bench hatch style
    legend_elems = [
        Patch(facecolor=OUTCOME_COLOR[o], label=o, edgecolor="white")
        for o in outcomes_shown
    ]
    legend_elems.append(Patch(facecolor="#888", label="─── KernelBench", edgecolor="white"))
    legend_elems.append(Patch(facecolor="#888", hatch="///", label="/// PopcornBench", edgecolor="#333"))

    axes[0][2].legend(handles=legend_elems, fontsize=8, frameon=False,
                      loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
                      title="outcome / bench", title_fontsize=8)

    fig.suptitle("Outcome breakdown by tier" + ("" if include_errors else " (errors excluded)"),
                 fontsize=12, fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"plot1_outcomes{suffix}.png")
    plt.close(fig)
    print(f"Saved plot1_outcomes{suffix}.png")


# ---------------------------------------------------------------------------
# Plot 2 — Speedup distribution (tier-colored, bench on x-tick)
# ---------------------------------------------------------------------------

def plot2_speedup_dist(df: pd.DataFrame) -> None:
    sub = df.copy()
    # Non-correct outcomes contribute 0× speedup; clip to 0.01 so log scale renders them
    sub["speedup"] = sub["speedup"].where(sub["outcome"] == "correct", 0.0).fillna(0.0)
    sub["speedup"] = sub["speedup"].clip(lower=0.01)
    sub["bench_short"] = sub["bench"].map(BENCH_SHORT)
    sub["group"] = [f"{t}\n{b}" for t, b in zip(sub["tier"], sub["bench_short"])]

    fig, axes = _level_fig(height=5.5, sharey=True)

    for ax, lvl in zip(axes[0], LEVEL_ORDER):
        lvl_df = sub[sub["level"] == lvl]
        group_order = [
            f"{t}\n{BENCH_SHORT[b]}"
            for t in TIER_ORDER for b in BENCH_ORDER
            if not lvl_df[(lvl_df["tier"] == t) & (lvl_df["bench_short"] == BENCH_SHORT[b])].empty
        ]
        palette = {g: TIER_COLOR[g.split("\n")[0]] for g in group_order}

        sns.violinplot(
            data=lvl_df, x="group", y="speedup", hue="group",
            order=group_order, hue_order=group_order,
            inner=None, cut=0, density_norm="count",
            palette=palette, alpha=0.40, ax=ax, legend=False,
        )

        for xi, grp in enumerate(group_order):
            tier, bs = grp.split("\n")
            s = lvl_df[(lvl_df["tier"] == tier) & (lvl_df["bench_short"] == bs)]["speedup"]
            if s.empty:
                continue
            med = s.median()
            ax.plot(xi, med, "o", color=TIER_COLOR[tier],
                    markersize=6, zorder=5,
                    markeredgecolor="white", markeredgewidth=1.2)
            ax.text(xi + 0.12, med, f"{med:.1f}×",
                    va="center", fontsize=7.5, color=TIER_COLOR[tier], fontweight="bold")

        ax.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}×"))
        _max_ticks(ax, n=5)
        _hgrid(ax)
        ax.set_title(f"Level {lvl}")
        ax.set_xlabel("")
        ax.set_ylabel("Speedup (log)" if ax is axes[0][0] else "")
        sns.despine(ax=ax, left=True)

    _tier_legend(axes[0][2])
    fig.suptitle("Speedup distribution by tier and benchmark (incorrect / error = 0×)",
                 fontsize=12, fontweight="bold", x=0.01, ha="left", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot2_speedup_dist.png")
    plt.close(fig)
    print("Saved plot2_speedup_dist.png")


# ---------------------------------------------------------------------------
# Plot 3 — Per-problem speedup heatmap (best across tiers)
# ---------------------------------------------------------------------------

def plot3_speedup_heatmap(df: pd.DataFrame) -> None:
    sub = df[(df["outcome"] == "correct") & df["speedup"].notna()].copy()
    best = (
        sub.sort_values("speedup", ascending=False)
        .groupby(["bench", "level", "problem_id"])
        .first().reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 3.5))
    for ax, lvl in zip(axes, LEVEL_ORDER):
        lvl_df = best[best["level"] == lvl]
        pivot = lvl_df.pivot_table(
            index="bench", columns="problem_id", values="speedup", aggfunc="max"
        ).reindex(BENCH_ORDER).reindex(sorted(lvl_df["problem_id"].unique()), axis=1)
        log_vals = np.log2(pivot.values.astype(float))

        sns.heatmap(
            log_vals, ax=ax,
            xticklabels=pivot.columns,
            yticklabels=["KernelBench", "PopcornBench"],
            cmap="Blues", vmin=0, vmax=np.nanmax(log_vals) or 1.0,
            linewidths=0, cbar=(ax is axes[-1]),
            cbar_kws={"label": "log₂(speedup)", "shrink": 0.7},
            mask=np.isnan(log_vals),
        )
        ax.set_title(f"Level {lvl}", fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("Problem ID")
        ax.set_ylabel("")
        n = pivot.shape[1]
        step = max(1, n // 10)
        ax.set_xticks(np.arange(0, n, step) + 0.5)
        ax.set_xticklabels(pivot.columns[::step], rotation=90, fontsize=7)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("Best speedup per problem (across tiers) — deeper blue = faster",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot3_speedup_heatmap.png")
    plt.close(fig)
    print("Saved plot3_speedup_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 4 — SOL score (incorrects/errors weighted as 0)
# ---------------------------------------------------------------------------

def plot4_sol_score(df: pd.DataFrame) -> None:
    sub = df.copy()
    # Non-correct outcomes contribute 0 to SOL score
    sub["sol_score"] = sub["sol_score"].where(sub["outcome"] == "correct", 0.0).fillna(0.0)

    if sub.empty:
        print("No data — skipping plot4"); return

    sub["bench_short"] = sub["bench"].map(BENCH_SHORT)
    sub["group"] = [f"{t}\n{b}" for t, b in zip(sub["tier"], sub["bench_short"])]
    group_order = [
        f"{t}\n{BENCH_SHORT[b]}"
        for t in TIER_ORDER for b in BENCH_ORDER
        if not sub[(sub["tier"] == t) & (sub["bench_short"] == BENCH_SHORT[b])].empty
    ]
    palette = {g: TIER_COLOR[g.split("\n")[0]] for g in group_order}

    fig, axes = _level_fig(height=5.0, sharey=True)
    for ax, lvl in zip(axes[0], LEVEL_ORDER):
        lvl_df = sub[sub["level"] == lvl]
        lvl_order = [g for g in group_order if not lvl_df[lvl_df["group"] == g].empty]

        sns.boxplot(
            data=lvl_df, x="group", y="sol_score", hue="group",
            order=lvl_order, hue_order=lvl_order,
            palette=palette, ax=ax, width=0.45,
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
            linewidth=0.8, legend=False,
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Level {lvl}")
        ax.set_ylabel("SOL score (0–1)" if ax is axes[0][0] else "")
        ax.set_xlabel("")
        _hgrid(ax)
        sns.despine(ax=ax, left=(ax is not axes[0][0]))

    _tier_legend(axes[0][2])
    fig.suptitle("SOL score by tier (incorrect / error treated as 0)",
                 fontsize=12, fontweight="bold", x=0.01, ha="left", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot4_sol_score.png")
    plt.close(fig)
    print("Saved plot4_sol_score.png")


# ---------------------------------------------------------------------------
# Plot 5 — Kernel fusion ratio vs speedup
# ---------------------------------------------------------------------------

def plot5_fusion_vs_speedup(df: pd.DataFrame) -> None:
    sub = df[(df["outcome"] == "correct") & df["speedup"].notna() & df["fusion_ratio"].notna()].copy()
    if sub.empty:
        print("No fusion data — skipping plot5"); return

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for tier in TIER_ORDER:
        for bench in BENCH_ORDER:
            s = sub[(sub["tier"] == tier) & (sub["bench"] == bench)]
            if s.empty:
                continue
            ax.scatter(s["fusion_ratio"], s["speedup"],
                       color=TIER_COLOR[tier], marker=BENCH_MARKER[bench],
                       alpha=0.55, s=28, edgecolors="none")

    ax.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=1)
    ax.axvline(1.0, color=C_GRAY, linestyle=":",  linewidth=1)
    def _lfmt(v, _):
        if v >= 10: return f"{int(v)}×"
        if v >= 1:  return f"{v:.1f}×"
        return f"{v:.2f}×"

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_lfmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_lfmt))
    _max_ticks(ax, n=5)
    _max_ticks(ax, n=4, axis="x")
    ax.tick_params(axis="x", rotation=30)

    ax.set_xlabel("Kernel fusion ratio\n(ref kernels / custom kernels)")
    ax.set_ylabel("Speedup vs reference")
    _hgrid(ax)
    ax.set_title("Fusion ratio vs speedup")
    sns.despine(ax=ax)

    _tier_bench_legend(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot5_fusion_vs_speedup.png")
    plt.close(fig)
    print("Saved plot5_fusion_vs_speedup.png")


# ---------------------------------------------------------------------------
# Plot 6 — Occupancy vs DRAM utilization
# ---------------------------------------------------------------------------

def plot6_occupancy_dram(df: pd.DataFrame) -> None:
    sub = df[df["occupancy_pct"].notna() & df["dram_util_pct"].notna()].copy()
    sub["occupancy_pct"] = sub["occupancy_pct"].clip(upper=100)
    if sub.empty:
        print("No roofline data — skipping plot6"); return

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for tier in TIER_ORDER:
        for bench in BENCH_ORDER:
            s = sub[(sub["tier"] == tier) & (sub["bench"] == bench)]
            if s.empty:
                continue
            sz = s["speedup"].fillna(1.0).clip(lower=0.1).apply(
                lambda x: max(15, min(120, x * 30))
            )
            ax.scatter(s["occupancy_pct"], s["dram_util_pct"],
                       color=TIER_COLOR[tier], marker=BENCH_MARKER[bench],
                       alpha=0.50, s=sz, edgecolors="none")

    ax.set_xlabel("Warp occupancy (%)")
    ax.set_ylabel("DRAM utilization (%)")
    _max_ticks(ax, n=5)
    _max_ticks(ax, n=5, axis="x")
    _hgrid(ax)
    ax.set_title("Occupancy vs DRAM utilization  (size ∝ speedup)")
    sns.despine(ax=ax)

    _tier_bench_legend(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot6_occupancy_dram.png")
    plt.close(fig)
    print("Saved plot6_occupancy_dram.png")


# ---------------------------------------------------------------------------
# Plot 7 — Warp stall breakdown
# ---------------------------------------------------------------------------

def plot7_warp_stalls(df: pd.DataFrame) -> None:
    stall_cols = [c for c in WARP_STALL_COLS if c in df.columns]
    sub = df[df[stall_cols].notna().any(axis=1)].copy()
    if sub.empty:
        print("No warp stall data — skipping plot7"); return

    raw = sub[stall_cols].astype(float).fillna(0.0)
    row_sums = raw.sum(axis=1).replace(0, np.nan)
    norm = raw.div(row_sums, axis=0).fillna(0.0)
    norm["bench"] = sub["bench"].values
    norm["level"] = sub["level"].values
    norm["tier"]  = sub["tier"].values

    agg = norm.groupby(["bench", "level", "tier"])[stall_cols].mean().reset_index()

    # Order: tier-first, then bench, then level
    order = [
        (b, l, t)
        for t in TIER_ORDER
        for b in BENCH_ORDER
        for l in LEVEL_ORDER
        if len(agg[(agg["bench"] == b) & (agg["level"] == l) & (agg["tier"] == t)]) > 0
    ]
    agg_ordered = pd.DataFrame([
        agg[(agg["bench"] == b) & (agg["level"] == l) & (agg["tier"] == t)].iloc[0]
        for b, l, t in order
    ])

    col_colors = _stall_colors(stall_cols)

    # Order stalls: dominant first (by mean share across all rows), then by category
    mean_share = agg_ordered[stall_cols].mean()
    dominant = mean_share.idxmax()
    cat_order = ["memory", "compute", "scheduler", "sync", "other"]
    other_cols = sorted(
        [c for c in stall_cols if c != dominant],
        key=lambda c: (cat_order.index(WARP_STALL_CATEGORY.get(c, "other")), c),
    )
    ordered_cols = [dominant] + other_cols

    row_labels = [f"{t}  {BENCH_SHORT[b]} L{l}" for b, l, t in order]

    fig, ax = plt.subplots(figsize=(14, max(4, len(order) * 0.6)))
    y    = np.arange(len(order))
    left = np.zeros(len(order))

    for col in ordered_cols:
        if col not in agg_ordered.columns:
            continue
        vals = agg_ordered[col].values.astype(float)
        ax.barh(y, vals, left=left, color=col_colors[col], height=0.62, linewidth=0,
                label=WARP_STALL_LABELS.get(col, col))
        left += vals

    # Dividers between tier groups
    for i in range(1, len(order)):
        if order[i][2] != order[i - 1][2]:
            ax.axhline(i - 0.5, color="#444", linewidth=1.2, linestyle="--")

    ax.set_yticks(y)
    ax.set_yticklabels(row_labels, fontsize=8)
    for tick, (b, l, t) in zip(ax.get_yticklabels(), order):
        tick.set_color(TIER_COLOR[t])

    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    _max_ticks(ax, n=5, axis="x")
    ax.set_xlabel("Share of warp stall cycles")
    ax.set_title("Warp stall breakdown by tier")
    sns.despine(ax=ax, left=True)

    # Legend grouped by category
    cat_labels = {"memory": "── Memory", "compute": "── Compute",
                  "scheduler": "── Scheduler", "sync": "── Sync", "other": "── Other"}
    legend_handles = []
    seen_cats: set[str] = set()
    for col in ordered_cols:
        cat = WARP_STALL_CATEGORY.get(col, "other")
        if cat not in seen_cats:
            legend_handles.append(Patch(color="none", label=cat_labels[cat]))
            seen_cats.add(cat)
        legend_handles.append(Patch(color=col_colors[col],
                                    label=f"  {WARP_STALL_LABELS.get(col, col)}"))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8, frameon=False, title="stall type", title_fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot7_warp_stalls.png")
    plt.close(fig)
    print("Saved plot7_warp_stalls.png")


# ---------------------------------------------------------------------------
# Plot 7b — Warp stall share per agent tier (3-column, one per stall type)
# ---------------------------------------------------------------------------

_FAST_STALL_COLS = ["stall_long_scoreboard", "stall_short_scoreboard", "stall_mio_throttle"]
_FAST_STALL_TITLES = {
    "stall_long_scoreboard":  "Long scoreboard\n(L2/DRAM wait)",
    "stall_short_scoreboard": "Short scoreboard\n(shared mem / L1)",
    "stall_mio_throttle":     "MIO throttle\n(mem inst queue full)",
}

def plot7b_warp_stalls_per_agent(df: pd.DataFrame) -> None:
    stall_cols = [c for c in _FAST_STALL_COLS if c in df.columns]
    sub = df[df[stall_cols].notna().any(axis=1)].copy()
    if sub.empty:
        print("No warp stall data — skipping plot7b"); return

    raw = sub[stall_cols].astype(float).fillna(0.0)
    row_sums = raw.sum(axis=1).replace(0, np.nan)
    norm = raw.div(row_sums, axis=0).fillna(0.0)
    for col in stall_cols:
        sub[col] = norm[col].values

    benches = [b for b in BENCH_ORDER if b in sub["bench"].values]
    n_rows = len(benches)
    n_cols = len(stall_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.2 * n_rows),
                             sharey="row", sharex="col")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    bar_w = 0.22
    x = np.arange(len(LEVEL_ORDER))

    for row_i, bench in enumerate(benches):
        bsub = sub[sub["bench"] == bench]
        for col_i, col in enumerate(stall_cols):
            ax = axes[row_i, col_i]
            for ti, tier in enumerate(TIER_ORDER):
                tsub = bsub[bsub["tier"] == tier]
                means = [
                    tsub[tsub["level"] == lvl][col].mean()
                    for lvl in LEVEL_ORDER
                ]
                offset = (ti - 1) * bar_w
                ax.bar(x + offset, means, width=bar_w,
                       color=TIER_COLOR[tier], label=tier if (row_i == 0 and col_i == 0) else "_")

            ax.set_xticks(x)
            ax.set_xticklabels([f"L{l}" for l in LEVEL_ORDER])
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax.set_ylim(0, 1)
            _max_ticks(ax, n=4, axis="y")
            sns.despine(ax=ax)

            if row_i == 0:
                ax.set_title(_FAST_STALL_TITLES.get(col, col), fontsize=9, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(f"{bench}\nmean share", fontsize=8)

    fig.legend(
        handles=[Patch(color=TIER_COLOR[t], label=t) for t in TIER_ORDER],
        loc="upper right", bbox_to_anchor=(1.0, 1.02),
        ncol=3, fontsize=9, frameon=False, title="agent tier", title_fontsize=9,
    )
    fig.suptitle("Warp stall share by stall type and agent tier", fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot7b_warp_stalls_per_agent.png")
    plt.close(fig)
    print("Saved plot7b_warp_stalls_per_agent.png")


# ---------------------------------------------------------------------------
# Plot 8 — Scorecard (ordered by tier; incorrects count as 0×)
# ---------------------------------------------------------------------------

def plot8_scorecard(df: pd.DataFrame) -> None:
    def agg_group(g):
        total = len(g)
        compiled     = g["compiled"].sum()
        correct_mask = g["outcome"] == "correct"
        correct      = correct_mask.sum()
        # Incorrects / errors contribute 0× to the speedup pool
        all_speeds   = g["speedup"].where(correct_mask, 0.0).fillna(0.0)
        return pd.Series({
            "compile_rate":   compiled / total if total else 0.0,
            "correct_rate":   correct  / total if total else 0.0,
            "median_speedup": float(all_speeds.median()) if total else np.nan,
            "n": total,
        })

    sc = (
        df.groupby(["bench", "level", "tier"])
        .apply(agg_group, include_groups=False)
        .reset_index()
    )

    # Order: tier-first, then level, with KB/PB interleaved within each level
    order = [
        (b, l, t)
        for t in TIER_ORDER
        for l in LEVEL_ORDER
        for b in BENCH_ORDER
        if len(sc[(sc["bench"] == b) & (sc["level"] == l) & (sc["tier"] == t)]) > 0
    ]
    sc_ordered = pd.DataFrame([
        sc[(sc["bench"] == b) & (sc["level"] == l) & (sc["tier"] == t)].iloc[0]
        for b, l, t in order
    ])

    bar_colors = [TIER_COLOR[t] for b, l, t in order]
    # KB solid, PB lighter
    bar_alphas  = [1.0 if b == "KernelBench" else 0.5 for b, l, t in order]

    def _n(row): return int(row["n"]) if pd.notna(row["n"]) else 0
    ylabels = [
        f"{t}  L{l}  {BENCH_SHORT[b]}  (n={_n(sc_ordered.iloc[i])})"
        for i, (b, l, t) in enumerate(order)
    ]

    metrics = [
        ("compile_rate",   "Compile rate",     None),
        ("correct_rate",   "Correctness rate",  None),
        ("median_speedup", "Median speedup",    1.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, max(4.5, len(order) * 0.6)))
    x = np.arange(len(order))

    # Tier boundary positions for dividers
    tier_changes = [i for i in range(1, len(order)) if order[i][2] != order[i - 1][2]]

    for ax, (col, title, ref) in zip(axes, metrics):
        vals = sc_ordered[col].values.astype(float)
        for j, (v, color, alpha) in enumerate(zip(vals, bar_colors, bar_alphas)):
            ax.barh(x[j], v, color=color, height=0.62, linewidth=0, alpha=alpha)
            if pd.notna(v) and v > 0:
                fmt    = f"{v:.2f}×" if col == "median_speedup" else f"{v:.0%}"
                offset = 0.04 if col == "median_speedup" else 0.01
                ax.text(v + offset, x[j], fmt, va="center", fontsize=7.5,
                        color=color, fontweight="bold", alpha=min(alpha + 0.3, 1.0))

        ax.set_yticks(x)
        ax.set_yticklabels(ylabels if ax is axes[0] else [""] * len(order), fontsize=7.5)
        if ax is axes[0]:
            for tick, color, alpha in zip(ax.get_yticklabels(), bar_colors, bar_alphas):
                tick.set_color(color)

        if col != "median_speedup":
            ax.set_xlim(0, 1.18)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        else:
            xmax = sc_ordered[col].dropna().max()
            ax.set_xlim(0, max(xmax * 1.3, 2.0))

        if ref is not None:
            ax.axvline(ref, color=C_GRAY, linestyle="--", linewidth=0.9)

        for i in tier_changes:
            ax.axhline(i - 0.5, color="#444", linewidth=0.9, linestyle="--")

        _max_ticks(ax, n=5, axis="x")
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        sns.despine(ax=ax, left=(ax is not axes[0]))

    # Tier + bench legend
    legend_elems = [Patch(facecolor=TIER_COLOR[t], label=t) for t in TIER_ORDER]
    legend_elems += [
        Patch(facecolor="#888", alpha=1.0, label="KB (solid)"),
        Patch(facecolor="#888", alpha=0.5, label="PB (faded)"),
    ]
    axes[2].legend(handles=legend_elems, fontsize=8, frameon=False,
                   loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
                   title="tier / bench", title_fontsize=8)

    fig.suptitle("Scorecard — incorrects count as 0×",
                 fontsize=12, fontweight="bold", x=0.01, ha="left", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot8_scorecard.png")
    plt.close(fig)
    print("Saved plot8_scorecard.png")


# ---------------------------------------------------------------------------
# Plot 9 — Outcomes by domain (bench × level facets, best result across tiers)
# ---------------------------------------------------------------------------

def plot9_outcomes_domain(df: pd.DataFrame) -> None:
    # Best result per (bench, level, problem_id) across tiers for domain analysis
    best = (
        df.sort_values(["bench", "level", "problem_id"])
        .groupby(["bench", "level", "problem_id"])
        .apply(lambda g: g.sort_values(
            ["outcome", "speedup"],
            key=lambda s: s.map({"correct":0,"incorrect":1,"compile_fail":2,"error":3})
                           if s.name == "outcome" else -s.fillna(0)
        ).iloc[0])
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    panels = [(b, l) for b in BENCH_ORDER for l in LEVEL_ORDER]

    for ax, (bench, lvl) in zip(axes.flat, panels):
        sub = best[(best["bench"] == bench) & (best["level"] == lvl)]
        d_out = sub.groupby(["domain", "outcome"]).size().reset_index(name="n")
        dtotals = d_out.groupby("domain")["n"].sum().rename("total")
        d_out = d_out.join(dtotals, on="domain")
        d_out["frac"] = d_out["n"] / d_out["total"]

        pivot = d_out.pivot_table(index="domain", columns="outcome",
                                   values="frac", fill_value=0)
        for col in OUTCOME_ORDER:
            if col not in pivot.columns:
                pivot[col] = 0.0
        pivot = pivot[[c for c in OUTCOME_ORDER if c in pivot.columns]]
        pivot = pivot.sort_values("correct", ascending=False)

        x = np.arange(len(pivot))
        bottom = np.zeros(len(pivot))
        for outcome in OUTCOME_ORDER:
            vals = pivot[outcome].values
            ax.bar(x, vals, bottom=bottom, color=OUTCOME_COLOR[outcome],
                   width=0.62, linewidth=0)
            for j, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0.10:
                    ax.text(j, b + v / 2, f"{v:.0%}", ha="center", va="center",
                            fontsize=7.5, color="white", fontweight="bold")
            bottom += vals

        for j, domain in enumerate(pivot.index):
            n = int(dtotals[domain])
            ax.text(j, 1.03, f"n={n}", ha="center", va="bottom", fontsize=7, color="#777")

        color = BENCH_COLOR[bench]
        ax.set_title(f"{bench} L{lvl}", fontsize=10, fontweight="bold", color=color, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylim(0, 1.16)
        ax.set_ylabel("Fraction" if ax in axes[:, 0] else "")
        sns.despine(ax=ax, left=True)

    # Outcome color key at bottom
    for i, (outcome, color) in enumerate(OUTCOME_COLOR.items()):
        fig.text(0.15 + i * 0.19, 0.01, f"■ {outcome}", color=color,
                 fontsize=9, fontweight="bold", ha="center")

    fig.suptitle("Outcomes by domain (best result across tiers)",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUTPUT_DIR / "plot9_outcomes_domain.png")
    plt.close(fig)
    print("Saved plot9_outcomes_domain.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading trajectories from {RUNS_DIR} …")
    df = load_dataframe()
    print(f"Loaded {len(df)} trajectories  |  outcomes: {dict(df['outcome'].value_counts())}")
    print("Counts by (bench, level, tier):")
    for (b, l, t), n in df.groupby(["bench", "level", "tier"], observed=True).size().items():
        print(f"  {b} L{l} {t}: {n}")
    print()

    plot1_outcomes(df, include_errors=True)
    plot1_outcomes(df, include_errors=False)
    plot2_speedup_dist(df)
    plot3_speedup_heatmap(df)
    plot4_sol_score(df)
    plot5_fusion_vs_speedup(df)
    plot6_occupancy_dram(df)
    plot7_warp_stalls(df)
    plot7b_warp_stalls_per_agent(df)
    plot8_scorecard(df)
    plot9_outcomes_domain(df)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
