#!/usr/bin/env python3
"""
plot_sweep_results.py

Creates thesis-grade diagrams for distributed early-exit + selective offloading sweeps.

Input:
  - results.csv (one row per sweep)
Output:
  - a folder of PNG/PDF plots

Notes:
  - Uses matplotlib only (no seaborn).
  - Does NOT assume energy exists for every row.
  - Computes Pareto frontiers for accuracy vs latency and accuracy vs energy/CO2.

Author: you + ChatGPT
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def savefig_all(fig, out_dir: str, name: str, formats: List[str], dpi: int = 200) -> None:
    for fmt in formats:
        p = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def pareto_front(df: pd.DataFrame, x: str, y: str, maximize_y: bool = True) -> pd.DataFrame:
    """
    2D Pareto frontier.
    - minimize x
    - maximize y (default)

    Returns subset of df that is non-dominated.
    """
    d = df[[x, y]].dropna().copy()
    if d.empty:
        return df.iloc[0:0]

    # Sort by x asc, then y desc (if maximize_y)
    d = d.sort_values(by=[x, y], ascending=[True, False if maximize_y else True])

    keep_idx = []
    best_y = -np.inf if maximize_y else np.inf

    for idx, row in d.iterrows():
        yv = row[y]
        if maximize_y:
            if yv > best_y:
                best_y = yv
                keep_idx.append(idx)
        else:
            if yv < best_y:
                best_y = yv
                keep_idx.append(idx)

    return df.loc[keep_idx].sort_values(by=x)


def annotate_topk(fig, ax, df: pd.DataFrame, x: str, y: str, k: int = 8, label_cols=("thr0", "thr1")):
    """
    Annotate top-k points by y (highest) with thr0/thr1 labels.
    Keeps clutter low by limiting to k.
    """
    d = df[[x, y, *label_cols]].dropna().copy()
    if d.empty:
        return
    d = d.sort_values(by=y, ascending=False).head(k)

    for _, row in d.iterrows():
        label = f"{label_cols[0]}={row[label_cols[0]]:.2f}, {label_cols[1]}={row[label_cols[1]]:.2f}"
        ax.annotate(
            label,
            (row[x], row[y]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )


# -----------------------------
# Plot builders
# -----------------------------
def plot_tradeoff_accuracy_latency(
    df: pd.DataFrame,
    out_dir: str,
    formats: List[str],
    latency_col: str,
    title_suffix: str,
    name_suffix: str,
):
    """
    Scatter: Accuracy vs Latency (avg or p95).
    Includes per-criterion plot + overall plot with Pareto frontier.
    """
    # Overall
    fig, ax = plt.subplots()
    ax.set_title(f"Accuracy vs {title_suffix} ({latency_col})")
    ax.set_xlabel(f"{title_suffix} (ms)")
    ax.set_ylabel("Accuracy")

    for crit in sorted(df["criterion"].dropna().unique()):
        d = df[df["criterion"] == crit].copy()
        ax.scatter(d[latency_col], d["accuracy"], label=str(crit), alpha=0.9)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    # Pareto frontier (overall)
    frontier = pareto_front(df, x=latency_col, y="accuracy", maximize_y=True)
    if not frontier.empty:
        ax.plot(frontier[latency_col], frontier["accuracy"], linewidth=2)
        annotate_topk(fig, ax, frontier, x=latency_col, y="accuracy", k=8)

    savefig_all(fig, out_dir, f"tradeoff_accuracy_vs_{name_suffix}", formats)

    # Per-criterion separate figures (cleaner for thesis)
    for crit in sorted(df["criterion"].dropna().unique()):
        d = df[df["criterion"] == crit].copy()
        fig, ax = plt.subplots()
        ax.set_title(f"{crit}: Accuracy vs {title_suffix}")
        ax.set_xlabel(f"{title_suffix} (ms)")
        ax.set_ylabel("Accuracy")
        ax.scatter(d[latency_col], d["accuracy"], alpha=0.9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        frontier_c = pareto_front(d, x=latency_col, y="accuracy", maximize_y=True)
        if not frontier_c.empty:
            ax.plot(frontier_c[latency_col], frontier_c["accuracy"], linewidth=2)
            annotate_topk(fig, ax, frontier_c, x=latency_col, y="accuracy", k=8)

        savefig_all(fig, out_dir, f"{crit}_tradeoff_accuracy_vs_{name_suffix}", formats)


def plot_exit_offload_bars(df: pd.DataFrame, out_dir: str, formats: List[str]):
    """
    For each criterion, show exit0/exit1/offload percentages (bar chart across sweeps).
    Also provide an "overall distribution" view (scatter).
    """
    for crit in sorted(df["criterion"].dropna().unique()):
        d = df[df["criterion"] == crit].copy().sort_values(by=["thr0", "thr1"])
        xlabels = [f"{r.thr0:.2f}/{r.thr1:.2f}" for r in d.itertuples(index=False)]

        fig, ax = plt.subplots(figsize=(max(10, len(d) * 0.35), 5))
        ax.set_title(f"{crit}: Exit/Offload Rates per (thr0/thr1)")
        ax.set_xlabel("thr0/thr1")
        ax.set_ylabel("Percentage (%)")

        x = np.arange(len(d))
        width = 0.25

        ax.bar(x - width, d["exit0_pct"], width=width, label="exit0_pct")
        ax.bar(x, d["exit1_pct"], width=width, label="exit1_pct")
        ax.bar(x + width, d["offload_pct"], width=width, label="offload_pct")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=60, ha="right")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()

        savefig_all(fig, out_dir, f"{crit}_exit_offload_rates", formats)

    # Scatter overview: offload_pct vs accuracy
    fig, ax = plt.subplots()
    ax.set_title("Accuracy vs Offload Rate")
    ax.set_xlabel("Offload (%)")
    ax.set_ylabel("Accuracy")
    for crit in sorted(df["criterion"].dropna().unique()):
        d = df[df["criterion"] == crit]
        ax.scatter(d["offload_pct"], d["accuracy"], label=str(crit), alpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    savefig_all(fig, out_dir, "accuracy_vs_offload_pct", formats)


def plot_latency_decomposition(df: pd.DataFrame, out_dir: str, formats: List[str], which: str = "avg"):
    """
    Latency decomposition plots:
      - client_rtt_{avg/p95} vs total_edge_{avg/p95}
      - edge_infer_{avg/p95} vs total_edge_{avg/p95}
      - (optional) offload-only components: edge_cloud_rtt_{avg/p95}, cloud_compute_{avg/p95}
    """
    assert which in ("avg", "p95")

    client = f"client_rtt_{which}"
    total_edge = f"total_edge_{which}"
    edge_infer = f"edge_infer_{which}"
    edge_cloud = f"edge_cloud_rtt_{which}"
    cloud_compute = f"cloud_compute_{which}"
    total_cloud = f"total_cloud_{which}"

    # Client vs Edge total
    fig, ax = plt.subplots()
    ax.set_title(f"Client RTT vs Edge Total ({which})")
    ax.set_xlabel(f"{total_edge} (ms)")
    ax.set_ylabel(f"{client} (ms)")
    for crit in sorted(df["criterion"].dropna().unique()):
        d = df[df["criterion"] == crit]
        ax.scatter(d[total_edge], d[client], label=str(crit), alpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    savefig_all(fig, out_dir, f"client_vs_total_edge_{which}", formats)

    # Edge infer vs total edge
    fig, ax = plt.subplots()
    ax.set_title(f"Edge Infer vs Total Edge ({which})")
    ax.set_xlabel(f"{edge_infer} (ms)")
    ax.set_ylabel(f"{total_edge} (ms)")
    for crit in sorted(df["criterion"].dropna().unique()):
        d = df[df["criterion"] == crit]
        ax.scatter(d[edge_infer], d[total_edge], label=str(crit), alpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    savefig_all(fig, out_dir, f"edge_infer_vs_total_edge_{which}", formats)

    # Offload components (only where offload exists)
    fig, ax = plt.subplots()
    ax.set_title(f"Edge↔Cloud RTT vs Cloud Compute ({which}) [rows with offload measurements]")
    ax.set_xlabel(f"{edge_cloud} (ms)")
    ax.set_ylabel(f"{cloud_compute} (ms)")
    for crit in sorted(df["criterion"].dropna().unique()):
        d = df[df["criterion"] == crit].dropna(subset=[edge_cloud, cloud_compute])
        if len(d) == 0:
            continue
        ax.scatter(d[edge_cloud], d[cloud_compute], label=str(crit), alpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    savefig_all(fig, out_dir, f"edge_cloud_rtt_vs_cloud_compute_{which}", formats)

    # Total cloud vs edge-cloud rtt (context)
    fig, ax = plt.subplots()
    ax.set_title(f"Total Cloud vs Edge↔Cloud RTT ({which}) [rows with offload measurements]")
    ax.set_xlabel(f"{edge_cloud} (ms)")
    ax.set_ylabel(f"{total_cloud} (ms)")
    for crit in sorted(df["criterion"].dropna().unique()):
        d = df[df["criterion"] == crit].dropna(subset=[edge_cloud, total_cloud])
        if len(d) == 0:
            continue
        ax.scatter(d[edge_cloud], d[total_cloud], label=str(crit), alpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    savefig_all(fig, out_dir, f"total_cloud_vs_edge_cloud_rtt_{which}", formats)


def plot_energy_tradeoffs(df: pd.DataFrame, out_dir: str, formats: List[str]):
    """
    Energy / CO2 tradeoffs:
      - accuracy vs edge_energy_kwh
      - accuracy vs cloud_energy_kwh
      - accuracy vs total_energy_kwh
      - accuracy vs total_co2_kg
    Also Pareto for accuracy vs total_energy_kwh.
    """
    d = df.copy()

    # Total energy / co2 (if both exist)
    d["total_energy_kwh"] = d["edge_energy_kwh"] + d["cloud_energy_kwh"]
    d["total_co2_kg"] = d["edge_co2_kg"] + d["cloud_co2_kg"]

    def scatter_tradeoff(xcol: str, name: str, title: str):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(xcol)
        ax.set_ylabel("Accuracy")
        for crit in sorted(d["criterion"].dropna().unique()):
            sub = d[d["criterion"] == crit].dropna(subset=[xcol, "accuracy"])
            if len(sub) == 0:
                continue
            ax.scatter(sub[xcol], sub["accuracy"], label=str(crit), alpha=0.9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()
        savefig_all(fig, out_dir, name, formats)

    scatter_tradeoff("edge_energy_kwh", "accuracy_vs_edge_energy_kwh", "Accuracy vs Edge Energy (kWh) [per sweep]")
    scatter_tradeoff("cloud_energy_kwh", "accuracy_vs_cloud_energy_kwh", "Accuracy vs Cloud Energy (kWh) [per sweep]")
    scatter_tradeoff("total_energy_kwh", "accuracy_vs_total_energy_kwh", "Accuracy vs Total Energy (kWh) [per sweep]")
    scatter_tradeoff("total_co2_kg", "accuracy_vs_total_co2_kg", "Accuracy vs Total CO₂ (kg) [per sweep]")

    # Pareto frontier: minimize total_energy_kwh, maximize accuracy
    f = pareto_front(d.dropna(subset=["total_energy_kwh", "accuracy"]), x="total_energy_kwh", y="accuracy", maximize_y=True)
    if not f.empty:
        fig, ax = plt.subplots()
        ax.set_title("Pareto Frontier: Accuracy vs Total Energy (kWh)")
        ax.set_xlabel("total_energy_kwh")
        ax.set_ylabel("accuracy")
        ax.scatter(d["total_energy_kwh"], d["accuracy"], alpha=0.5)
        ax.plot(f["total_energy_kwh"], f["accuracy"], linewidth=2)
        annotate_topk(fig, ax, f, x="total_energy_kwh", y="accuracy", k=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        savefig_all(fig, out_dir, "pareto_accuracy_vs_total_energy_kwh", formats)


def plot_threshold_heatmaps_per_criterion(df: pd.DataFrame, out_dir: str, formats: List[str], metric: str):
    """
    Heatmap-like plot (not an image heatmap; thesis-safe with scatter grid):
      - x=thr0, y=thr1, size/value = metric (color by default matplotlib)
    Creates per-criterion maps.
    """
    assert metric in ("accuracy", "client_rtt_avg", "client_rtt_p95", "offload_pct", "total_edge_avg", "total_edge_p95")

    for crit in sorted(df["criterion"].dropna().unique()):
        d = df[df["criterion"] == crit].copy()
        d = d.dropna(subset=["thr0", "thr1", metric])
        if d.empty:
            continue

        fig, ax = plt.subplots()
        ax.set_title(f"{crit}: Threshold Map ({metric})")
        ax.set_xlabel("thr0")
        ax.set_ylabel("thr1")

        sc = ax.scatter(d["thr0"], d["thr1"], c=d[metric], alpha=0.9)
        fig.colorbar(sc, ax=ax, label=metric)

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        savefig_all(fig, out_dir, f"{crit}_threshold_map_{metric}", formats)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to results.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory for plots")
    ap.add_argument("--formats", default="png,pdf", help="Comma-separated: png,pdf,svg")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    formats = [x.strip() for x in args.formats.split(",") if x.strip()]
    ensure_dir(args.out_dir)

    df = pd.read_csv(args.csv)

    # Normalize numeric columns (robust)
    numeric_cols = [
        "thr0", "thr1",
        "n_total", "n_ok", "n_fail",
        "accuracy",
        "exit0_pct", "exit1_pct", "offload_pct", "final_pct",
        "client_rtt_avg", "client_rtt_p95",
        "total_edge_avg", "total_edge_p95",
        "edge_infer_avg", "edge_infer_p95",
        "edge_cloud_rtt_avg", "edge_cloud_rtt_p95",
        "cloud_compute_avg", "cloud_compute_p95",
        "total_cloud_avg", "total_cloud_p95",
        "edge_energy_kwh", "edge_co2_kg",
        "cloud_energy_kwh", "cloud_co2_kg",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = safe_float_series(df[c])

    # Basic sanity
    required = {"criterion", "thr0", "thr1", "accuracy", "client_rtt_avg", "client_rtt_p95"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {missing}")

    # Core plots
    plot_tradeoff_accuracy_latency(
        df=df,
        out_dir=args.out_dir,
        formats=formats,
        latency_col="client_rtt_avg",
        title_suffix="Client RTT (avg)",
        name_suffix="client_rtt_avg",
    )

    plot_tradeoff_accuracy_latency(
        df=df,
        out_dir=args.out_dir,
        formats=formats,
        latency_col="client_rtt_p95",
        title_suffix="Client RTT (p95)",
        name_suffix="client_rtt_p95",
    )

    plot_exit_offload_bars(df, args.out_dir, formats)

    plot_latency_decomposition(df, args.out_dir, formats, which="avg")
    plot_latency_decomposition(df, args.out_dir, formats, which="p95")

    # Energy (only meaningful if columns exist with non-NaN)
    if ("edge_energy_kwh" in df.columns) or ("cloud_energy_kwh" in df.columns):
        plot_energy_tradeoffs(df, args.out_dir, formats)

    # Threshold maps (useful for presenting sweeps)
    plot_threshold_heatmaps_per_criterion(df, args.out_dir, formats, metric="accuracy")
    plot_threshold_heatmaps_per_criterion(df, args.out_dir, formats, metric="offload_pct")
    plot_threshold_heatmaps_per_criterion(df, args.out_dir, formats, metric="client_rtt_avg")
    plot_threshold_heatmaps_per_criterion(df, args.out_dir, formats, metric="client_rtt_p95")

    print(f"[ok] wrote plots to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
