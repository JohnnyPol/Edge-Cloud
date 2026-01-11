#!/usr/bin/env python3
"""
plot_notebook_style_sweeps.py

Implements "notebook-style" plots (accuracy/emissions/energy/time/exits vs threshold)
adapted to sweep CSV format:

Columns expected (from your results.csv):
- criterion, thr0, thr1
- accuracy
- client_rtt_avg, client_rtt_p95
- total_edge_avg, total_edge_p95
- edge_infer_avg, edge_infer_p95
- exit0_pct, exit1_pct, offload_pct, final_pct
- edge_energy_kwh, edge_co2_kg, cloud_energy_kwh, cloud_co2_kg  (optional)

Notes:
- Replaces "phase" with "criterion"
- Replaces "threshold" with thr0/thr1
- Produces plots similar to your notebook functions, and saves them to out_dir.

Author: you + ChatGPT
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_num(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def savefig(fig, out_dir: str, name: str, formats=("png"), dpi: int = 200) -> None:
    for fmt in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def lineplot_by_threshold(
    df: pd.DataFrame,
    ycol: str,
    title: str,
    ylabel: str,
    out_dir: str,
    name: str,
    base_line: Optional[float] = None,
    base_label: Optional[str] = None,
    use_thr: str = "thr0",  # "thr0" or "thr1"
    formats=("png", "pdf"),
):
    """
    Notebook-style line plot:
      x = thr0 or thr1
      y = ycol
      hue = criterion
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel(use_thr)
    ax.set_ylabel(ylabel)

    for crit in sorted(df["criterion"].dropna().unique()):
        d = df[df["criterion"] == crit].sort_values(by=use_thr)
        ax.plot(d[use_thr], d[ycol], marker="o", label=str(crit))

    if base_line is not None:
        ax.axhline(base_line, linestyle="--", label=base_label if base_label else "base")

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="criterion")
    savefig(fig, out_dir, name, formats=formats)


def plot_accuracy_points_thr0_thr1(
    df: pd.DataFrame,
    out_dir: str,
    formats=("png", "pdf"),
    annotate: bool = False,
    same_color_scale: bool = True,
):
    """
    For each criterion:
      x = thr0
      y = thr1
      point color = accuracy
    """

    # Determine a global color scale (so maxprob/margin/entropy are comparable)
    vmin = vmax = None
    if same_color_scale:
        acc_all = df["accuracy"].dropna()
        if len(acc_all) > 0:
            vmin = float(acc_all.min())
            vmax = float(acc_all.max())

    for crit in sorted(df["criterion"].dropna().unique()):
        d = df[df["criterion"] == crit].copy()
        d = d.dropna(subset=["thr0", "thr1", "accuracy"])
        if d.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 7))

        sc = ax.scatter(
            d["thr0"].astype(float),
            d["thr1"].astype(float),
            c=d["accuracy"].astype(float),
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(f"{crit}: Accuracy over (thr0, thr1)")
        ax.set_xlabel("thr0")
        ax.set_ylabel("thr1")
        ax.grid(True, linestyle="--", alpha=0.35)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("accuracy")

        # Optional: label each point with its accuracy (can clutter)
        if annotate and len(d) <= 40:
            for _, r in d.iterrows():
                ax.text(
                    float(r["thr0"]),
                    float(r["thr1"]),
                    f"{float(r['accuracy']):.3f}",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )

        savefig(fig, out_dir, f"{crit}_accuracy_thr0_thr1", formats=formats)


def plot_exits_per_threshold(
    df: pd.DataFrame,
    out_dir: str,
    use_thr: str = "thr0",
    formats=("png", "pdf"),
):
    """
    Robust exits plot for 2-threshold sweeps.

    For each criterion, plot:
      x = thr0 (or thr1)
      y = {exit0_pct, exit1_pct, offload_pct, final_pct} as separate curves.

    This avoids duplicate-label issues because we don't pivot by exit_point index.
    """
    exit_cols = ["exit0_pct", "exit1_pct", "offload_pct", "final_pct"]
    have = [c for c in exit_cols if c in df.columns]
    if not have:
        return

    for crit in sorted(df["criterion"].dropna().unique()):
        d = df[df["criterion"] == crit].copy()

        # Sort by x-axis threshold; keep all (thr0,thr1) points
        d = d.sort_values(by=use_thr)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{crit}: Exit/Offload % vs {use_thr}")
        ax.set_xlabel(use_thr)
        ax.set_ylabel("percentage (%)")

        for col in have:
            ax.plot(d[use_thr], d[col], marker="o", label=col.replace("_pct", ""))

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(title="exit type")
        savefig(fig, out_dir, f"{crit}_exits_vs_{use_thr}", formats=formats)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to results.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--formats", default="png", help="Comma-separated: png,pdf,svg")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    formats = tuple([x.strip() for x in args.formats.split(",") if x.strip()])
    ensure_dir(args.out_dir)
    
    df = pd.read_csv(args.csv)

    dups = df.duplicated(subset=["criterion", "thr0", "thr1"]).sum()
    print(f"[info] duplicate (criterion,thr0,thr1) rows: {dups}")
    # Normalize numeric columns we might use
    to_num(
        df,
        [
            "thr0", "thr1", "accuracy",
            "client_rtt_avg", "client_rtt_p95",
            "total_edge_avg", "total_edge_p95",
            "edge_infer_avg", "edge_infer_p95",
            "exit0_pct", "exit1_pct", "offload_pct", "final_pct",
            "edge_energy_kwh", "edge_co2_kg",
            "cloud_energy_kwh", "cloud_co2_kg",
        ],
    )

    # Derived totals (if present)
    if "edge_co2_kg" in df.columns and "cloud_co2_kg" in df.columns:
        df["total_co2_kg"] = df["edge_co2_kg"] + df["cloud_co2_kg"]
    if "edge_energy_kwh" in df.columns and "cloud_energy_kwh" in df.columns:
        df["total_energy_kwh"] = df["edge_energy_kwh"] + df["cloud_energy_kwh"]

    # ------------------------------------------------------------------
    # 1) Accuracy per threshold (thr0 and thr1)
    # ------------------------------------------------------------------
    if "accuracy" in df.columns:
        plot_accuracy_points_thr0_thr1(df, args.out_dir, formats=formats, annotate=False)


    # ------------------------------------------------------------------
    # 2) Emissions (CO2) per threshold
    # ------------------------------------------------------------------
    if "total_co2_kg" in df.columns:
        lineplot_by_threshold(
            df, "total_co2_kg",
            title="Total CO₂ (edge+cloud) vs thr0",
            ylabel="CO₂ (kg)",
            out_dir=args.out_dir,
            name="co2_total_vs_thr0",
            use_thr="thr0",
            formats=formats,
        )
        lineplot_by_threshold(
            df, "total_co2_kg",
            title="Total CO₂ (edge+cloud) vs thr1",
            ylabel="CO₂ (kg)",
            out_dir=args.out_dir,
            name="co2_total_vs_thr1",
            use_thr="thr1",
            formats=formats,
        )

    # ------------------------------------------------------------------
    # 3) Energy per threshold
    # ------------------------------------------------------------------
    if "total_energy_kwh" in df.columns:
        lineplot_by_threshold(
            df, "total_energy_kwh",
            title="Total Energy (edge+cloud) vs thr0",
            ylabel="Energy (kWh)",
            out_dir=args.out_dir,
            name="energy_total_vs_thr0",
            use_thr="thr0",
            formats=formats,
        )
        lineplot_by_threshold(
            df, "total_energy_kwh",
            title="Total Energy (edge+cloud) vs thr1",
            ylabel="Energy (kWh)",
            out_dir=args.out_dir,
            name="energy_total_vs_thr1",
            use_thr="thr1",
            formats=formats,
        )

    # ------------------------------------------------------------------
    # 4) Inference time plots (your notebook plotted "inference_time" and avg time)
    #    Here we map to your sweep metrics:
    #      - total_edge_avg / total_edge_p95 (end-to-end on edge server)
    #      - client_rtt_avg / client_rtt_p95 (what client experiences)
    # ------------------------------------------------------------------
    for ycol, title, ylabel, fname in [
        ("total_edge_avg", "Total Edge Time (avg) vs thr0", "time (ms)", "total_edge_avg_vs_thr0"),
        ("total_edge_p95", "Total Edge Time (p95) vs thr0", "time (ms)", "total_edge_p95_vs_thr0"),
        ("client_rtt_avg", "Client RTT (avg) vs thr0", "time (ms)", "client_rtt_avg_vs_thr0"),
        ("client_rtt_p95", "Client RTT (p95) vs thr0", "time (ms)", "client_rtt_p95_vs_thr0"),
    ]:
        if ycol in df.columns:
            lineplot_by_threshold(
                df, ycol,
                title=title,
                ylabel=ylabel,
                out_dir=args.out_dir,
                name=fname,
                use_thr="thr0",
                formats=formats,
            )

    for ycol, title, ylabel, fname in [
        ("total_edge_avg", "Total Edge Time (avg) vs thr1", "time (ms)", "total_edge_avg_vs_thr1"),
        ("total_edge_p95", "Total Edge Time (p95) vs thr1", "time (ms)", "total_edge_p95_vs_thr1"),
        ("client_rtt_avg", "Client RTT (avg) vs thr1", "time (ms)", "client_rtt_avg_vs_thr1"),
        ("client_rtt_p95", "Client RTT (p95) vs thr1", "time (ms)", "client_rtt_p95_vs_thr1"),
    ]:
        if ycol in df.columns:
            lineplot_by_threshold(
                df, ycol,
                title=title,
                ylabel=ylabel,
                out_dir=args.out_dir,
                name=fname,
                use_thr="thr1",
                formats=formats,
            )

    # ------------------------------------------------------------------
    # 5) Exits plot (like your plot_exits)
    # ------------------------------------------------------------------
    plot_exits_per_threshold(df, args.out_dir, use_thr="thr0", formats=formats)
    plot_exits_per_threshold(df, args.out_dir, use_thr="thr1", formats=formats)

    print(f"[ok] wrote notebook-style plots to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
