#!/usr/bin/env python3
"""
plot_qoe_bars.py

Bar charts to compare QoE across all variants within a scenario set (s1 or s2),
comparing different algorithms (RL + baselines).

Reads:
results/scenarios/<variant>/{rl_agent_metrics,local_metrics,mec_metrics,cloud_metrics,random_metrics,greedy_by_size_metrics}.csv

Each CSV must contain a QoE column.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


POLICY_FILES = {
    "RL (Actor-Critic)": "rl_agent_metrics.csv",
    "Local": "local_metrics.csv",
    "MEC": "mec_metrics.csv",
    "Cloud": "cloud_metrics.csv",
    "Random": "random_metrics.csv",
    "Greedy-by-Size": "greedy_by_size_metrics.csv",
}


def find_variant_dirs(results_dir: Path, scenario_set: str) -> List[Path]:
    dirs = [p for p in results_dir.iterdir() if p.is_dir()]
    dirs = [p for p in dirs if p.name.startswith(f"{scenario_set}_")]
    return sorted(dirs, key=lambda x: x.name)


def load_mean_qoe(variant_dir: Path, filename: str) -> Optional[float]:
    path = variant_dir / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "QoE" not in df.columns:
        raise ValueError(f"Missing 'QoE' column in {path}")
    return float(pd.to_numeric(df["QoE"], errors="coerce").mean())


def plot_grouped_bars(
    qoe_table: pd.DataFrame,
    scenario_set: str,
    out_path: Path,
    dpi: int = 180,
):
    """
    qoe_table:
      rows = variants
      cols = policies
      values = mean QoE
    """
    variants = qoe_table.index.tolist()
    policies = qoe_table.columns.tolist()

    n_groups = len(variants)
    n_policies = len(policies)

    x = np.arange(n_groups)
    width = 0.8 / max(1, n_policies)

    fig = plt.figure(figsize=(max(10, 1.7 * n_groups), 6))
    ax = plt.gca()

    for i, pol in enumerate(policies):
        ax.bar(x + (i - (n_policies - 1) / 2) * width, qoe_table[pol].values, width, label=pol)

    ax.set_title(f"{scenario_set.upper()} — Mean QoE Comparison (All Variants)")
    ax.set_xlabel("Scenario Variant")
    ax.set_ylabel("Mean QoE")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_overall_means(
    qoe_table: pd.DataFrame,
    scenario_set: str,
    out_path: Path,
    dpi: int = 180,
):
    """
    One bar per policy = average(mean QoE across variants).
    """
    overall = qoe_table.mean(axis=0, skipna=True).sort_values(ascending=False)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.bar(overall.index.tolist(), overall.values)
    ax.set_title(f"{scenario_set.upper()} — Overall Mean QoE (Averaged Across Variants)")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean QoE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xticklabels(overall.index.tolist(), rotation=20, ha="right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results/scenarios",
                    help="Root results dir containing scenario variant folders")
    ap.add_argument("--scenario-set", type=str, required=True, choices=["s1", "s2"],
                    help="Which scenario set to plot (s1 or s2)")
    ap.add_argument("--out-dir", type=str, default="results/analysis/bars",
                    help="Where to save bar charts and tables")
    ap.add_argument("--dpi", type=int, default=180, help="Output image DPI")
    ap.add_argument("--save-csv", action="store_true",
                    help="Also save the mean QoE table to CSV")
    ap.add_argument("--overall", action="store_true",
                    help="Also create overall-mean QoE bar chart")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    variant_dirs = find_variant_dirs(results_dir, args.scenario_set)
    if not variant_dirs:
        print(f"No variant folders found for {args.scenario_set} under {results_dir}")
        return

    # Build table: rows=variants, cols=policies
    rows = []
    variant_names = []

    for vdir in variant_dirs:
        variant_names.append(vdir.name)
        row = {}
        for pol, fname in POLICY_FILES.items():
            row[pol] = load_mean_qoe(vdir, fname)
        rows.append(row)

    qoe_table = pd.DataFrame(rows, index=variant_names)

    # Save table (optional)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_csv:
        csv_path = out_dir / f"{args.scenario_set}_mean_qoe_table.csv"
        qoe_table.to_csv(csv_path, index=True)
        print(f"✅ Saved table: {csv_path}")

    # Plot grouped bars
    grouped_path = out_dir / f"{args.scenario_set}_mean_qoe_grouped_bars.png"
    plot_grouped_bars(qoe_table, args.scenario_set, grouped_path, dpi=args.dpi)
    print(f"✅ Saved grouped bar chart: {grouped_path}")

    # Plot overall means (optional)
    if args.overall:
        overall_path = out_dir / f"{args.scenario_set}_overall_mean_qoe_bars.png"
        plot_overall_means(qoe_table, args.scenario_set, overall_path, dpi=args.dpi)
        print(f"✅ Saved overall mean bar chart: {overall_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
