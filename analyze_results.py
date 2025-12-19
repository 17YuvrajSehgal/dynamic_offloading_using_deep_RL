#!/usr/bin/env python3
"""
analyze_results.py

Reads scenario result CSVs from results/scenarios/<scenario_variant>/ and produces:
- QoE over time plots (raw + optional rolling mean)
- Battery over time plots
- Summary tables for RL + baselines per variant and per scenario-set (s1, s2)

Expected CSV columns (same across files):
QoE,Latency,Energy,Battery,OffloadRatio,Action,TaskClass,Success
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

REQUIRED_COLS = ["QoE", "Latency", "Energy", "Battery", "OffloadRatio", "Action", "TaskClass", "Success"]


def find_variant_dirs(results_dir: Path, scenario_set: str) -> List[Path]:
    """
    scenario_set: 's1' or 's2' or 'all'
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    dirs = [p for p in results_dir.iterdir() if p.is_dir()]
    if scenario_set in ("s1", "s2"):
        dirs = [p for p in dirs if p.name.startswith(f"{scenario_set}_")]
    elif scenario_set == "all":
        dirs = [p for p in dirs if p.name.startswith(("s1_", "s2_"))]
    else:
        raise ValueError("scenario_set must be one of: s1, s2, all")

    # Sort nicely: s1_class1_90, s1_class2_90, ...
    dirs = sorted(dirs, key=lambda x: x.name)
    return dirs


def load_policy_csv(variant_dir: Path, filename: str) -> Optional[pd.DataFrame]:
    path = variant_dir / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # Ensure numeric where appropriate
    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=max(1, window // 4)).mean()


def summarize_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Return key summary stats for a policy in a scenario variant.
    """
    out = {}
    out["mean_qoe"] = float(df["QoE"].mean())
    out["median_qoe"] = float(df["QoE"].median())
    out["mean_latency"] = float(df["Latency"].mean())
    out["mean_energy"] = float(df["Energy"].mean())
    out["success_rate"] = float(df["Success"].mean())  # 0..1
    out["offload_ratio"] = float(df["OffloadRatio"].mean())
    out["final_battery"] = float(df["Battery"].iloc[-1]) if len(df) else np.nan
    out["min_battery"] = float(df["Battery"].min()) if len(df) else np.nan
    out["timesteps"] = int(len(df))
    return out


def make_variant_plots(
    variant_name: str,
    variant_dir: Path,
    policy_dfs: Dict[str, pd.DataFrame],
    out_dir: Path,
    smooth_window: int = 50,
    dpi: int = 180,
) -> Tuple[Path, Path]:
    """
    Creates two figures for a variant:
    - QoE over time (raw + rolling mean)
    - Battery over time (raw)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- QoE plot -----
    fig1 = plt.figure(figsize=(11, 5))
    ax1 = plt.gca()
    for label, df in policy_dfs.items():
        y_raw = df["QoE"]
        y_sm = rolling_mean(y_raw, smooth_window)
        x = np.arange(len(df))
        ax1.plot(x, y_sm, label=f"{label} (roll{smooth_window})")
    ax1.set_title(f"{variant_name} — QoE over time (rolling mean)")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("QoE")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, ncol=2)
    qoe_path = out_dir / f"{variant_name}_qoe.png"
    fig1.tight_layout()
    fig1.savefig(qoe_path, dpi=dpi)
    plt.close(fig1)

    # ----- Battery plot -----
    fig2 = plt.figure(figsize=(11, 5))
    ax2 = plt.gca()
    for label, df in policy_dfs.items():
        x = np.arange(len(df))
        ax2.plot(x, df["Battery"], label=label)
    ax2.set_title(f"{variant_name} — Battery over time")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Battery (J)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8, ncol=3)
    batt_path = out_dir / f"{variant_name}_battery.png"
    fig2.tight_layout()
    fig2.savefig(batt_path, dpi=dpi)
    plt.close(fig2)

    return qoe_path, batt_path


def make_scenario_dashboard(
    scenario_set: str,
    variant_dirs: List[Path],
    all_variant_policy_dfs: Dict[str, Dict[str, pd.DataFrame]],
    out_dir: Path,
    smooth_window: int = 50,
    dpi: int = 180,
) -> Optional[Path]:
    """
    Creates one “dashboard” figure per scenario_set:
    - Row 1: QoE rolling mean
    - Row 2: Battery
    - Columns: variants (e.g., class1_90, class2_90, class3_90, random)

    If too many variants, still works; it scales horizontally.
    """
    if not variant_dirs:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(variant_dirs)

    fig = plt.figure(figsize=(4.5 * n, 8.5))

    # QoE row
    for i, vdir in enumerate(variant_dirs):
        variant_name = vdir.name
        ax = fig.add_subplot(2, n, i + 1)
        policy_dfs = all_variant_policy_dfs[variant_name]
        for label, df in policy_dfs.items():
            x = np.arange(len(df))
            y = rolling_mean(df["QoE"], smooth_window)
            ax.plot(x, y, label=label)
        ax.set_title(f"{variant_name}\nQoE (roll{smooth_window})")
        ax.set_xlabel("t")
        ax.set_ylabel("QoE")
        ax.grid(True, alpha=0.25)
        if i == n - 1:
            ax.legend(fontsize=8, loc="best")

    # Battery row
    for i, vdir in enumerate(variant_dirs):
        variant_name = vdir.name
        ax = fig.add_subplot(2, n, n + i + 1)
        policy_dfs = all_variant_policy_dfs[variant_name]
        for label, df in policy_dfs.items():
            x = np.arange(len(df))
            ax.plot(x, df["Battery"], label=label)
        ax.set_title(f"{variant_name}\nBattery")
        ax.set_xlabel("t")
        ax.set_ylabel("J")
        ax.grid(True, alpha=0.25)
        if i == n - 1:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"{scenario_set.upper()} — RL vs Baselines (QoE & Battery)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = out_dir / f"{scenario_set}_dashboard_qoe_battery.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def save_tables(
    rows: List[Dict],
    out_dir: Path,
    filename: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    # Nice ordering
    preferred = [
        "scenario_set", "variant", "policy",
        "mean_qoe", "median_qoe",
        "success_rate", "offload_ratio",
        "mean_latency", "mean_energy",
        "final_battery", "min_battery",
        "timesteps",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols].sort_values(["scenario_set", "variant", "policy"])

    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def pretty_print_variant_table(df: pd.DataFrame, variant_name: str) -> None:
    show = df.copy()
    show["success_rate"] = (100.0 * show["success_rate"]).round(1).astype(str) + "%"
    show["offload_ratio"] = show["offload_ratio"].round(3)
    show["mean_qoe"] = show["mean_qoe"].map(lambda x: f"{x:.6f}")
    show["final_battery"] = show["final_battery"].map(lambda x: f"{x:.2f}")
    show["mean_latency"] = show["mean_latency"].map(lambda x: f"{x:.6f}")
    show["mean_energy"] = show["mean_energy"].map(lambda x: f"{x:.3f}")

    cols = ["policy", "mean_qoe", "success_rate", "offload_ratio", "mean_latency", "mean_energy", "final_battery"]
    print("\n" + "=" * 110)
    print(f"SUMMARY — {variant_name}")
    print("=" * 110)
    print(show[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results/scenarios",
                    help="Root directory containing scenario variant folders")
    ap.add_argument("--scenario-set", type=str, default="all", choices=["s1", "s2", "all"],
                    help="Which scenario set to analyze")
    ap.add_argument("--out-dir", type=str, default="results/analysis",
                    help="Where to write plots + tables")
    ap.add_argument("--smooth", type=int, default=50,
                    help="Rolling mean window for QoE plots")
    ap.add_argument("--dpi", type=int, default=180,
                    help="Output image DPI")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    variant_dirs = find_variant_dirs(results_dir, args.scenario_set)
    if not variant_dirs:
        print(f"No scenario variant folders found under: {results_dir} for scenario_set={args.scenario_set}")
        return

    all_rows: List[Dict] = []
    dashboards_by_set: Dict[str, List[Path]] = {"s1": [], "s2": []}

    # For scenario dashboards
    per_set_variants: Dict[str, List[Path]] = {"s1": [], "s2": []}
    per_set_variant_policy_dfs: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {"s1": {}, "s2": {}}

    for vdir in variant_dirs:
        variant_name = vdir.name
        scenario_set = "s1" if variant_name.startswith("s1_") else "s2"

        # Load all available policy CSVs for this variant
        policy_dfs: Dict[str, pd.DataFrame] = {}
        for label, fname in POLICY_FILES.items():
            df = load_policy_csv(vdir, fname)
            if df is not None:
                policy_dfs[label] = df

        if not policy_dfs:
            print(f"WARNING: No policy CSVs found in {vdir}")
            continue

        # Store for dashboard
        per_set_variants[scenario_set].append(vdir)
        per_set_variant_policy_dfs[scenario_set][variant_name] = policy_dfs

        # Plots per variant
        variant_out = out_dir / "plots" / variant_name
        qoe_path, batt_path = make_variant_plots(
            variant_name=variant_name,
            variant_dir=vdir,
            policy_dfs=policy_dfs,
            out_dir=variant_out,
            smooth_window=args.smooth,
            dpi=args.dpi,
        )

        # Table rows per policy
        variant_rows = []
        for label, df in policy_dfs.items():
            s = summarize_metrics(df)
            row = {
                "scenario_set": scenario_set,
                "variant": variant_name,
                "policy": label,
                **s,
                "qoe_plot": str(qoe_path),
                "battery_plot": str(batt_path),
            }
            all_rows.append(row)
            variant_rows.append(row)

        # Pretty print variant summary (console)
        pretty_print_variant_table(pd.DataFrame(variant_rows), variant_name)

    # Save combined table
    tables_out = out_dir / "tables"
    all_csv = save_tables(all_rows, tables_out, "all_summaries.csv")
    print(f"\n✅ Saved combined summary table: {all_csv}")

    # Save per-scenario-set tables + dashboards
    for sset in ("s1", "s2"):
        vdirs = per_set_variants[sset]
        if not vdirs:
            continue

        # Scenario dashboard figure
        dash_out = out_dir / "dashboards"
        dash_path = make_scenario_dashboard(
            scenario_set=sset,
            variant_dirs=vdirs,
            all_variant_policy_dfs=per_set_variant_policy_dfs[sset],
            out_dir=dash_out,
            smooth_window=args.smooth,
            dpi=args.dpi,
        )
        if dash_path:
            print(f"✅ Saved {sset.upper()} dashboard: {dash_path}")

        # Scenario-only CSV
        s_rows = [r for r in all_rows if r["scenario_set"] == sset]
        s_csv = save_tables(s_rows, tables_out, f"{sset}_summaries.csv")
        print(f"✅ Saved {sset.upper()} summary table: {s_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
