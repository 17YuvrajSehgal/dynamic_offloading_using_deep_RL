#!/usr/bin/env python3
"""
plot_scenario_dashboard_bars.py

Creates ONE image per scenario group (s1 or s2) with 4 rows (cases) and 4 columns (metrics).
Each row is the bar-comparison plot (QoE, Latency, Final Battery, Success Rate) for that case.

Folder layout expected:
  results/scenarios/
    s1_class1_90/
      cloud_metrics.csv, local_metrics.csv, mec_metrics.csv, random_metrics.csv, rl_agent_metrics.csv
    s1_class2_90/
    s1_class3_90/
    s1_random/
    s2_class1_90/
    ...

Usage:
  python plot_scenario_dashboard_bars.py --scenario s1 --results-dir results/scenarios
  python plot_scenario_dashboard_bars.py --scenario s2 --results-dir results/scenarios
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Order + naming
POLICY_ORDER = ["cloud", "local", "mec", "random", "rl_agent", "greedy_by_size", "greedy"]
DISPLAY = {
    "cloud": "Always-Cloud",
    "local": "Always-Local",
    "mec": "Always-MEC",
    "random": "Random-Policy",
    "rl_agent": "RL-Agent",
    "greedy_by_size": "Greedy-By-Size",
    "greedy": "Greedy-By-Size",
}

CASES = [
    ("class1_90", "Class 1 (90%)"),
    ("class2_90", "Class 2 (90%)"),
    ("class3_90", "Class 3 (90%)"),
    ("random",    "Random Mix"),
]


def policy_key_from_filename(p: Path) -> str:
    s = p.stem.lower()
    if s.endswith("_metrics"):
        s = s[:-len("_metrics")]
    return s


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    out["mean_qoe"] = float(safe_numeric(df["QoE"]).mean()) if "QoE" in df.columns else np.nan
    out["mean_latency"] = float(safe_numeric(df["Latency"]).mean()) if "Latency" in df.columns else np.nan
    out["final_battery"] = float(safe_numeric(df["Battery"]).iloc[-1]) if "Battery" in df.columns and len(safe_numeric(df["Battery"])) else np.nan
    out["success_rate"] = float(safe_numeric(df["Success"]).mean()) if "Success" in df.columns else np.nan
    return out


def load_case(case_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Returns: {policy_key: summary_dict}
    """
    summaries: Dict[str, Dict[str, float]] = {}
    for csv_path in sorted(case_dir.glob("*_metrics.csv")):
        key = policy_key_from_filename(csv_path)
        try:
            df = pd.read_csv(csv_path)
            summaries[key] = summarize(df)
        except Exception as e:
            print(f"[WARN] Could not load {csv_path}: {e}")
    return summaries


def ordered_policies(keys: List[str]) -> List[str]:
    out = [k for k in POLICY_ORDER if k in keys]
    out += [k for k in keys if k not in out]
    return out


def bar(ax, labels: List[str], vals: List[float], title: str, ylim=None):
    x = np.arange(len(labels))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["s1", "s2"], help="Scenario group (s1 or s2)")
    ap.add_argument("--results-dir", default="results/scenarios", help="Folder containing scenario subfolders")
    ap.add_argument("--out", default=None, help="Output image path")
    args = ap.parse_args()

    base = Path(args.results_dir)
    out_path = Path(args.out) if args.out else base / f"{args.scenario}_dashboard_bars.png"

    # 4 rows (cases) × 4 cols (metrics)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 11))
    fig.suptitle(
        f"Strategy Comparison Dashboard — {args.scenario.upper()} (all 4 cases)",
        fontsize=16
    )

    for r, (suffix, row_name) in enumerate(CASES):
        case_name = f"{args.scenario}_{suffix}"
        case_dir = base / case_name

        if not case_dir.exists():
            for c in range(4):
                axes[r, c].text(0.5, 0.5, f"Missing\n{case_name}", ha="center", va="center")
                axes[r, c].set_axis_off()
            continue

        summaries = load_case(case_dir)
        if not summaries:
            for c in range(4):
                axes[r, c].text(0.5, 0.5, f"No *_metrics.csv\nin {case_name}", ha="center", va="center")
                axes[r, c].set_axis_off()
            continue

        pols = ordered_policies(list(summaries.keys()))
        labels = [DISPLAY.get(p, p) for p in pols]

        qoe = [summaries[p]["mean_qoe"] for p in pols]
        lat = [summaries[p]["mean_latency"] for p in pols]
        bat = [summaries[p]["final_battery"] for p in pols]
        suc = [summaries[p]["success_rate"] for p in pols]

        # Column titles only on first row (keeps it clean)
        bar(axes[r, 0], labels, qoe, "Average QoE" if r == 0 else "", ylim=None)
        bar(axes[r, 1], labels, lat, "Average Latency (s)" if r == 0 else "", ylim=None)
        bar(axes[r, 2], labels, bat, "Final Battery (J)" if r == 0 else "", ylim=None)
        bar(axes[r, 3], labels, suc, "Success Rate" if r == 0 else "", ylim=(0, 1.0))

        # Add row label on far-left
        axes[r, 0].set_ylabel(row_name, fontsize=12)

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
