#!/usr/bin/env python3
"""
plot_scenario_results.py

Generates publication-quality plots for Scenario results (QoE + Battery only),
similar to the "complete figure" layout, but WITHOUT the Decisions heatmap.

Usage:
    python plot_scenario_results.py --scenario s1 --results-dir results/scenarios
    python plot_scenario_results.py --scenario s2 --results-dir results/scenarios
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


# Color scheme (keep yours)
COLORS = {
    "Join-Networks AC": "#1f77b4",       # Blue
    "DDDQN": "#d62728",                  # Red
    "cloud": "#8c564b",                  # Brown
    "mec": "#2ca02c",                    # Green
    "random": "#9467bd",                 # Purple
    "local": "#ff7f0e",                  # Orange
    "SeparatedNetworks AC": "#e377c2",   # Pink (RL agent)
    "Greedy By Size": "#000000",         # Black
}

# Shaded "event" windows (red bands) per scenario
# Values are x-axis ranges (Task ID) to shade.
SHADE_WINDOWS = {
    "s1": [(500, 750), (1250, 1500)],
    "s2": [(500, 1000)],
}

SHADE_STYLE = dict(color="red", alpha=0.12, zorder=0)


def smooth(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def load_scenario_results(results_dir: str, scenario: str) -> Dict[str, pd.DataFrame]:
    """
    Load results for all methods (baselines + RL) for a given scenario folder.

    Returns:
        Dictionary mapping method key to DataFrame
    """
    results: Dict[str, pd.DataFrame] = {}
    scenario_dir = Path(results_dir) / scenario

    if not scenario_dir.exists():
        print(f"Warning: Scenario directory not found: {scenario_dir}")
        return results

    # RL results
    rl_file = scenario_dir / "rl_agent_metrics.csv"
    if rl_file.exists():
        results["RL-Agent"] = pd.read_csv(rl_file)
        print(f"Loaded RL results: {len(results['RL-Agent'])} timesteps")

    # Baselines
    for baseline in ["local", "mec", "cloud", "random", "greedy_by_size"]:
        baseline_file = scenario_dir / f"{baseline}_metrics.csv"
        if baseline_file.exists():
            results[baseline] = pd.read_csv(baseline_file)
            print(f"Loaded {baseline} results: {len(results[baseline])} timesteps")

    return results


def _pretty_label(method_key: str) -> str:
    """Convert internal keys to plot labels."""
    if method_key == "RL-Agent":
        return "SeparatedNetworks AC"
    if method_key == "greedy_by_size":
        return "Greedy By Size"
    return method_key.replace("_", " ").title()


def plot_scenario_row(
    fig: plt.Figure,
    gs: GridSpec,
    row_idx: int,
    results: Dict[str, pd.DataFrame],
    class_label: str,
    scenario_prefix,
    show_legend: bool = False,
    x_max: int = 2000,
):
    """
    Plot one row of results (QoE, Battery) for a scenario distribution.

    Args:
        fig: Matplotlib figure
        gs: Shared GridSpec (4 rows x 2 cols)
        row_idx: 0..3
        results: mapping of method->DataFrame
        class_label: "Class 1" / "Class 2" / "Class 3" / "Random"
        show_legend: legend only once (top-left plot)
        x_max: x-axis max (Task ID)
    """
    ax_qoe = fig.add_subplot(gs[row_idx, 0])
    ax_battery = fig.add_subplot(gs[row_idx, 1])

    # --- Shade scenario-specific event windows ---
    for (start, end) in SHADE_WINDOWS.get(scenario_prefix, []):
        ax_qoe.axvspan(start, end, **SHADE_STYLE)
        ax_battery.axvspan(start, end, **SHADE_STYLE)

    # --- QoE ---
    for method, df in results.items():
        if "QoE" not in df.columns:
            continue

        qoe_vals = pd.to_numeric(df["QoE"], errors="coerce").to_numpy()
        qoe_smooth = smooth(qoe_vals, window=10)
        x = np.arange(len(qoe_smooth))

        label = _pretty_label(method)
        color = COLORS.get(label, COLORS.get(method, "#000000"))

        ax_qoe.plot(x, qoe_smooth, label=label, color=color, linewidth=1.5, alpha=0.85)

    ax_qoe.set_ylim(-0.10, 0.01)
    ax_qoe.axhline(y=0, color="black", linestyle="--", linewidth=0.6, alpha=0.35)
    ax_qoe.set_ylabel(class_label, fontsize=10, fontweight="bold")
    ax_qoe.grid(True, alpha=0.28)
    ax_qoe.set_xlim(0, x_max)

    if row_idx == 0:
        ax_qoe.set_title("QoE", fontsize=11, fontweight="bold")
    if row_idx == 3:
        ax_qoe.set_xlabel("Task ID", fontsize=9)
    else:
        ax_qoe.set_xticklabels([])

    # --- Battery ---
    for method, df in results.items():
        if "Battery" not in df.columns:
            continue

        bat_vals = pd.to_numeric(df["Battery"], errors="coerce").to_numpy()
        x = np.arange(len(bat_vals))

        label = _pretty_label(method)
        color = COLORS.get(label, COLORS.get(method, "#000000"))

        ax_battery.plot(x, bat_vals, label=label, color=color, linewidth=1.5, alpha=0.85)

        # marker every 100
        marker_idx = np.arange(0, len(x), 100)
        if len(marker_idx) > 0:
            ax_battery.plot(
                x[marker_idx],
                bat_vals[marker_idx],
                marker="o",
                markersize=3,
                color=color,
                linestyle="None",
                alpha=0.9,
            )

    ax_battery.set_ylim(0, 4500)
    ax_battery.grid(True, alpha=0.28)
    ax_battery.set_xlim(0, x_max)

    if row_idx == 0:
        ax_battery.set_title("Battery", fontsize=11, fontweight="bold")
    if row_idx == 3:
        ax_battery.set_xlabel("Task ID", fontsize=9)
    else:
        ax_battery.set_xticklabels([])

    # Legend (only once)
    if show_legend and row_idx == 0:
        ax_qoe.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)


def create_complete_figure(
    scenario_prefix: str,
    results_dir: str,
    output_file: Optional[str] = None,
):
    """
    Create the complete figure (4 rows x 2 cols) for a scenario prefix (s1 or s2).
    """
    scenarios = [
        (f"{scenario_prefix}_class1_90", "Class 1"),
        (f"{scenario_prefix}_class2_90", "Class 2"),
        (f"{scenario_prefix}_class3_90", "Class 3"),
        (f"{scenario_prefix}_random", "Random"),
    ]

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    fig.suptitle(
        f"Scenario {scenario_prefix[-1]} Results",
        fontsize=14,
        fontweight="bold",
    )

    # One shared GridSpec -> fixes alignment
    gs = GridSpec(
        4,
        2,
        figure=fig,
        width_ratios=[1.15, 1.0],
        hspace=0.35,
        wspace=0.25,
    )

    for row_idx, (scenario_key, class_label) in enumerate(scenarios):
        print(f"\nPlotting {class_label} ...")
        results = load_scenario_results(results_dir, scenario_key)

        if not results:
            print(f"  Warning: No results found for {scenario_key}")
            continue

        plot_scenario_row(
            fig=fig,
            gs=gs,
            row_idx=row_idx,
            results=results,
            class_label=class_label,
            scenario_prefix=scenario_prefix,
            show_legend=(row_idx == 0),
            x_max=2000,
        )

    if output_file is None:
        output_file = os.path.join(results_dir, f"{scenario_prefix}_complete_figure_no_decisions.png")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✅ Figure saved to: {output_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate plots for scenario results (QoE + Battery only)")
    parser.add_argument(
        "--scenario",
        type=str,
        default="s1",
        choices=["s1", "s2"],
        help="Scenario to plot (s1 or s2)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/scenarios",
        help="Directory containing scenario results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: results-dir/<scenario>_complete_figure_no_decisions.png)",
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Generating plots for Scenario {args.scenario.upper()} (NO decisions heatmap)")
    print(f"{'='*80}")
    print(f"Results directory: {args.results_dir}")

    create_complete_figure(args.scenario, args.results_dir, args.output)

    print(f"\n{'='*80}")
    print("Plotting complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
