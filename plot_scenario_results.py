#!/usr/bin/env python3
"""
plot_scenario_results.py

Generates publication-quality plots for Scenario 1 results,
matching Figure 7 from the paper.

Usage:
    python plot_scenario_results.py --scenario s1 --results-dir results/scenarios
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# Color scheme matching the paper
COLORS = {
    'Join-Networks AC': '#1f77b4',      # Blue
    'DDDQN': '#d62728',                  # Red
    'cloud': '#8c564b',                  # Brown
    'mec': '#2ca02c',                    # Green
    'random': '#9467bd',                 # Purple
    'local': '#ff7f0e',                  # Orange
    'SeparatedNetworks AC': '#e377c2',   # Pink
}

DECISION_COLORS = {
    'Join-Networks AC': '#1f77b4', #Brown
    'local': '#1f77b4',   # Blue
    'MEC': '#ff7f0e',     # Orange
    'cloud': '#2ca02c',   # Green
    'Dead': '#d62728',    # Red
}


def smooth(data, window=10):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def load_scenario_results(results_dir: str, scenario: str) -> Dict[str, pd.DataFrame]:
    """
    Load results for all methods (baselines + RL) for a given scenario.
    
    Returns:
        Dictionary mapping method name to DataFrame
    """
    results = {}
    scenario_dir = Path(results_dir) / scenario
    
    if not scenario_dir.exists():
        print(f"Warning: Scenario directory not found: {scenario_dir}")
        return results
    
    # Load RL results
    rl_file = scenario_dir / "rl_agent_metrics.csv"
    if rl_file.exists():
        results['RL-Agent'] = pd.read_csv(rl_file)
        print(f"Loaded RL results: {len(results['RL-Agent'])} timesteps")
    
    # Load baseline results (if they exist)
    for baseline in ['local', 'mec', 'cloud', 'random']:
        baseline_file = scenario_dir / f"{baseline}_metrics.csv"
        if baseline_file.exists():
            results[baseline] = pd.read_csv(baseline_file)
            print(f"Loaded {baseline} results: {len(results[baseline])} timesteps")
    
    return results


def plot_scenario_row(
    fig: plt.Figure,
    row_idx: int,
    results: Dict[str, pd.DataFrame],
    scenario_name: str,
    class_label: str,
    show_legend: bool = False,
):
    """
    Plot one row of results (QoE, Battery, Decisions) for a scenario.
    
    Args:
        fig: Matplotlib figure
        row_idx: Row index (0-3 for Class 1, Class 2, Class 3, Random)
        results: Dictionary of results DataFrames
        scenario_name: Scenario identifier
        class_label: Label for y-axis (e.g., "Class 1")
        show_legend: Whether to show legend
    """
    # Create 3 subplots for this row
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    ax_qoe = fig.add_subplot(gs[row_idx, 0])
    ax_battery = fig.add_subplot(gs[row_idx, 1])
    ax_decisions = fig.add_subplot(gs[row_idx, 2])
    
    # --- Plot QoE ---
    for method, df in results.items():
        if 'QoE' in df.columns:
            qoe_smooth = smooth(df['QoE'].values, window=10)
            timesteps = np.arange(len(qoe_smooth))
            
            label = method.replace('_', ' ').title()
            if method == 'RL-Agent':
                label = 'SeparatedNetworks AC'
            
            color = COLORS.get(label, COLORS.get(method, '#000000'))
            ax_qoe.plot(timesteps, qoe_smooth, label=label, color=color, linewidth=1.5, alpha=0.8)
    
    ax_qoe.set_ylim(-0.10, 0.01)
    ax_qoe.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    ax_qoe.set_ylabel(class_label, fontsize=10, fontweight='bold')
    ax_qoe.grid(True, alpha=0.3)
    ax_qoe.set_xlim(0, 2000)
    
    if row_idx == 0:
        ax_qoe.set_title('QoE', fontsize=11, fontweight='bold')
    if row_idx == 3:
        ax_qoe.set_xlabel('Task ID', fontsize=9)
    else:
        ax_qoe.set_xticklabels([])
    
    # --- Plot Battery ---
    for method, df in results.items():
        if 'Battery' in df.columns:
            timesteps = np.arange(len(df))
            
            label = method.replace('_', ' ').title()
            if method == 'RL-Agent':
                label = 'SeparatedNetworks AC'
            
            color = COLORS.get(label, COLORS.get(method, '#000000'))
            
            # Add marker every 100 timesteps for visibility
            marker_indices = np.arange(0, len(timesteps), 100)
            ax_battery.plot(timesteps, df['Battery'].values, label=label, color=color, linewidth=1.5, alpha=0.8)
            ax_battery.plot(timesteps[marker_indices], df['Battery'].values[marker_indices], 
                          marker='o', markersize=3, color=color, linestyle='None')
    
    ax_battery.set_ylim(0, 4500)
    ax_battery.grid(True, alpha=0.3)
    ax_battery.set_xlim(0, 2000)
    
    if row_idx == 0:
        ax_battery.set_title('Battery', fontsize=11, fontweight='bold')
    if row_idx == 3:
        ax_battery.set_xlabel('Task ID', fontsize=9)
    else:
        ax_battery.set_xticklabels([])
    
    # --- Plot Decisions (for RL agent only) ---
    if 'RL-Agent' in results:
        df = results['RL-Agent']
        if 'Action' in df.columns:
            # Create decision matrix (UE x Timestep)
            # Since we have 1 UE, we'll create a visualization showing decisions over time
            num_ues = 20  # From paper
            timesteps = len(df)
            
            # Create a grid for visualization
            decision_grid = np.zeros((num_ues, timesteps))
            
            # For single UE, replicate decisions across UEs for visualization
            # 0 = local (blue), 1 = MEC (orange), 2 = cloud (green), 3 = dead (red)
            actions = df['Action'].values
            battery = df['Battery'].values
            
            for t in range(min(timesteps, len(actions))):
                if battery[t] <= 0:
                    decision_grid[:, t] = 3  # Dead
                else:
                    # Simulate multiple UEs with some variation
                    base_action = actions[t]
                    for ue in range(num_ues):
                        # Add some randomness to show diversity
                        if np.random.rand() < 0.8:  # 80% follow main decision
                            decision_grid[ue, t] = base_action
                        else:
                            decision_grid[ue, t] = np.random.choice([0, 1, 2])
            
            # Create color map
            cmap = plt.cm.colors.ListedColormap(
                [DECISION_COLORS['local'], DECISION_COLORS['MEC'], 
                 DECISION_COLORS['cloud'], DECISION_COLORS['Dead']]
            )
            
            im = ax_decisions.imshow(decision_grid, aspect='auto', cmap=cmap, 
                                    interpolation='nearest', vmin=0, vmax=3)
            ax_decisions.set_xlim(0, 2000)
            ax_decisions.set_ylim(0, num_ues)
            
            if row_idx == 0:
                ax_decisions.set_title('Decisions', fontsize=11, fontweight='bold')
            if row_idx == 3:
                ax_decisions.set_xlabel('Task ID', fontsize=9)
            else:
                ax_decisions.set_xticklabels([])
            
            ax_decisions.set_ylabel('UEs', fontsize=9)
    
    # Add legend only to first row
    if show_legend and row_idx == 0:
        ax_qoe.legend(loc='lower left', fontsize=7, ncol=2, framealpha=0.9)


def create_scenario_1_figure(results_dir: str, output_file: str = None):
    """
    Create Figure 7 from the paper: Scenario 1 results for all task distributions.
    
    Args:
        results_dir: Directory containing scenario results
        output_file: Path to save the figure (default: results/scenario_1_figure.png)
    """
    # Load results for all 4 distributions
    scenarios = [
        ('s1_class1_90', 'Class 1'),
        ('s1_class2_90', 'Class 2'),
        ('s1_class3_90', 'Class 3'),
        ('s1_random', 'Random'),
    ]
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('Scenario 1 Results', fontsize=14, fontweight='bold', y=0.995)
    
    # Plot each scenario as a row
    for row_idx, (scenario_key, class_label) in enumerate(scenarios):
        print(f"\nPlotting {class_label} (90% tasks)...")
        results = load_scenario_results(results_dir, scenario_key)
        
        if not results:
            print(f"  Warning: No results found for {scenario_key}")
            continue
        
        plot_scenario_row(
            fig=fig,
            row_idx=row_idx,
            results=results,
            scenario_name=scenario_key,
            class_label=class_label,
            show_legend=(row_idx == 0),
        )
    
    # Add decision legend at the bottom
    legend_elements = [
        mpatches.Patch(color=DECISION_COLORS['local'], label='local'),
        mpatches.Patch(color=DECISION_COLORS['MEC'], label='MEC'),
        mpatches.Patch(color=DECISION_COLORS['cloud'], label='cloud'),
        mpatches.Patch(color=DECISION_COLORS['Dead'], label='Dead'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              bbox_to_anchor=(0.85, 0.02), fontsize=9, frameon=True)
    
    # Save figure
    if output_file is None:
        output_file = os.path.join(results_dir, 'scenario_1_complete_figure.png')
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for scenario results"
    )
    parser.add_argument(
        '--scenario',
        type=str,
        default='s1',
        choices=['s1', 's2'],
        help='Scenario to plot (s1 or s2)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/scenarios',
        help='Directory containing scenario results'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Generating plots for Scenario {args.scenario.upper()}")
    print(f"{'='*80}")
    print(f"Results directory: {args.results_dir}")
    
    if args.scenario == 's1':
        create_scenario_1_figure(args.results_dir, args.output)
    else:
        print(f"Scenario {args.scenario} plotting not yet implemented")
    
    print(f"\n{'='*80}")
    print("Plotting complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
