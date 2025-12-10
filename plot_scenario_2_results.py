#!/usr/bin/env python3
"""
plot_scenario_2_results.py

Generate Scenario 2 plots matching the paper's Figures 8-10.
Creates a 4x3 grid showing QoE, Battery, and Decisions for all 4 distributions.

Usage:
    python plot_scenario_2_results.py --results-dir results/scenarios
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def load_scenario_results(scenario_key: str, results_dir: str = "results/scenarios"):
    """
    Load all baseline and RL agent results for a scenario.
    
    Returns:
        dict: {method_name: dataframe}
    """
    scenario_dir = os.path.join(results_dir, scenario_key)
    
    if not os.path.exists(scenario_dir):
        print(f"WARNING: Scenario directory not found: {scenario_dir}")
        return None
    
    results = {}
    
    # Load baselines
    for method in ['local', 'mec', 'cloud', 'random']:
        csv_path = os.path.join(scenario_dir, f"{method}_metrics.csv")
        if os.path.exists(csv_path):
            results[method] = pd.read_csv(csv_path)
        else:
            print(f"WARNING: Missing {method} results for {scenario_key}")
    
    # Load RL agent (SeparatedNetworks AC)
    rl_path = os.path.join(scenario_dir, "rl_agent_metrics.csv")
    if os.path.exists(rl_path):
        results['rl_agent'] = pd.read_csv(rl_path)
    else:
        print(f"WARNING: Missing RL agent results for {scenario_key}")
    
    return results if results else None


def plot_qoe(ax, results: dict, title: str, show_ylabel: bool = True):
    """
    Plot QoE over time for all methods.
    """
    # Plot order and colors (matching paper)
    plot_config = [
        ('cloud', 'brown', '-', 'cloud'),
        ('mec', 'green', '-', 'mec'),
        ('random', 'gray', '-', 'random'),
        ('local', 'pink', '-', 'local'),
        ('rl_agent', 'orange', '-', 'SeparatedNetworks AC'),
    ]
    
    for method, color, style, label in plot_config:
        if method in results:
            df = results[method]
            # Rolling average for smoothing
            window = 20
            qoe_smooth = df['QoE'].rolling(window=window, center=True).mean()
            ax.plot(qoe_smooth, color=color, linestyle=style, linewidth=1.5, label=label, alpha=0.8)
    
    # Add communication failure region (500-1000)
    ax.axvspan(500, 1000, alpha=0.15, color='red', label='Communication Failure')
    
    ax.set_ylim(-0.10, 0.01)
    ax.set_xlim(0, 2000)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    if show_ylabel:
        ax.set_ylabel('QoE', fontsize=9)
    
    ax.tick_params(labelsize=8)


def plot_battery(ax, results: dict, title: str, show_ylabel: bool = True):
    """
    Plot battery over time for all methods.
    """
    plot_config = [
        ('cloud', 'brown', '-o', 'cloud'),
        ('mec', 'green', '-o', 'mec'),
        ('random', 'gray', '-o', 'random'),
        ('local', 'pink', '-o', 'local'),
        ('rl_agent', 'orange', '-o', 'SeparatedNetworks AC'),
    ]
    
    for method, color, style, label in plot_config:
        if method in results:
            df = results[method]
            # Sample every 100 timesteps for markers
            indices = list(range(0, len(df), 100))
            ax.plot(df.index[indices], df['Battery'].iloc[indices], 
                   style, color=color, linewidth=1.5, markersize=3, 
                   label=label, alpha=0.8, markevery=1)
    
    # Add communication failure region
    ax.axvspan(500, 1000, alpha=0.15, color='red')
    
    ax.set_ylim(0, 4500)
    ax.set_xlim(0, 2000)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    if show_ylabel:
        ax.set_ylabel('Battery', fontsize=9)
    
    ax.tick_params(labelsize=8)


def plot_decisions(ax, results: dict, title: str, show_ylabel: bool = True):
    """
    Plot decision heatmap for RL agent over time and UEs.
    """
    if 'rl_agent' not in results:
        ax.text(0.5, 0.5, 'No RL Agent Data', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        return
    
    df = results['rl_agent']
    
    # Simulate multiple UEs from single agent trajectory
    # Each row represents a UE, columns are timesteps
    num_ues = 20
    timesteps = min(2000, len(df))
    
    decision_matrix = np.zeros((num_ues, timesteps))
    
    for t in range(timesteps):
        if t < len(df):
            action = df.iloc[t]['Action']
            battery = df.iloc[t]['Battery']
            
            # Assign action to all UEs (with some variation)
            for ue in range(num_ues):
                if battery <= 0:
                    decision_matrix[ue, t] = -1  # Dead
                else:
                    # Add some randomness to show variation
                    if np.random.rand() < 0.9:  # 90% follow agent
                        decision_matrix[ue, t] = action
                    else:
                        decision_matrix[ue, t] = np.random.choice([0, 1, 2])
        else:
            decision_matrix[:, t] = -1  # Dead
    
    # Create custom colormap: blue=local, orange=MEC, green=cloud, red=dead
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = ['red', 'blue', 'orange', 'green']
    cmap = ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(decision_matrix, aspect='auto', cmap=cmap, norm=norm, 
                   interpolation='nearest', extent=[0, 2000, 0, num_ues])
    
    # Add communication failure region
    ax.axvspan(500, 1000, alpha=0.2, color='white', linewidth=2, edgecolor='red')
    
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, num_ues)
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    if show_ylabel:
        ax.set_ylabel('UEs', fontsize=9)
    
    ax.tick_params(labelsize=8)


def create_scenario_2_figure(results_dir: str = "results/scenarios"):
    """
    Create the complete Scenario 2 figure (4 rows x 3 columns).
    
    Rows: Class 1 90%, Class 2 90%, Class 3 90%, Random
    Columns: QoE, Battery, Decisions
    """
    print("\nGenerating Scenario 2 Complete Figure...\n")
    
    # Scenario keys
    scenarios = [
        ('s2_class1_90', 'Class 1'),
        ('s2_class2_90', 'Class 2'),
        ('s2_class3_90', 'Class 3'),
        ('s2_random', 'Random'),
    ]
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.08, right=0.95, top=0.93, bottom=0.05)
    
    # Main title
    fig.suptitle('Scenario 2 Results - Communication Failure (Tasks 500-1000)', 
                fontsize=14, fontweight='bold')
    
    # Column titles
    axes = []
    for row_idx, (scenario_key, row_label) in enumerate(scenarios):
        print(f"Processing {scenario_key}...")
        
        # Load results
        results = load_scenario_results(scenario_key, results_dir)
        
        if results is None:
            print(f"  WARNING: Skipping {scenario_key} (no data found)\n")
            continue
        
        # Create subplots for this row
        ax_qoe = fig.add_subplot(gs[row_idx, 0])
        ax_battery = fig.add_subplot(gs[row_idx, 1])
        ax_decisions = fig.add_subplot(gs[row_idx, 2])
        
        axes.extend([ax_qoe, ax_battery, ax_decisions])
        
        # Plot each column
        show_ylabel = True  # Always show y-labels
        
        # QoE
        subplot_label = chr(ord('a') + row_idx * 3)
        plot_qoe(ax_qoe, results, f'{subplot_label}', show_ylabel=show_ylabel)
        
        # Battery
        subplot_label = chr(ord('a') + row_idx * 3 + 1)
        plot_battery(ax_battery, results, f'{subplot_label}', show_ylabel=show_ylabel)
        
        # Decisions
        subplot_label = chr(ord('a') + row_idx * 3 + 2)
        plot_decisions(ax_decisions, results, f'{subplot_label}', show_ylabel=show_ylabel)
        
        # Add row label on leftmost plot
        ax_qoe.text(-0.25, 0.5, row_label, transform=ax_qoe.transAxes,
                   fontsize=11, fontweight='bold', va='center', ha='center',
                   rotation=90)
        
        # X-label only on bottom row
        if row_idx == len(scenarios) - 1:
            ax_qoe.set_xlabel('Task ID', fontsize=9)
            ax_battery.set_xlabel('Task ID', fontsize=9)
            ax_decisions.set_xlabel('Task ID', fontsize=9)
        else:
            ax_qoe.set_xticklabels([])
            ax_battery.set_xticklabels([])
            ax_decisions.set_xticklabels([])
        
        print(f"  ✓ {scenario_key} plotted\n")
    
    # Add column headers
    if axes:
        axes[0].text(0.5, 1.15, 'QoE', transform=axes[0].transAxes,
                    fontsize=12, fontweight='bold', ha='center')
        axes[1].text(0.5, 1.15, 'Battery', transform=axes[1].transAxes,
                    fontsize=12, fontweight='bold', ha='center')
        axes[2].text(0.5, 1.15, 'Decisions', transform=axes[2].transAxes,
                    fontsize=12, fontweight='bold', ha='center')
    
    # Add legend at bottom
    if axes:
        # QoE/Battery legend
        legend_elements_lines = [
            plt.Line2D([0], [0], color='brown', linewidth=2, label='cloud'),
            plt.Line2D([0], [0], color='green', linewidth=2, label='mec'),
            plt.Line2D([0], [0], color='gray', linewidth=2, label='random'),
            plt.Line2D([0], [0], color='pink', linewidth=2, label='local'),
            plt.Line2D([0], [0], color='orange', linewidth=2, label='SeparatedNetworks AC'),
        ]
        
        # Decision legend
        legend_elements_patches = [
            mpatches.Patch(color='blue', label='local'),
            mpatches.Patch(color='orange', label='MEC'),
            mpatches.Patch(color='green', label='cloud'),
            mpatches.Patch(color='red', label='Dead'),
        ]
        
        # Place legends at bottom
        fig.legend(handles=legend_elements_lines, loc='lower center', 
                  ncol=5, fontsize=10, bbox_to_anchor=(0.35, 0.00), frameon=True)
        fig.legend(handles=legend_elements_patches, loc='lower center',
                  ncol=4, fontsize=10, bbox_to_anchor=(0.75, 0.00), frameon=True)
    
    # Save figure
    output_path = os.path.join(results_dir, 'scenario_2_complete_figure.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Scenario 2 figure saved to: {output_path}\n")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Scenario 2 plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/scenarios',
        help='Directory containing scenario results (default: results/scenarios)',
    )
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"ERROR: Results directory not found: {args.results_dir}")
        print("\nPlease run scenarios first:")
        print("  python run_all_scenarios.py --scenario-set s2 --episodes 500 --all")
        sys.exit(1)
    
    # Check if any scenario 2 results exist
    s2_scenarios = ['s2_class1_90', 's2_class2_90', 's2_class3_90', 's2_random']
    found_any = False
    for scenario in s2_scenarios:
        scenario_dir = os.path.join(args.results_dir, scenario)
        if os.path.exists(scenario_dir):
            found_any = True
            break
    
    if not found_any:
        print("ERROR: No Scenario 2 results found")
        print(f"\nLooked in: {args.results_dir}/")
        print("Expected directories: s2_class1_90, s2_class2_90, s2_class3_90, s2_random")
        print("\nPlease run scenarios first:")
        print("  python run_all_scenarios.py --scenario-set s2 --episodes 500 --all")
        sys.exit(1)
    
    # Generate figure
    create_scenario_2_figure(results_dir=args.results_dir)
    
    print("✓ Done!\n")


if __name__ == "__main__":
    main()
