#!/usr/bin/env python3
"""
run_all_scenarios.py

Master script to run all scenarios with unified parameters.
Automatically trains/evaluates all scenario variants and generates plots.

Usage:
    # Run all Scenario 1 variants
    python run_all_scenarios.py --scenario-set s1 --episodes 500 --train --eval --plot
    
    # Run specific scenarios only
    python run_all_scenarios.py --scenarios s1_class1_90 s1_class2_90 --episodes 300 --train
    
    # Evaluate existing models and plot
    python run_all_scenarios.py --scenario-set s1 --eval --plot
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

from scenario_config import ALL_SCENARIOS, get_scenario, list_scenarios
from run_scenario import train_rl_on_scenario, evaluate_rl_on_scenario


SCENARIO_SETS = {
    's1': ['s1_class1_90', 's1_class2_90', 's1_class3_90', 's1_random'],
    's1_base': ['s1_base'],
    's2': ['s2_base'],
    'all': ['s1_class1_90', 's1_class2_90', 's1_class3_90', 's1_random', 's2_base'],
}


def run_all_scenarios(
    scenarios: list,
    episodes: int = 500,
    do_train: bool = True,
    do_eval: bool = True,
    do_plot: bool = True,
    device: str = None,
    results_dir: str = "results/scenarios",
):
    """
    Run training/evaluation for multiple scenarios.
    
    Args:
        scenarios: List of scenario keys to run
        episodes: Number of training episodes
        do_train: Whether to train agents
        do_eval: Whether to evaluate agents
        do_plot: Whether to generate plots at the end
        device: Device to use ('cuda' or 'cpu')
        results_dir: Directory to save results
    """
    print("\n" + "="*80)
    print("RUNNING ALL SCENARIOS")
    print("="*80)
    print(f"Scenarios to run: {scenarios}")
    print(f"Training episodes: {episodes}")
    print(f"Train: {do_train}")
    print(f"Evaluate: {do_eval}")
    print(f"Plot: {do_plot}")
    print(f"Device: {device or 'auto'}")
    print("="*80 + "\n")
    
    # Validate scenarios
    for scenario in scenarios:
        if scenario not in ALL_SCENARIOS:
            print(f"ERROR: Unknown scenario '{scenario}'")
            print(f"Available: {list(ALL_SCENARIOS.keys())}")
            return False
    
    # Track timing and results
    start_time = time.time()
    results_summary = []
    
    # Run each scenario
    for idx, scenario_key in enumerate(scenarios, 1):
        scenario_start = time.time()
        
        print("\n" + "#"*80)
        print(f"# SCENARIO {idx}/{len(scenarios)}: {scenario_key}")
        print("#"*80 + "\n")
        
        scenario_config = get_scenario(scenario_key)
        print(f"Name: {scenario_config.name}")
        print(f"Description: {scenario_config.description}")
        print(f"Task distribution: Class1={scenario_config.task_distribution[0]:.0%}, "
              f"Class2={scenario_config.task_distribution[1]:.0%}, "
              f"Class3={scenario_config.task_distribution[2]:.0%}")
        print()
        
        try:
            # Training
            if do_train:
                print(f"[{scenario_key}] Starting training...")
                train_rl_on_scenario(
                    scenario_key=scenario_key,
                    episodes=episodes,
                    device=device,
                    save_dir=results_dir,
                )
                print(f"[{scenario_key}] ✓ Training complete\n")
            
            # Evaluation
            if do_eval:
                print(f"[{scenario_key}] Starting evaluation...")
                eval_df = evaluate_rl_on_scenario(
                    scenario_key=scenario_key,
                    device=device,
                    output_dir=results_dir,
                )
                
                if eval_df is not None:
                    mean_qoe = eval_df['QoE'].mean()
                    final_battery = eval_df['Battery'].iloc[-1]
                    success_rate = eval_df['Success'].mean()
                    
                    results_summary.append({
                        'scenario': scenario_key,
                        'mean_qoe': mean_qoe,
                        'final_battery': final_battery,
                        'success_rate': success_rate,
                    })
                    
                    print(f"[{scenario_key}] ✓ Evaluation complete")
                    print(f"  Mean QoE: {mean_qoe:.6f}")
                    print(f"  Final Battery: {final_battery:.2f} J")
                    print(f"  Success Rate: {success_rate:.1%}\n")
                else:
                    print(f"[{scenario_key}] ✗ Evaluation failed\n")
            
            scenario_elapsed = time.time() - scenario_start
            print(f"[{scenario_key}] Time elapsed: {scenario_elapsed/60:.1f} minutes\n")
            
        except Exception as e:
            print(f"\n[{scenario_key}] ERROR: {str(e)}")
            print(f"[{scenario_key}] Skipping to next scenario...\n")
            continue
    
    # Generate plots
    if do_plot:
        print("\n" + "="*80)
        print("GENERATING PLOTS")
        print("="*80 + "\n")
        
        try:
            # Determine which scenario set to plot
            if all(s.startswith('s1_') or s == 's1_base' for s in scenarios):
                from plot_scenario_results import create_scenario_1_figure
                print("Generating Scenario 1 complete figure...")
                create_scenario_1_figure(results_dir)
                print("✓ Scenario 1 plots generated\n")
            else:
                print("Plotting for mixed scenarios not yet implemented")
                print("Run plot_scenario_results.py manually for custom plots\n")
                
        except Exception as e:
            print(f"ERROR generating plots: {str(e)}")
            print("You can generate plots manually with: python plot_scenario_results.py\n")
    
    # Print summary
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("ALL SCENARIOS COMPLETE")
    print("="*80)
    print(f"Total time: {total_elapsed/60:.1f} minutes\n")
    
    if results_summary:
        print("Results Summary:")
        print("-" * 80)
        print(f"{'Scenario':<20} {'Mean QoE':>12} {'Final Battery':>15} {'Success Rate':>15}")
        print("-" * 80)
        
        for result in results_summary:
            print(f"{result['scenario']:<20} "
                  f"{result['mean_qoe']:>12.6f} "
                  f"{result['final_battery']:>15.2f} J "
                  f"{result['success_rate']:>14.1%}")
        
        print("-" * 80)
    
    print(f"\nResults saved to: {os.path.abspath(results_dir)}/")
    print("="*80 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run all scenarios with unified parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all Scenario 1 variants (train + eval + plot)
  python run_all_scenarios.py --scenario-set s1 --episodes 500 --train --eval --plot
  
  # Run only Class 1 and Class 2 (train only)
  python run_all_scenarios.py --scenarios s1_class1_90 s1_class2_90 --episodes 300 --train
  
  # Evaluate existing models and generate plots
  python run_all_scenarios.py --scenario-set s1 --eval --plot
  
  # Quick test with fewer episodes
  python run_all_scenarios.py --scenario-set s1 --episodes 50 --train --eval
        """
    )
    
    # Scenario selection
    scenario_group = parser.add_mutually_exclusive_group(required=True)
    scenario_group.add_argument(
        '--scenario-set',
        type=str,
        choices=list(SCENARIO_SETS.keys()),
        help='Predefined set of scenarios to run'
    )
    scenario_group.add_argument(
        '--scenarios',
        nargs='+',
        help='Specific scenario keys to run'
    )
    scenario_group.add_argument(
        '--list',
        action='store_true',
        help='List all available scenarios and exit'
    )
    
    # Training/evaluation options
    parser.add_argument(
        '--episodes',
        type=int,
        default=500,
        help='Number of training episodes (default: 500)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train agents on scenarios'
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Evaluate trained agents'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots after evaluation'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Shortcut for --train --eval --plot'
    )
    
    # Device options
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    
    # Output options
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/scenarios',
        help='Directory to save results (default: results/scenarios)'
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_scenarios()
        print("\nPredefined Scenario Sets:")
        print("="*80)
        for set_name, scenarios in SCENARIO_SETS.items():
            print(f"\n{set_name}:")
            for s in scenarios:
                print(f"  - {s}")
        print("\n" + "="*80)
        return
    
    # Determine scenarios to run
    if args.scenario_set:
        scenarios = SCENARIO_SETS[args.scenario_set]
    else:
        scenarios = args.scenarios
    
    # Determine what to do
    do_train = args.train or args.all
    do_eval = args.eval or args.all
    do_plot = args.plot or args.all
    
    # Require at least one action
    if not (do_train or do_eval or do_plot):
        print("ERROR: Must specify at least one of --train, --eval, --plot, or --all")
        parser.print_help()
        sys.exit(1)
    
    # Run scenarios
    success = run_all_scenarios(
        scenarios=scenarios,
        episodes=args.episodes,
        do_train=do_train,
        do_eval=do_eval,
        do_plot=do_plot,
        device=args.device,
        results_dir=args.results_dir,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
