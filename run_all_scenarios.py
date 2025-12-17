#!/usr/bin/env python3
"""
run_all_scenarios.py

Master script to run all scenarios with unified parameters.
Automatically trains/evaluates all scenario variants, runs baselines, and generates plots.

Supports:
- Scenario 1: MEC unavailability
- Scenario 2: Communication failure

Usage:
    # Run all Scenario 1 variants (RL + baselines + plots)
    python run_all_scenarios.py --scenario-set s1 --episodes 500 --all
    
    # Run all Scenario 2 variants
    python run_all_scenarios.py --scenario-set s2 --episodes 500 --all
    
    # Run specific scenarios only
    python run_all_scenarios.py --scenarios s1_class1_90 s2_class1_90 --episodes 300 --train
    
    # Evaluate existing models and plot
    python run_all_scenarios.py --scenario-set s1 --eval --plot
"""

import argparse
import os
import sys
import time

from run_baselines_scenario import run_all_baselines
from run_scenario import train_rl_on_scenario, evaluate_rl_on_scenario
from scenario_config import ALL_SCENARIOS, get_scenario, list_scenarios

SCENARIO_SETS = {
    's1': ['s1_class1_90', 's1_class2_90', 's1_class3_90', 's1_random'],
    's1_base': ['s1_base'],
    's2': ['s2_class1_90', 's2_class2_90', 's2_class3_90', 's2_random'],
    's2_base': ['s2_base'],
    'all': [
        's1_class1_90', 's1_class2_90', 's1_class3_90', 's1_random',
        's2_class1_90', 's2_class2_90', 's2_class3_90', 's2_random',
    ],
}


def run_all_scenarios(
    scenarios: list,
    episodes: int = 500,
    do_train: bool = True,
    do_eval: bool = True,
    do_baselines: bool = True,
    do_plot: bool = True,
    device: str = None,
    results_dir: str = "results/scenarios",
):
    """
    Run training/evaluation for multiple scenarios.
    
    Args:
        scenarios: List of scenario keys to run
        episodes: Number of training episodes
        do_train: Whether to train RL agents
        do_eval: Whether to evaluate RL agents
        do_baselines: Whether to run baseline comparisons
        do_plot: Whether to generate plots at the end
        device: Device to use ('cuda' or 'cpu')
        results_dir: Directory to save results
    """
    print("\n" + "="*80)
    print("RUNNING ALL SCENARIOS")
    print("="*80)
    print(f"Scenarios to run: {scenarios}")
    print(f"Training episodes: {episodes}")
    print(f"Train RL: {do_train}")
    print(f"Evaluate RL: {do_eval}")
    print(f"Run baselines: {do_baselines}")
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
            # --- Run Baselines First ---
            if do_baselines:
                print(f"\n{'='*80}")
                print(f"[{scenario_key}] Running Baseline Policies")
                print(f"{'='*80}\n")
                
                run_all_baselines(
                    scenario_key=scenario_key,
                    output_dir=results_dir,
                )
                print(f"[{scenario_key}] ✓ Baselines complete\n")
            
            # --- RL Training ---
            if do_train:
                print(f"\n{'='*80}")
                print(f"[{scenario_key}] Training RL Agent")
                print(f"{'='*80}\n")
                
                train_rl_on_scenario(
                    scenario_key=scenario_key,
                    episodes=episodes,
                    device=device,
                    save_dir=results_dir,
                )
                print(f"[{scenario_key}] ✓ RL training complete\n")
            
            # --- RL Evaluation ---
            if do_eval:
                print(f"\n{'='*80}")
                print(f"[{scenario_key}] Evaluating RL Agent")
                print(f"{'='*80}\n")
                
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
                    
                    print(f"[{scenario_key}] ✓ RL evaluation complete")
                    print(f"  Mean QoE: {mean_qoe:.6f}")
                    print(f"  Final Battery: {final_battery:.2f} J")
                    print(f"  Success Rate: {success_rate:.1%}\n")
                else:
                    print(f"[{scenario_key}] ✗ RL evaluation failed\n")
            
            scenario_elapsed = time.time() - scenario_start
            print(f"[{scenario_key}] Time elapsed: {scenario_elapsed/60:.1f} minutes\n")
            
        except Exception as e:
            print(f"\n[{scenario_key}] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"[{scenario_key}] Skipping to next scenario...\n")
            continue
    
    # Generate plots
    if do_plot:
        print("\n" + "="*80)
        print("GENERATING PLOTS")
        print("="*80 + "\n")
        
        try:
            # Determine which scenario set to plot
            has_s1 = any(s.startswith('s1_') or s == 's1_base' for s in scenarios)
            has_s2 = any(s.startswith('s2_') or s == 's2_base' for s in scenarios)
            
            if has_s1:
                from plot_scenario_results import create_scenario_1_figure
                print("Generating Scenario 1 complete figure...")
                create_scenario_1_figure(results_dir)
                print("✓ Scenario 1 plots generated\n")
            
            if has_s2:
                from plot_scenario_2_results import create_scenario_2_figure
                print("Generating Scenario 2 complete figure...")
                create_scenario_2_figure(results_dir)
                print("✓ Scenario 2 plots generated\n")
            
            if not (has_s1 or has_s2):
                print("No recognized scenario sets found for plotting\n")
                
        except Exception as e:
            print(f"ERROR generating plots: {str(e)}")
            import traceback
            traceback.print_exc()
            print("You can generate plots manually with:")
            print("  python plot_scenario_results.py      # For Scenario 1")
            print("  python plot_scenario_2_results.py    # For Scenario 2\n")
    
    # Print summary
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("ALL SCENARIOS COMPLETE")
    print("="*80)
    print(f"Total time: {total_elapsed/60:.1f} minutes\n")
    
    if results_summary:
        print("RL Agent Results Summary:")
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
  # Run all Scenario 1 variants (train + baselines + eval + plot)
  python run_all_scenarios.py --scenario-set s1 --episodes 500 --all
  
  # Run all Scenario 2 variants (communication failure)
  python run_all_scenarios.py --scenario-set s2 --episodes 500 --all
  
  # Run only Class 1 from both scenarios
  python run_all_scenarios.py --scenarios s1_class1_90 s2_class1_90 --episodes 300 --train
  
  # Evaluate existing models and generate plots
  python run_all_scenarios.py --scenario-set s1 --eval --plot
  
  # Quick test with fewer episodes
  python run_all_scenarios.py --scenario-set s2 --episodes 50 --train --eval
  
  # Run baselines only (no RL training)
  python run_all_scenarios.py --scenario-set s2 --baselines --plot
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
        help='Train RL agents on scenarios'
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Evaluate trained RL agents'
    )
    parser.add_argument(
        '--baselines',
        action='store_true',
        help='Run baseline policies (local, MEC, cloud, random)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots after evaluation'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Shortcut for --train --eval --baselines --plot'
    )
    parser.add_argument(
        '--no-baselines',
        action='store_true',
        help='Skip baseline policies (when using --all)'
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
    if args.all:
        do_train = True
        do_eval = True
        do_baselines = not args.no_baselines
        do_plot = True
    else:
        do_train = args.train
        do_eval = args.eval
        do_baselines = args.baselines
        do_plot = args.plot
    
    # Require at least one action
    if not (do_train or do_eval or do_baselines or do_plot):
        print("ERROR: Must specify at least one of --train, --eval, --baselines, --plot, or --all")
        parser.print_help()
        sys.exit(1)
    
    # Run scenarios
    success = run_all_scenarios(
        scenarios=scenarios,
        episodes=args.episodes,
        do_train=do_train,
        do_eval=do_eval,
        do_baselines=do_baselines,
        do_plot=do_plot,
        device=args.device,
        results_dir=args.results_dir,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
