#!/usr/bin/env python3
"""
run_baselines_scenario.py

Run baseline policies (local, MEC, cloud, random) on specific scenarios
for comparison with RL agents. Matches the paper's evaluation methodology.

Supports both Scenario 1 (MEC unavailability) and Scenario 2 (communication failure).

Usage:
    # Run all baselines on a scenario
    python run_baselines_scenario.py --scenario s1_class1_90 --all
    
    # Run specific baseline
    python run_baselines_scenario.py --scenario s2_class1_90 --policy mec
    
    # Run with custom timesteps
    python run_baselines_scenario.py --scenario s1_class1_90 --policy local --timesteps 2000
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from scenario_config import get_scenario, list_scenarios, ALL_SCENARIOS
from EnvConfig import EnvConfig
from models import UE, BaseStation, MECServer, CloudServer, TaskFactory


class BaselineSimulator:
    """
    Simulates baseline policies on scenarios with MEC unavailability
    and communication failure support.
    """
    
    def __init__(self, scenario_key: str):
        self.scenario = get_scenario(scenario_key)
        self.scenario_key = scenario_key
        
        # Create entities
        self.ue = UE(n=0, x_m=50.0, y_m=50.0)
        self.bs = BaseStation()
        self.mec = MECServer()
        self.cloud = CloudServer()
        
        # Task factory based on scenario distribution
        self.task_factory = TaskFactory(mode="random")  # Will sample based on distribution
    
    def sample_task(self):
        """Sample a task according to scenario distribution."""
        # Choose class based on scenario distribution
        class_probs = self.scenario.task_distribution
        task_class = np.random.choice([1, 2, 3], p=class_probs)
        
        # Create task factory for that class
        factory = TaskFactory(mode="fixed", fixed_class=task_class)
        return factory.sample()
    
    def is_mec_available(self, timestep: int) -> bool:
        """Check if MEC is available at given timestep."""
        return self.scenario.is_mec_available(timestep)
    
    def has_communication(self, timestep: int) -> bool:
        """Check if communication is available at given timestep."""
        return self.scenario.has_communication(timestep)
    
    def get_channel_quality(self, timestep: int) -> float:
        """Get channel quality multiplier at given timestep."""
        return self.scenario.get_channel_quality_multiplier(timestep)
    
    def run_policy(
        self,
        policy: str,
        timesteps: int = None,
    ) -> Dict:
        """
        Run a baseline policy on the scenario.
        
        Args:
            policy: 'local', 'mec', 'cloud', or 'random'
            timesteps: Number of timesteps (default: from scenario)
        
        Returns:
            Dictionary with metrics
        """
        if timesteps is None:
            timesteps = self.scenario.total_timesteps
        
        print(f"\nRunning {policy.upper()} policy on {self.scenario_key}...")
        
        # Reset UE
        self.ue.battery_j = EnvConfig.UE_MAX_BATTERY
        
        # Metrics storage
        metrics = {
            "QoE": [],
            "Latency": [],
            "Energy": [],
            "Battery": [],
            "OffloadRatio": [],
            "Action": [],
            "TaskClass": [],
            "Success": [],
        }
        
        for t in range(timesteps):
            if (t + 1) % 200 == 0:
                print(f"  Progress: {t+1}/{timesteps} ({100*(t+1)/timesteps:.1f}%)")
            
            # Check if battery is depleted
            if self.ue.battery_j <= 0:
                # Fill remaining with failure metrics
                for _ in range(timesteps - t):
                    metrics["QoE"].append(EnvConfig.FAIL_PENALTY)
                    metrics["Latency"].append(0.0)
                    metrics["Energy"].append(0.0)
                    metrics["Battery"].append(0.0)
                    metrics["OffloadRatio"].append(0.0)
                    metrics["Action"].append(-1)  # Dead
                    metrics["TaskClass"].append(0)
                    metrics["Success"].append(0)
                break
            
            # Apply idle drain
            self.ue.drain_idle()
            
            # Sample task
            task = self.sample_task()
            
            # Store battery before execution
            battery_before = self.ue.battery_j
            
            # Get channel quality and communication status
            channel_quality = self.get_channel_quality(t)
            has_communication = self.has_communication(t)
            mec_available = self.is_mec_available(t)
            
            # Decide action based on policy
            if policy == "random":
                action = np.random.choice([0, 1, 2])
            elif policy == "local":
                action = 0
            elif policy == "mec":
                action = 1
            elif policy == "cloud":
                action = 2
            else:
                raise ValueError(f"Unknown policy: {policy}")
            
            # Execute action
            if action == 0:  # Local - always works
                latency = self.ue.local_latency(task.cpu_cycles)
                energy = self.ue.local_energy(task.cpu_cycles)
                
            elif action == 1:  # MEC
                # Check if offloading is possible
                if not has_communication:
                    # SCENARIO 2: Communication failure → cannot reach MEC
                    latency = task.latency_deadline * 10.0
                    energy = 0.0
                elif not mec_available:
                    # SCENARIO 1: MEC unavailable
                    latency = task.latency_deadline * 10.0
                    energy = 0.0
                else:
                    # Normal MEC offloading
                    latency, energy = self.ue.offload_to_mec(
                        task, self.bs, self.mec,
                        n_ues=EnvConfig.NUM_UES,
                        channel_quality_multiplier=channel_quality,
                    )
                    
            else:  # Cloud (action == 2)
                # Check if offloading is possible
                if not has_communication:
                    # SCENARIO 2: Communication failure → cannot reach Cloud
                    latency = task.latency_deadline * 10.0
                    energy = 0.0
                else:
                    # Normal cloud offloading
                    latency, energy = self.ue.offload_to_cloud(
                        task, self.bs, self.cloud,
                        n_ues=EnvConfig.NUM_UES,
                        channel_quality_multiplier=channel_quality,
                    )
            
            # Update battery
            self.ue.battery_j = max(self.ue.battery_j - energy, 0.0)
            
            # Compute success and QoE
            success = latency <= task.latency_deadline
            
            if success:
                if battery_before > 0:
                    qoe = -(energy / battery_before)
                else:
                    qoe = EnvConfig.FAIL_PENALTY
            else:
                qoe = EnvConfig.FAIL_PENALTY
            
            offload = 1.0 if action in (1, 2) else 0.0
            
            # Store metrics
            metrics["QoE"].append(qoe)
            metrics["Latency"].append(latency)
            metrics["Energy"].append(energy)
            metrics["Battery"].append(self.ue.battery_j)
            metrics["OffloadRatio"].append(offload)
            metrics["Action"].append(action)
            metrics["TaskClass"].append(task.cls)
            metrics["Success"].append(int(success))
        
        return metrics


def save_baseline_results(
    scenario_key: str,
    policy: str,
    metrics: Dict,
    output_dir: str = "results/scenarios",
):
    """
    Save baseline results to CSV.
    
    Args:
        scenario_key: Scenario identifier
        policy: Policy name
        metrics: Dictionary of metrics
        output_dir: Output directory
    """
    df = pd.DataFrame(metrics)
    
    scenario_dir = os.path.join(output_dir, scenario_key)
    os.makedirs(scenario_dir, exist_ok=True)
    
    csv_path = os.path.join(scenario_dir, f"{policy}_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Saved {policy.upper()} results to {csv_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"{policy.upper()} POLICY SUMMARY")
    print(f"{'='*80}")
    print(f"Mean QoE: {df['QoE'].mean():.6f}")
    print(f"Mean Latency: {df['Latency'].mean():.6f} s")
    print(f"Mean Energy: {df['Energy'].mean():.3f} J")
    print(f"Final Battery: {df['Battery'].iloc[-1]:.2f} J")
    print(f"Success Rate: {df['Success'].mean():.1%}")
    print(f"Offload Ratio: {df['OffloadRatio'].mean():.3f}")
    print(f"{'='*80}\n")
    
    return df


def run_all_baselines(
    scenario_key: str,
    timesteps: int = None,
    output_dir: str = "results/scenarios",
):
    """
    Run all baseline policies on a scenario.
    
    Args:
        scenario_key: Scenario identifier
        timesteps: Number of timesteps (default: from scenario)
        output_dir: Output directory
    """
    print(f"\n{'='*80}")
    print(f"RUNNING ALL BASELINES ON: {scenario_key}")
    print(f"{'='*80}\n")
    
    simulator = BaselineSimulator(scenario_key)
    policies = ["local", "mec", "cloud", "random"]
    
    results_summary = []
    
    for policy in policies:
        print(f"\n{'#'*80}")
        print(f"# Running {policy.upper()} policy")
        print(f"{'#'*80}")
        
        metrics = simulator.run_policy(policy, timesteps=timesteps)
        df = save_baseline_results(scenario_key, policy, metrics, output_dir)
        
        results_summary.append({
            'policy': policy,
            'mean_qoe': df['QoE'].mean(),
            'final_battery': df['Battery'].iloc[-1],
            'success_rate': df['Success'].mean(),
        })
    
    # Print summary table
    print(f"\n{'='*80}")
    print("BASELINE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Policy':<15} {'Mean QoE':>12} {'Final Battery':>15} {'Success Rate':>15}")
    print("-" * 80)
    
    for result in results_summary:
        print(f"{result['policy']:<15} "
              f"{result['mean_qoe']:>12.6f} "
              f"{result['final_battery']:>15.2f} J "
              f"{result['success_rate']:>14.1%}")
    
    print("-" * 80)
    print(f"\n✅ All baseline results saved to: {output_dir}/{scenario_key}/")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline policies on scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available scenarios",
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario key (e.g., 's1_class1_90', 's2_class1_90')",
    )
    
    parser.add_argument(
        "--policy",
        type=str,
        choices=["local", "mec", "cloud", "random"],
        help="Baseline policy to run",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all baseline policies",
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        help="Number of timesteps (default: from scenario config)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scenarios",
        help="Output directory (default: results/scenarios)",
    )
    
    args = parser.parse_args()
    
    # List scenarios
    if args.list:
        list_scenarios()
        return
    
    # Require scenario
    if not args.scenario:
        parser.print_help()
        print("\nERROR: --scenario is required (or use --list to see options)")
        sys.exit(1)
    
    # Validate scenario
    if args.scenario not in ALL_SCENARIOS:
        print(f"ERROR: Unknown scenario '{args.scenario}'")
        print(f"\nAvailable scenarios: {list(ALL_SCENARIOS.keys())}")
        print("\nUse --list to see full details")
        sys.exit(1)
    
    # Run all or specific policy
    if args.all:
        run_all_baselines(
            args.scenario,
            timesteps=args.timesteps,
            output_dir=args.output_dir,
        )
    elif args.policy:
        simulator = BaselineSimulator(args.scenario)
        metrics = simulator.run_policy(args.policy, timesteps=args.timesteps)
        save_baseline_results(
            args.scenario,
            args.policy,
            metrics,
            output_dir=args.output_dir,
        )
    else:
        print("ERROR: Must specify either --policy or --all")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
