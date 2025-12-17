#!/usr/bin/env python3
"""
run_scenario.py

Runs simulations for specific scenarios from the paper.
Supports:
- Training RL agents on specific scenarios
- Evaluating trained agents on scenarios
- Running baseline comparisons
- Generating scenario-specific plots

Usage:
    # List available scenarios
    python run_scenario.py --list
    
    # Train RL on Scenario 1 with 90% Class 1 tasks
    python run_scenario.py --scenario s1_class1_90 --train --episodes 500
    
    # Evaluate all baselines + RL on a scenario
    python run_scenario.py --scenario s1_class1_90 --eval-all
    
    # Run full pipeline (train + eval + plots)
    python run_scenario.py --scenario s1_class1_90 --full-pipeline
"""

import argparse
import os
import sys

import numpy as np
import torch

from offload_rl.EnvConfig import EnvConfig
from offload_rl.ac_agent import ActorCriticAgent
from offload_rl.models import UE, BaseStation, MECServer, CloudServer
from offload_rl.rl_env import OffloadEnv
from run_baselines_scenario import run_all_baselines
from scenario_config import get_scenario, list_scenarios, ALL_SCENARIOS


def make_scenario_env(scenario_key: str, ue_index: int = 0) -> OffloadEnv:
    """
    Create an RL environment configured for the given scenario.
    
    Args:
        scenario_key: Scenario identifier (e.g., 's1_class1_90')
        ue_index: Index of the UE (for multi-UE setups)
    
    Returns:
        Configured OffloadEnv instance
    """
    scenario = get_scenario(scenario_key)
    
    # Create entities
    ue = UE(n=ue_index, x_m=50.0, y_m=50.0)
    bs = BaseStation()
    mec = MECServer()
    cloud = CloudServer(fiber_distance_m=EnvConfig.CLOUD_FIBER_DISTANCE)
    
    # Create environment with scenario
    env = OffloadEnv(
        ue=ue,
        bs=bs,
        mec=mec,
        cloud=cloud,
        max_steps=scenario.total_timesteps,
        scenario_config=scenario,
    )
    
    return env


def train_rl_on_scenario(
    scenario_key: str,
    episodes: int = 500,
    gamma: float = 0.99,
    lr_actor: float = 1e-5,
    lr_critic: float = 1e-4,
    device: str = None,
    save_dir: str = "results/scenarios",
):
    """
    Train an Actor-Critic agent on a specific scenario.
    
    Args:
        scenario_key: Scenario identifier
        episodes: Number of training episodes
        gamma: Discount factor
        lr_actor: Actor learning rate
        lr_critic: Critic learning rate
        device: 'cuda' or 'cpu'
        save_dir: Directory to save trained models
    """
    from train_rl import print_gpu_info
    
    # Print diagnostics
    print_gpu_info()
    
    # Device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[train_rl_on_scenario] Using device: {device}")
    
    # Get scenario config
    scenario = get_scenario(scenario_key)
    print(f"\n{'='*80}")
    print(f"TRAINING ON SCENARIO: {scenario.name}")
    print(f"{'='*80}")
    print(f"Description: {scenario.description}")
    print(f"Task Distribution: Class1={scenario.task_distribution[0]:.1%}, "
          f"Class2={scenario.task_distribution[1]:.1%}, "
          f"Class3={scenario.task_distribution[2]:.1%}")
    print(f"Total timesteps: {scenario.total_timesteps}")
    print(f"Training episodes: {episodes}")
    print(f"{'='*80}\n")
    
    # Create environment
    env = make_scenario_env(scenario_key)
    init_state = env.reset()
    state_dim = init_state.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action space: 3 (local, MEC, cloud)\n")
    
    # Create agent
    agent = ActorCriticAgent(
        state_dim=state_dim,
        n_actions=3,
        gamma=gamma,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        device=device,
    )
    
    print(f"Actor device: {next(agent.actor.parameters()).device}")
    print(f"Critic device: {next(agent.critic.parameters()).device}\n")
    
    # Training loop
    all_returns = []
    all_steps = []
    
    print("Starting training...\n")
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_return += reward
            steps += 1
            
            if steps > 100000:
                print(f"WARNING: Episode {ep+1} exceeded 100k steps. Breaking.")
                break
        
        all_returns.append(ep_return)
        all_steps.append(steps)
        
        if (ep + 1) % 10 == 0 or ep == 0:
            avg_return = np.mean(all_returns[-10:])
            avg_steps = np.mean(all_steps[-10:])
            print(
                f"Episode {ep+1:4d}/{episodes} | "
                f"steps={steps:5d} | "
                f"return={ep_return:9.3f} | "
                f"avg_return(10)={avg_return:9.3f} | "
                f"battery={info.get('battery', 0):7.2f} J"
            )
            sys.stdout.flush()
    
    # Training summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Scenario: {scenario.name}")
    print(f"Total episodes: {episodes}")
    print(f"Final episode return: {all_returns[-1]:.3f}")
    print(f"Avg return (last 10): {np.mean(all_returns[-10:]):.3f}")
    print(f"Best episode return: {max(all_returns):.3f}")
    print(f"{'='*80}\n")
    
    # Save models
    os.makedirs(save_dir, exist_ok=True)
    scenario_dir = os.path.join(save_dir, scenario_key)
    os.makedirs(scenario_dir, exist_ok=True)
    
    actor_path = os.path.join(scenario_dir, "actor.pt")
    critic_path = os.path.join(scenario_dir, "critic.pt")
    
    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)
    
    print(f"✅ Saved actor to {actor_path}")
    print(f"✅ Saved critic to {critic_path}\n")
    
    return all_returns


def evaluate_rl_on_scenario(
    scenario_key: str,
    model_dir: str = None,
    device: str = None,
    output_dir: str = "results/scenarios",
):
    """
    Evaluate a trained RL agent on a scenario.
    
    Args:
        scenario_key: Scenario identifier
        model_dir: Directory containing actor.pt and critic.pt (default: results/scenarios/{scenario_key})
        device: 'cuda' or 'cpu'
        output_dir: Directory to save evaluation results
    """
    import pandas as pd
    
    # Device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model directory
    if model_dir is None:
        model_dir = os.path.join("results/scenarios", scenario_key)
    
    actor_path = os.path.join(model_dir, "actor.pt")
    if not os.path.exists(actor_path):
        print(f"ERROR: Trained model not found at {actor_path}")
        print(f"Please train first using: python run_scenario.py --scenario {scenario_key} --train")
        return None
    
    # Get scenario
    scenario = get_scenario(scenario_key)
    print(f"\n{'='*80}")
    print(f"EVALUATING RL AGENT ON: {scenario.name}")
    print(f"{'='*80}")
    print(f"Loading model from: {model_dir}\n")
    
    # Create environment and agent
    env = make_scenario_env(scenario_key)
    init_state = env.reset()
    state_dim = init_state.shape[0]
    
    agent = ActorCriticAgent(
        state_dim=state_dim,
        n_actions=3,
        device=device,
    )
    
    # Load trained weights
    agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    agent.actor.eval()
    print(f"✅ Loaded trained actor from {actor_path}\n")
    
    # Evaluation loop
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
    
    state = env.reset()
    ue = env.ue
    
    print("Running evaluation...")
    for t in range(scenario.total_timesteps):
        if (t + 1) % 200 == 0:
            print(f"  Progress: {t+1}/{scenario.total_timesteps} ({100*(t+1)/scenario.total_timesteps:.1f}%)")
        
        action = agent.select_action(state)
        battery_before = ue.battery_j
        
        next_state, _, _, info = env.step(action)
        
        # Extract metrics
        latency = info.get("latency", 0.0)
        energy = info.get("energy", 0.0)
        battery = info.get("battery", 0.0)
        success = info.get("success", False)
        task_class = info.get("task_class", 0)
        deadline = info.get("deadline", latency if latency > 0 else 1.0)
        
        # Compute QoE
        if battery_before <= 0:
            qoe = EnvConfig.FAIL_PENALTY
        elif latency > 0:
            if success:
                qoe = -(energy / battery_before) if battery_before > 0 else EnvConfig.FAIL_PENALTY
            else:
                qoe = EnvConfig.FAIL_PENALTY
        else:
            qoe = 0.0
        
        offload = 1.0 if action in (1, 2) else 0.0
        
        metrics["QoE"].append(qoe)
        metrics["Latency"].append(latency)
        metrics["Energy"].append(energy)
        metrics["Battery"].append(battery)
        metrics["OffloadRatio"].append(offload)
        metrics["Action"].append(action)
        metrics["TaskClass"].append(task_class)
        metrics["Success"].append(int(success))
        
        state = next_state
        
        if battery <= 0:
            print(f"\nWARNING: Battery depleted at timestep {t+1}")
            break
    
    # Save results
    df = pd.DataFrame(metrics)
    
    output_scenario_dir = os.path.join(output_dir, scenario_key)
    os.makedirs(output_scenario_dir, exist_ok=True)
    
    csv_path = os.path.join(output_scenario_dir, "rl_agent_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Saved evaluation results to {csv_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Mean QoE: {df['QoE'].mean():.6f}")
    print(f"Mean Latency: {df['Latency'].mean():.6f} s")
    print(f"Mean Energy: {df['Energy'].mean():.3f} J")
    print(f"Final Battery: {df['Battery'].iloc[-1]:.2f} J")
    print(f"Success Rate: {df['Success'].mean():.1%}")
    print(f"Offload Ratio: {df['OffloadRatio'].mean():.3f}")
    print(f"{'='*80}\n")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run scenario-based simulations from the paper",
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
        help="Scenario key (e.g., 's1_class1_90', 's1_class2_90', etc.)",
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train RL agent on the scenario",
    )
    
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate trained RL agent on the scenario",
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes (default: 500)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use for training/evaluation",
    )
    
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full pipeline: baselines + train RL + eval RL",
    )
    
    args = parser.parse_args()
    
    # List scenarios
    if args.list:
        list_scenarios()
        return
    
    # Require scenario for other operations
    if not args.scenario and not args.list:
        parser.print_help()
        print("\nERROR: --scenario is required (or use --list to see options)")
        sys.exit(1)
    
    # Validate scenario
    if args.scenario not in ALL_SCENARIOS:
        print(f"ERROR: Unknown scenario '{args.scenario}'")
        print(f"\nAvailable scenarios: {list(ALL_SCENARIOS.keys())}")
        print("\nUse --list to see full details")
        sys.exit(1)
    
    # Full pipeline: run baselines + train RL + evaluate RL.
    if args.full_pipeline:
        print("\n" + "="*80)
        print("RUNNING FULL PIPELINE")
        print("="*80)

        # 1) Baselines (Always-Local, Always-MEC, Always-Cloud, Random, Greedy-by-Size)
        run_all_baselines(
            args.scenario,
            timesteps=None,
            output_dir="results/scenarios",
        )

        # 2) Train RL
        train_rl_on_scenario(
            args.scenario,
            episodes=args.episodes,
            device=args.device,
        )

        # 3) Evaluate RL
        evaluate_rl_on_scenario(
            args.scenario,
            device=args.device,
        )
        return
    
    # Train only
    if args.train:
        train_rl_on_scenario(
            args.scenario,
            episodes=args.episodes,
            device=args.device,
        )
    
    # Evaluate only
    if args.eval:
        evaluate_rl_on_scenario(
            args.scenario,
            device=args.device,
        )


if __name__ == "__main__":
    main()
