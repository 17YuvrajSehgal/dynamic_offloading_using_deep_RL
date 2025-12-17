#!/usr/bin/env python3
"""
debug_ac_learning.py

Deep debugging script to analyze why the AC agent isn't learning properly.
Tracks detailed metrics and compares with baselines.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from offload_rl.EnvConfig import EnvConfig
from offload_rl.ac_agent import ActorCriticAgent
from offload_rl.models import UE, BaseStation, MECServer, CloudServer
from offload_rl.rl_env import OffloadEnv


def make_env() -> OffloadEnv:
    ue = UE(n=0, x_m=50.0, y_m=50.0)
    bs = BaseStation()
    mec = MECServer()
    cloud = CloudServer(fiber_distance_m=EnvConfig.CLOUD_FIBER_DISTANCE)
    env = OffloadEnv(ue, bs, mec, cloud, max_steps=EnvConfig.TOTAL_TIME_T)
    return env


def analyze_episode_detailed(agent, env, episode_num):
    """Run one episode and collect detailed statistics."""
    state = env.reset()
    done = False
    
    stats = {
        'actions': [],
        'action_probs': [],  # probability distribution over actions
        'rewards': [],
        'battery_levels': [],
        'energies': [],
        'latencies': [],
        'task_classes': [],
        'successes': [],
        'td_errors': [],
        'critic_values': [],
    }
    
    step = 0
    while not done and step < 2000:
        # Get action probabilities (for debugging)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            probs = agent.actor(state_tensor).cpu().numpy()[0]
            value = agent.critic(state_tensor).cpu().item()
        
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        # Compute TD error for this step (before update)
        with torch.no_grad():
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            next_value = agent.critic(next_state_tensor).cpu().item()
            td_error = reward + agent.gamma * next_value * (1.0 - float(done)) - value
        
        # Update agent
        agent.update(state, action, reward, next_state, done)
        
        # Record stats
        stats['actions'].append(action)
        stats['action_probs'].append(probs)
        stats['rewards'].append(reward)
        stats['battery_levels'].append(env.ue.battery_j)
        stats['td_errors'].append(td_error)
        stats['critic_values'].append(value)
        
        if 'energy' in info:
            stats['energies'].append(info['energy'])
        if 'latency' in info:
            stats['latencies'].append(info['latency'])
        if 'task_class' in info:
            stats['task_classes'].append(info.get('task_class', 0))
        if 'success' in info:
            stats['successes'].append(int(info['success']))
        
        state = next_state
        step += 1
    
    return stats


def run_baseline_comparison(env, policy_name='local'):
    """Run a baseline policy for comparison."""
    state = env.reset()
    done = False
    
    stats = {
        'actions': [],
        'rewards': [],
        'battery_levels': [],
        'energies': [],
        'successes': [],
    }
    
    step = 0
    while not done and step < 2000:
        if policy_name == 'local':
            action = 0
        elif policy_name == 'mec':
            action = 1
        elif policy_name == 'cloud':
            action = 2
        elif policy_name == 'random':
            action = np.random.choice([0, 1, 2])
        else:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        next_state, reward, done, info = env.step(action)
        
        stats['actions'].append(action)
        stats['rewards'].append(reward)
        stats['battery_levels'].append(env.ue.battery_j)
        if 'energy' in info:
            stats['energies'].append(info['energy'])
        if 'success' in info:
            stats['successes'].append(int(info['success']))
        
        state = next_state
        step += 1
    
    return stats


def print_diagnostic_report(agent_stats_list, baseline_stats):
    """Print comprehensive diagnostic report."""
    print("\n" + "="*80)
    print("DIAGNOSTIC REPORT")
    print("="*80)
    
    # 1. Battery depletion analysis
    print("\n1. BATTERY DEPLETION ANALYSIS")
    print("-" * 80)
    final_batteries = [s['battery_levels'][-1] if s['battery_levels'] else 0 for s in agent_stats_list]
    avg_final_battery = np.mean(final_batteries)
    print(f"   Average final battery (RL): {avg_final_battery:.2f} J")
    print(f"   Baseline final battery ({baseline_stats['policy']}): {baseline_stats['battery_levels'][-1]:.2f} J")
    
    if avg_final_battery < 100:
        print("   ⚠️  PROBLEM: RL agent depletes battery completely!")
        print("   → Agent is not learning to conserve energy")
    
    # 2. Action distribution analysis
    print("\n2. ACTION DISTRIBUTION ANALYSIS")
    print("-" * 80)
    all_actions = []
    for stats in agent_stats_list:
        all_actions.extend(stats['actions'])
    
    action_counts = {0: all_actions.count(0), 1: all_actions.count(1), 2: all_actions.count(2)}
    total_actions = len(all_actions)
    
    print(f"   Local (0):  {action_counts[0]:5d} ({100*action_counts[0]/total_actions:5.1f}%)")
    print(f"   MEC (1):    {action_counts[1]:5d} ({100*action_counts[1]/total_actions:5.1f}%)")
    print(f"   Cloud (2):  {action_counts[2]:5d} ({100*action_counts[2]/total_actions:5.1f}%)")
    
    offload_ratio = (action_counts[1] + action_counts[2]) / total_actions
    print(f"   Offload ratio: {offload_ratio:.2f}")
    
    if offload_ratio > 0.8:
        print("   ⚠️  PROBLEM: Agent offloads too much (>80%)")
        print("   → This causes rapid battery depletion")
    
    # 3. Reward analysis
    print("\n3. REWARD ANALYSIS")
    print("-" * 80)
    all_rewards = []
    for stats in agent_stats_list:
        all_rewards.extend(stats['rewards'])
    
    print(f"   Mean reward: {np.mean(all_rewards):.6f}")
    print(f"   Std reward:  {np.std(all_rewards):.6f}")
    print(f"   Min reward:  {np.min(all_rewards):.6f}")
    print(f"   Max reward:  {np.max(all_rewards):.6f}")
    
    # Check reward distribution
    negative_rewards = sum(1 for r in all_rewards if r < 0)
    print(f"   Negative rewards: {negative_rewards}/{len(all_rewards)} ({100*negative_rewards/len(all_rewards):.1f}%)")
    
    # 4. TD Error analysis
    print("\n4. TD ERROR ANALYSIS (Learning Signal)")
    print("-" * 80)
    all_td_errors = []
    for stats in agent_stats_list:
        if stats['td_errors']:
            all_td_errors.extend(stats['td_errors'])
    
    if all_td_errors:
        print(f"   Mean TD error: {np.mean(all_td_errors):.6f}")
        print(f"   Std TD error:  {np.std(all_td_errors):.6f}")
        print(f"   Abs mean TD error: {np.mean(np.abs(all_td_errors)):.6f}")
        
        if np.mean(np.abs(all_td_errors)) < 0.001:
            print("   ⚠️  PROBLEM: TD errors are very small")
            print("   → Learning signal is weak, agent may not be learning")
    
    # 5. Critic value analysis
    print("\n5. CRITIC VALUE ANALYSIS")
    print("-" * 80)
    all_values = []
    for stats in agent_stats_list:
        if stats['critic_values']:
            all_values.extend(stats['critic_values'])
    
    if all_values:
        print(f"   Mean value: {np.mean(all_values):.6f}")
        print(f"   Std value:  {np.std(all_values):.6f}")
        print(f"   Value range: [{np.min(all_values):.3f}, {np.max(all_values):.3f}]")
    
    # 6. Action probability analysis
    print("\n6. ACTION PROBABILITY ANALYSIS")
    print("-" * 80)
    if agent_stats_list and agent_stats_list[0]['action_probs']:
        # Get probabilities from first and last episode
        first_probs = np.array(agent_stats_list[0]['action_probs'])
        last_probs = np.array(agent_stats_list[-1]['action_probs'])
        
        print("   First episode average probabilities:")
        print(f"     Local: {np.mean(first_probs[:, 0]):.3f}")
        print(f"     MEC:   {np.mean(first_probs[:, 1]):.3f}")
        print(f"     Cloud: {np.mean(first_probs[:, 2]):.3f}")
        
        print("   Last episode average probabilities:")
        print(f"     Local: {np.mean(last_probs[:, 0]):.3f}")
        print(f"     MEC:   {np.mean(last_probs[:, 1]):.3f}")
        print(f"     Cloud: {np.mean(last_probs[:, 2]):.3f}")
        
        # Check if probabilities changed
        prob_change = np.abs(np.mean(last_probs, axis=0) - np.mean(first_probs, axis=0))
        if np.max(prob_change) < 0.05:
            print("   ⚠️  PROBLEM: Action probabilities barely changed")
            print("   → Policy is not learning/updating")
    
    # 7. Recommendations
    print("\n7. RECOMMENDATIONS")
    print("-" * 80)
    issues = []
    
    if avg_final_battery < 100:
        issues.append("Battery depletion")
    if offload_ratio > 0.8:
        issues.append("Excessive offloading")
    if all_td_errors and np.mean(np.abs(all_td_errors)) < 0.001:
        issues.append("Weak learning signal")
    
    if issues:
        print("   Detected issues:")
        for issue in issues:
            print(f"     - {issue}")
        print("\n   Suggested fixes:")
        print("     1. Increase entropy coefficient (currently 0.01) to encourage exploration")
        print("     2. Add battery penalty: reward -= alpha * (battery_depleted)")
        print("     3. Scale rewards: multiply by a constant to make signal stronger")
        print("     4. Use reward shaping: add small positive reward for battery conservation")
        print("     5. Increase learning rates (currently actor=1e-5, critic=1e-4)")
        print("     6. Use experience replay or n-step returns for more stable learning")
    else:
        print("   ✓ No major issues detected")
    
    print("\n" + "="*80)


def main():
    print("="*80)
    print("AC AGENT LEARNING DIAGNOSTICS")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Create environment and agent
    env = make_env()
    init_state = env.reset()
    state_dim = init_state.shape[0]
    
    agent = ActorCriticAgent(
        state_dim=state_dim,
        n_actions=3,
        gamma=0.99,
        lr_actor=1e-5,
        lr_critic=1e-4,
        device=device,
    )
    
    # Run a few episodes and collect stats
    print("Running RL agent for 5 episodes...")
    agent_stats_list = []
    for ep in range(5):
        stats = analyze_episode_detailed(agent, env, ep)
        agent_stats_list.append(stats)
        print(f"  Episode {ep+1}: final battery={stats['battery_levels'][-1]:.2f} J, "
              f"return={sum(stats['rewards']):.3f}, steps={len(stats['actions'])}")
    
    # Run baseline for comparison
    print("\nRunning baseline (local) for comparison...")
    baseline_stats = run_baseline_comparison(env, policy_name='local')
    baseline_stats['policy'] = 'local'
    print(f"  Baseline: final battery={baseline_stats['battery_levels'][-1]:.2f} J, "
          f"return={sum(baseline_stats['rewards']):.3f}, steps={len(baseline_stats['actions'])}")
    
    # Print diagnostic report
    print_diagnostic_report(agent_stats_list, baseline_stats)
    
    # Save detailed stats to CSV for further analysis
    os.makedirs("debug_output", exist_ok=True)
    
    # Save episode-by-episode summary
    summary_data = []
    for i, stats in enumerate(agent_stats_list):
        summary_data.append({
            'episode': i + 1,
            'final_battery': stats['battery_levels'][-1] if stats['battery_levels'] else 0,
            'total_return': sum(stats['rewards']),
            'num_steps': len(stats['actions']),
            'offload_ratio': sum(1 for a in stats['actions'] if a in [1, 2]) / max(1, len(stats['actions'])),
            'success_rate': np.mean(stats['successes']) if stats['successes'] else 0,
            'mean_reward': np.mean(stats['rewards']) if stats['rewards'] else 0,
            'mean_td_error': np.mean(np.abs(stats['td_errors'])) if stats['td_errors'] else 0,
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv("debug_output/episode_summary.csv", index=False)
    print("\n✅ Saved episode summary to debug_output/episode_summary.csv")
    
    # Save detailed stats from last episode
    if agent_stats_list:
        last_ep = agent_stats_list[-1]
        detailed_df = pd.DataFrame({
            'step': range(len(last_ep['actions'])),
            'action': last_ep['actions'],
            'reward': last_ep['rewards'],
            'battery': last_ep['battery_levels'],
            'td_error': last_ep['td_errors'],
            'critic_value': last_ep['critic_values'],
        })
        detailed_df.to_csv("debug_output/last_episode_detailed.csv", index=False)
        print("✅ Saved last episode details to debug_output/last_episode_detailed.csv")


if __name__ == "__main__":
    main()

