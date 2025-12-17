"""train_rl_improved.py

Trains the improved Actor-Critic agent with better learning dynamics.
"""

import os
import sys
import numpy as np
import torch

from offload_rl.EnvConfig import EnvConfig
from offload_rl.ac_agent_improved import ImprovedActorCriticAgent
from offload_rl.models import UE, BaseStation, MECServer, CloudServer
from offload_rl.rl_env import OffloadEnv


def make_env() -> OffloadEnv:
    ue = UE(n=0, x_m=50.0, y_m=50.0)
    bs = BaseStation()
    mec = MECServer()
    cloud = CloudServer(fiber_distance_m=EnvConfig.CLOUD_FIBER_DISTANCE)
    env = OffloadEnv(ue, bs, mec, cloud, max_steps=EnvConfig.TOTAL_TIME_T)
    return env


def train(
    episodes: int = 100,
    gamma: float = 0.99,
    lr_actor: float = 3e-5,
    lr_critic: float = 3e-4,
    device: str = None,
    log_every: int = 10,
    reward_scale: float = 10.0,
    entropy_coeff: float = 0.05,
    battery_penalty: float = 0.1,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_rl_improved] Selected device: {device}")
    
    env = make_env()
    init_state = env.reset()
    state_dim = init_state.shape[0]
    print(f"[train_rl_improved] State dimension: {state_dim}")
    print(f"[train_rl_improved] Action space: 3 (local, MEC, cloud)")
    print(f"[train_rl_improved] Reward scale: {reward_scale}")
    print(f"[train_rl_improved] Entropy coeff: {entropy_coeff}")
    print(f"[train_rl_improved] Battery penalty: {battery_penalty}")
    print()

    agent = ImprovedActorCriticAgent(
        state_dim=state_dim,
        n_actions=3,
        gamma=gamma,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        device=device,
        reward_scale=reward_scale,
        entropy_coeff=entropy_coeff,
        battery_penalty=battery_penalty,
    )

    all_returns = []
    all_steps = []
    all_final_batteries = []

    print("Starting training with improved agent...\n")
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        ep_offload_actions = 0
        ep_local_actions = 0
        ep_successes = 0
        ep_tasks = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Pass battery level for penalty
            battery_level = env.ue.battery_j
            agent.update(state, action, reward, next_state, done, battery_level=battery_level)
            
            state = next_state
            ep_return += reward  # Note: this is original reward, not scaled
            steps += 1

            if action in (1, 2):
                ep_offload_actions += 1
            else:
                ep_local_actions += 1

            if "success" in info:
                ep_tasks += 1
                ep_successes += int(info["success"])

            if steps > 100000:
                print(f"WARNING: Episode {ep+1} exceeded 100k steps. Breaking.")
                break

        all_returns.append(ep_return)
        all_steps.append(steps)
        final_battery = info.get('battery', 0)
        all_final_batteries.append(final_battery)

        total_actions = ep_offload_actions + ep_local_actions
        offload_ratio = ep_offload_actions / max(1, total_actions)
        success_rate = ep_successes / max(1, ep_tasks)

        if (ep + 1) % log_every == 0 or ep == 0:
            avg_return_last = np.mean(all_returns[-log_every:])
            avg_steps_last = np.mean(all_steps[-log_every:])
            avg_battery_last = np.mean(all_final_batteries[-log_every:])
            print(
                f"Episode {ep + 1:03d} | "
                f"steps={steps:5d} | "
                f"return={ep_return:9.3f} | "
                f"battery={final_battery:7.2f} J | "
                f"offload_ratio={offload_ratio:5.2f} | "
                f"success_rate={success_rate:5.2f} | "
                f"avg_return({log_every})={avg_return_last:9.3f} | "
                f"avg_battery({log_every})={avg_battery_last:7.2f} J"
            )
            sys.stdout.flush()

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total episodes: {episodes}")
    print(f"Final episode return: {all_returns[-1]:.3f}")
    print(f"Avg return (last 10): {np.mean(all_returns[-10:]):.3f}")
    print(f"Avg return (all): {np.mean(all_returns):.3f}")
    print(f"Best episode return: {max(all_returns):.3f}")
    print(f"Avg final battery (last 10): {np.mean(all_final_batteries[-10:]):.2f} J")
    print(f"Avg final battery (all): {np.mean(all_final_batteries):.2f} J")
    print("="*60 + "\n")

    os.makedirs("results", exist_ok=True)
    torch.save(agent.actor.state_dict(), "results/actor_improved.pt")
    torch.save(agent.critic.state_dict(), "results/critic_improved.pt")
    print("✅ Saved improved actor to results/actor_improved.pt")
    print("✅ Saved improved critic to results/critic_improved.pt")

    return all_returns


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--log-every', type=int, default=10, help='Episode interval for logging')
    parser.add_argument('--reward-scale', type=float, default=10.0, help='Reward scaling factor')
    parser.add_argument('--entropy-coeff', type=float, default=0.05, help='Entropy coefficient')
    parser.add_argument('--battery-penalty', type=float, default=0.1, help='Battery depletion penalty')
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        device=args.device,
        log_every=args.log_every,
        reward_scale=args.reward_scale,
        entropy_coeff=args.entropy_coeff,
        battery_penalty=args.battery_penalty,
    )

