"""train_rl.py

Trains an Actor–Critic agent for dynamic offloading on top
of the existing MEC simulation models.

Usage
-----
    python train_rl.py

This will run a small training loop and print episode rewards.
"""
import os
import sys

import numpy as np
import torch

from offload_rl.EnvConfig import EnvConfig
from offload_rl.ac_agent import ActorCriticAgent
from offload_rl.models import UE, BaseStation, MECServer, CloudServer
from offload_rl.rl_env import OffloadEnv


def print_gpu_info():
    """Print detailed GPU information for cluster debugging."""
    print("\n" + "="*60)
    print("GPU DIAGNOSTICS")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
            print(f"  Max memory allocated: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")
        
        print(f"\nCurrent CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    else:
        print("WARNING: CUDA is not available. Training will use CPU.")
        print("\nPossible reasons:")
        print("  1. PyTorch installed without CUDA support")
        print("  2. CUDA drivers not installed")
        print("  3. No GPU available on this node")
        print("  4. Environment variables not set correctly")
    
    print("="*60 + "\n")


def make_env() -> OffloadEnv:
    # For RL we use a single UE at some nominal position
    ue = UE(n=0, x_m=50.0, y_m=50.0)
    bs = BaseStation()
    mec = MECServer()
    cloud = CloudServer(fiber_distance_m=EnvConfig.CLOUD_FIBER_DISTANCE)
    env = OffloadEnv(ue, bs, mec, cloud, max_steps=EnvConfig.TOTAL_TIME_T)
    return env


def train(
    episodes: int = 50,
    gamma: float = 0.99,
    lr_actor: float = 1e-5,
    lr_critic: float = 1e-4,
    device: str = None,
    log_every: int = 10,
):
    # Print GPU diagnostics
    print_gpu_info()
    
    # Auto-select device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[train_rl] Selected device: {device}")

    if device == "cuda":
        print("GPU memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
        print("GPU memory reserved:", torch.cuda.memory_reserved() / 1e9, "GB")

    # Verify device is actually available
    if device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA device requested but not available. Falling back to CPU.")
        device = "cpu"
    
    # Create environment
    env = make_env()
    init_state = env.reset()
    state_dim = init_state.shape[0]
    print(f"[train_rl] State dimension: {state_dim}")
    print(f"[train_rl] Action space: 3 (local, MEC, cloud)")
    print(f"[train_rl] Max steps per episode: {EnvConfig.TOTAL_TIME_T}")
    print(f"[train_rl] Task arrival rate (lambda): {EnvConfig.TASK_ARRIVAL_RATE}")

    # Create agent
    agent = ActorCriticAgent(
        state_dim=state_dim,
        n_actions=3,
        gamma=gamma,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        device=device,
    )

    if device == "cuda":
        print("[DEBUG] After agent creation:")
        print("  GPU memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
        print("  GPU memory reserved:", torch.cuda.memory_reserved() / 1e9, "GB")
    
    # Verify models are on correct device
    print(f"[train_rl] Actor device: {next(agent.actor.parameters()).device}")
    print(f"[train_rl] Critic device: {next(agent.critic.parameters()).device}")
    
    # Count parameters
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    print(f"[train_rl] Actor parameters: {actor_params:,}")
    print(f"[train_rl] Critic parameters: {critic_params:,}")
    print(f"[train_rl] Total parameters: {actor_params + critic_params:,}")
    print()

    all_returns = []
    all_steps = []

    print("Starting training...\n")
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        # Debug counters
        ep_offload_actions = 0  # how many times we chose MEC or Cloud
        ep_local_actions = 0    # how many times we chose Local
        ep_successes = 0
        ep_tasks = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_return += reward
            steps += 1

            # ---------------- Debug bookkeeping ----------------
            if action in (1, 2):
                ep_offload_actions += 1
            else:
                ep_local_actions += 1

            if "success" in info:
                ep_tasks += 1
                ep_successes += int(info["success"])

            if device == "cuda" and steps == 5:
                print("[DEBUG] After 5 steps:")
                print("  GPU memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
                print("  GPU memory reserved:", torch.cuda.memory_reserved() / 1e9, "GB")
            
            # Safety check to prevent infinite loops
            if steps > 100000:
                print(f"WARNING: Episode {ep+1} exceeded 100k steps. Breaking.")
                break

        all_returns.append(ep_return)
        all_steps.append(steps)

        # Episode-level debug stats
        total_actions = ep_offload_actions + ep_local_actions
        offload_ratio = ep_offload_actions / max(1, total_actions)
        success_rate = ep_successes / max(1, ep_tasks)

        if (ep + 1) % log_every == 0 or ep == 0:
            avg_return_last = np.mean(all_returns[-log_every:])
            avg_steps_last = np.mean(all_steps[-log_every:])
            print(
                f"Episode {ep + 1:03d} | "
                f"steps={steps:5d} | "
                f"return={ep_return:9.3f} | "
                f"battery={info.get('battery', 0):7.2f} J | "
                f"offload_ratio={offload_ratio:5.2f} | "
                f"success_rate={success_rate:5.2f} | "
                f"avg_return({log_every})={avg_return_last:9.3f} | "
                f"avg_steps({log_every})={avg_steps_last:6.1f}"
            )
            sys.stdout.flush()  # Force flush for cluster logs

    # Training summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total episodes: {episodes}")
    print(f"Final episode return: {all_returns[-1]:.3f}")
    print(f"Avg return (last 10 episodes): {np.mean(all_returns[-10:]):.3f}")
    print(f"Avg return (all episodes): {np.mean(all_returns):.3f}")
    print(f"Best episode return: {max(all_returns):.3f}")
    print(f"Avg steps per episode: {np.mean(all_steps):.1f}")
    print("="*60 + "\n")

    # Save models
    os.makedirs("results", exist_ok=True)
    torch.save(agent.actor.state_dict(), "results/actor_offloading.pt")
    torch.save(agent.critic.state_dict(), "results/critic_offloading.pt")
    print("✅ Saved actor weights to results/actor_offloading.pt")
    print("✅ Saved critic weights to results/critic_offloading.pt")
    
    # Print final GPU memory usage if using CUDA
    if device == "cuda":
        print(f"\nFinal GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return all_returns


if __name__ == "__main__":
    # You can override episodes from command line if needed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--log-every', type=int, default=10, help='Episode interval for detailed debug logging')
    args = parser.parse_args()
    
    train(episodes=args.episodes, device=args.device, log_every=args.log_every)
