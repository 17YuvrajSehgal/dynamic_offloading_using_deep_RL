"""train_rl.py

Trains an Actor–Critic agent for dynamic offloading on top
of the existing MEC simulation models.

Usage
-----
    python train_rl.py

This will run a small training loop and print episode rewards.
"""

import numpy as np
import torch

from EnvConfig import EnvConfig
from models import UE, BaseStation, MECServer, CloudServer
from rl_env import OffloadEnv
from ac_agent import ActorCriticAgent


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
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    init_state = env.reset()
    state_dim = init_state.shape[0]

    agent = ActorCriticAgent(
        state_dim=state_dim,
        n_actions=3,
        gamma=gamma,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        device=device,
    )

    all_returns = []

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

        all_returns.append(ep_return)
        print(
            f"Episode {ep+1:03d} | steps={steps:4d} | "
            f"return={ep_return:8.3f} | last_batt={info.get('battery', 0):7.2f}"
        )

    # Simple summary
    print("\nTraining finished.")
    print("Avg return over last 10 episodes:", np.mean(all_returns[-10:]))
    return all_returns


if __name__ == "__main__":
    train()
