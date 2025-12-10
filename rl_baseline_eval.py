# rl_baseline_eval.py

import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from EnvConfig import EnvConfig
from ac_agent import ActorCriticAgent
from train_rl import make_env  # reuse your existing helper

def smooth(vals, k: int = 10):
    """Moving-average smoothing for nicer plots."""
    vals = np.asarray(vals)
    if len(vals) < k:
        return vals
    kernel = np.ones(k) / k
    return np.convolve(vals, kernel, mode="valid")

def evaluate_rl_baseline(
    actor_path: str = "results/actor_offloading.pt",
    csv_name: str = "rl-agent_metrics.csv",
    T: int = EnvConfig.TOTAL_TIME_T,
):
    """
    Evaluate the trained RL actor as another 'baseline' and save metrics
    in the same CSV format as the other *_metrics.csv files.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rl_baseline_eval] Using device: {device}")

    # Build environment and agent skeleton
    env = make_env()
    init_state = env.reset()
    state_dim = init_state.shape[0]

    agent = ActorCriticAgent(
        state_dim=state_dim,
        n_actions=3,
        device=device,
    )
    # Load trained actor weights
    state_dict = torch.load(actor_path, map_location=device)
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()
    print(f"[rl_baseline_eval] Loaded actor weights from {actor_path}")

    # Metric buffers
    qoe_vals = []
    lat_vals = []
    eng_vals = []
    batt_vals = []
    offload_vals = []

    state = env.reset()
    ue = env.ue  # same UE object used internally

    for t in range(T):
        action = agent.select_action(state)  # 0=local, 1=mec, 2=cloud
        
        # Store battery BEFORE taking action (for QoE calculation)
        battery_before = ue.battery_j
        
        next_state, _, _, info = env.step(action)

        latency = float(info.get("latency", 0.0))
        energy = float(info.get("energy", 0.0))
        battery = float(info.get("battery", 0.0))
        success = bool(info.get("success", False))
        deadline = float(info.get("deadline", latency))

        # ---- QoE computation: SAME as Paper Equation 18 ----
        # Successful tasks: QoE = -E_consumed / B_n (current battery)
        # Failed tasks: QoE = η (FAIL_PENALTY)
        if battery_before <= 0.0:
            # dead UE: penalty and large "virtual" latency
            qoe = EnvConfig.FAIL_PENALTY
            latency_eff = deadline * 10.0
        else:
            if success:
                # Use battery BEFORE consumption as B_n
                qoe = -(energy / battery_before)
                latency_eff = latency
            else:
                qoe = EnvConfig.FAIL_PENALTY
                latency_eff = latency

        # Offload ratio: 1 if MEC/Cloud, 0 if local
        offload = 1.0 if action in (1, 2) else 0.0

        qoe_vals.append(qoe)
        lat_vals.append(latency_eff)
        eng_vals.append(energy)
        batt_vals.append(battery)
        offload_vals.append(offload)

        state = next_state

    # ---- Save to CSV in same format as baselines ----
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(
        {
            "QoE": qoe_vals,
            "Latency": lat_vals,
            "Energy": eng_vals,
            "Battery": batt_vals,
            "OffloadRatio": offload_vals,
        }
    )

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs[0, 0].plot(smooth(df["QoE"]))
    axs[0, 0].set_title("RL Agent — QoE")

    axs[0, 1].plot(df["Battery"])
    axs[0, 1].set_title("Avg Battery (J)")

    axs[1, 0].plot(smooth(df["Latency"]))
    axs[1, 0].set_title("Latency (s)")

    axs[1, 1].plot(smooth(df["OffloadRatio"]))
    axs[1, 1].set_title("Offload Ratio")

    for ax in axs.ravel():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Timestep")

    fig.suptitle("RL Agent Performance", fontsize=12)
    plt.tight_layout()
    plt.show()

    csv_path = os.path.join("results", csv_name)
    df.to_csv(csv_path, index=False)
    print(f"[rl_baseline_eval] Saved RL metrics to {csv_path}")

    # Print quick summary
    print(
        "Mean QoE:", np.mean(qoe_vals),
        "| Mean Latency:", np.mean(lat_vals),
        "| Final Battery:", batt_vals[-1],
        "| Mean Offload Ratio:", np.mean(offload_vals),
    )

def run_all_with_rl(train_first: bool = False):
    """
    Full pipeline:
      1) (optional) train RL actor
      2) evaluate RL and save rl-agent_metrics.csv + RL-only plot
      3) run all classical baselines and combined comparison plot
    """
    if train_first:
        from train_rl import train
        print("[run_all_with_rl] Training RL agent...")
        train()  # you can pass episodes=... here if you want

    print("[run_all_with_rl] Evaluating RL agent...")
    evaluate_rl_baseline()

    print("[run_all_with_rl] Running baselines and combined plots...")
    from main import run_all_baselines_and_plots
    run_all_baselines_and_plots()

if __name__ == "__main__":
    # By default: assume you've already trained RL (train_rl.py),
    # then run RL eval + baselines + combined comparison.
    run_all_with_rl(train_first=True)
