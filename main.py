"""
main.py
--------
Runs all baseline experiments for the 3-layer MEC simulation system
using a shared simulation environment for consistent comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sim import Simulator
from policy import AlwaysLocal, AlwaysMEC, AlwaysCloud, GreedyBySize
from EnvConfig import EnvConfig


# ===============================================================
# === Utility ===================================================
# ===============================================================
def smooth(vals, k: int = 10):
    """Apply a moving average filter to smooth noisy data."""
    if len(vals) < k:
        return vals
    kernel = np.ones(k) / k
    return np.convolve(vals, kernel, mode="valid")


# ===============================================================
# === Visualization: Network Layouts =============================
# ===============================================================
def plot_local_layout(sim):
    """Plot the distribution of UEs and MEC server (local zoomed view)."""
    plt.figure(figsize=(6, 5))
    xs = [ue.x_m for ue in sim.ues]
    ys = [ue.y_m for ue in sim.ues]
    plt.scatter(xs, ys, color="red", label="UEs")
    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.text(x + 1, y + 1, f"{i}", fontsize=8, color="darkred")
    plt.scatter([0], [0], color="green", label="MEC", s=80)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("UE and MEC Distribution (Local View)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_global_layout(sim):
    """Plot the distribution of UEs, MEC, and Cloud (global overview)."""
    plt.figure(figsize=(7, 5))
    xs = [ue.x_m for ue in sim.ues]
    ys = [ue.y_m for ue in sim.ues]
    plt.scatter(xs, ys, color="red", label="UEs")
    plt.scatter([0], [0], color="green", label="MEC", s=80)
    plt.scatter([1e6], [1e6], color="blue", label="Cloud", s=80)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("UE, MEC, and Cloud Server Distribution (Global View)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ===============================================================
# === Simulation Runner =========================================
# ===============================================================
def run_baseline(name, sim, policy, T=1000, save_csv=True):
    """Run a single baseline on a shared simulator environment."""
    print(f"\n▶ Running baseline: {name}")

    # Reset UE batteries and state before each run (so conditions identical)
    for ue in sim.ues:
        ue.battery_j = 4000.0

    # Run the simulation
    metrics = sim.run(T=T, policy=policy)

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs[0, 0].plot(smooth(metrics.qoe));           axs[0, 0].set_title(f"{name} — QoE")
    axs[0, 1].plot(metrics.battery);               axs[0, 1].set_title("Avg Battery (J)")
    axs[1, 0].plot(smooth(metrics.latency));       axs[1, 0].set_title("Latency (s)")
    axs[1, 1].plot(smooth(metrics.offload_ratio)); axs[1, 1].set_title("Offload Ratio")

    for ax in axs.ravel():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Timestep")

    fig.suptitle(f"Baseline: {name}", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Save CSV
    if save_csv:
        os.makedirs("results", exist_ok=True)
        df = pd.DataFrame({
            "QoE": metrics.qoe,
            "Latency": metrics.latency,
            "Energy": metrics.energy,
            "Battery": metrics.battery,
            "OffloadRatio": metrics.offload_ratio,
        })
        csv_path = f"results/{name.replace(' ', '_').lower()}_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Results saved to {csv_path}")

    return metrics


# ===============================================================
# === Summary Comparison ========================================
# ===============================================================
def summarize_results(result_dir="results"):
    """Aggregate results for all baselines and plot comparison."""
    csv_files = [f for f in os.listdir(result_dir) if f.endswith("_metrics.csv")]
    all_stats = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(result_dir, file))
        name = file.replace("_metrics.csv", "").replace("_", " ").title()
        stats = {
            "Baseline": name,
            "Mean_QoE": df["QoE"].mean(),
            "Mean_Latency": df["Latency"].mean(),
            "Mean_Energy": df["Energy"].mean(),
            "Final_Battery": df["Battery"].iloc[-1],
            "Mean_Offload_Ratio": df["OffloadRatio"].mean(),
        }
        all_stats.append(stats)
    summary_df = pd.DataFrame(all_stats)
    summary_df.to_csv(os.path.join(result_dir, "baseline_summary.csv"), index=False)
    print("✅ Combined summary saved to results/baseline_summary.csv")

    # Plot bar charts
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    x = summary_df["Baseline"]

    axs[0].bar(x, summary_df["Mean_QoE"], color="cornflowerblue")
    axs[0].set_title("Average QoE"); axs[0].set_ylabel("QoE")

    axs[1].bar(x, summary_df["Mean_Latency"], color="salmon")
    axs[1].set_title("Average Latency (s)")

    axs[2].bar(x, summary_df["Final_Battery"], color="seagreen")
    axs[2].set_title("Final Avg Battery (J)")

    for ax in axs:
        ax.set_xticklabels(x, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Baseline Strategy Comparison", fontsize=13)
    plt.tight_layout()
    plt.show()


# ===============================================================
# === Main Entry Point ==========================================
# ===============================================================
if __name__ == "__main__":
    """
    Creates one shared environment and runs all baselines under identical
    network conditions (same UE, MEC, Cloud positions).
    """

    # Create shared simulation environment
    sim = Simulator(n_ues=EnvConfig.NUM_UES, lam=1.0)
    print("✅ Shared simulation environment initialized.")

    # Visualize network layout (once)
    plot_local_layout(sim)
    plot_global_layout(sim)

    # Define baselines
    baselines = [
        ("Always-Local", AlwaysLocal()),
        ("Always-MEC", AlwaysMEC()),
        ("Always-Cloud", AlwaysCloud()),
        ("Greedy-By-Size", GreedyBySize(size_threshold_bits=150e3 * 8)),
    ]

    # Run each baseline on same environment
    for name, policy in baselines:
        run_baseline(name, sim, policy, T=500)

    # Generate comparison summary
    summarize_results()
