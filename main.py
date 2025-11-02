"""
main.py
--------
Runs baseline experiments for the 3-layer MEC simulation system.
Includes:
 - Local and global network layout plots (UEs, MEC, Cloud)
 - Execution of all baseline policies
 - Automatic metric collection and CSV export
 - Combined summary table and comparison plots

Author: [Your Name]
Course: COSC 5P06 – Mobile Edge Computing Simulation
"""

# ===============================================================
# === Imports ===================================================
# ===============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sim import Simulator
from policy import AlwaysLocal, AlwaysMEC, AlwaysCloud, GreedyBySize
from EnvConfig import EnvConfig


# ===============================================================
# === Utility Functions =========================================
# ===============================================================

def smooth(vals, k: int = 10):
    """Applies a simple moving average to smooth noisy metric curves."""
    if len(vals) < k:
        return vals
    kernel = np.ones(k) / k
    return np.convolve(vals, kernel, mode='valid')


# ===============================================================
# === Visualization: Network Layouts =============================
# ===============================================================

def plot_local_layout(sim: Simulator):
    """Plot the distribution of UEs and the MEC server (zoomed-in local view)."""
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


def plot_global_layout(sim: Simulator):
    """Plot the overall network including UEs, MEC, and Cloud servers."""
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

def run_and_plot(name: str, policy, lam: float = 1.0, T: int = 1000, save_csv: bool = True):
    """
    Run a single baseline experiment for the given policy.
    Produces metric plots and optionally saves CSV results.
    """
    print(f"\n▶ Running baseline: {name}")

    # --- Initialize simulation environment ---
    sim = Simulator(n_ues=EnvConfig.NUM_UES, lam=lam)

    # --- Plot system layout once per run ---
    plot_local_layout(sim)
    plot_global_layout(sim)

    # --- Run simulation ---
    metrics = sim.run(T=T, policy=policy)

    # --- Plot metrics ---
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

    # --- Save metrics to CSV ---
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

def summarize_results(result_dir: str = "results"):
    """
    Aggregate all baseline results, compute averages,
    and produce a summary table and bar-chart comparison.
    """
    print("\n📊 Generating combined baseline summary...")

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
    summary_path = os.path.join(result_dir, "baseline_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Combined summary saved to {summary_path}")

    # --- Plot comparison chart ---
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
    Runs all baseline experiments and produces:
      - Local/global layout plots
      - Per-baseline metric graphs and CSVs
      - Combined summary table and comparison figure
    """

    baselines = [
        ("Always-Local", AlwaysLocal()),
        ("Always-MEC", AlwaysMEC()),
        ("Always-Cloud", AlwaysCloud()),
        ("Greedy-By-Size", GreedyBySize(size_threshold_bits=150e3 * 8)),
    ]

    for name, policy in baselines:
        run_and_plot(name, policy, lam=1.0, T=500)

    summarize_results()
