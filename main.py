import matplotlib.pyplot as plt
import numpy as np
from sim import Simulator
from policy import AlwaysLocal, AlwaysMEC, AlwaysCloud, GreedyBySize
from EnvConfig import EnvConfig
import pandas as pd
import os


def smooth(vals, k=10):
    """Utility for smoothing noisy curves."""
    if len(vals) < k:
        return vals
    kern = np.ones(k) / k
    return np.convolve(vals, kern, mode='valid')


def run_and_plot(name, policy, lam=1.0, T=1000, save_csv=True):
    """Run simulation with a given policy, plot results, and optionally save metrics."""
    print(f"\n▶ Running baseline: {name}")
    sim = Simulator(n_ues=EnvConfig.NUM_UES, lam=lam)
    plot_local_layout(sim)
    plot_global_layout(sim)
    metrics = sim.run(T=T, policy=policy)

    # --- Plot results ---
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs[0, 0].plot(smooth(metrics.qoe)); axs[0, 0].set_title(f"{name} — QoE")
    axs[0, 1].plot(metrics.battery); axs[0, 1].set_title("Avg Battery (J)")
    axs[1, 0].plot(smooth(metrics.latency)); axs[1, 0].set_title("Latency (s)")
    axs[1, 1].plot(smooth(metrics.offload_ratio)); axs[1, 1].set_title("Offload Ratio")

    for ax in axs.ravel():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Timestep")

    fig.suptitle(f"Baseline: {name}", fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- Save results to CSV for later comparison ---
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

def summarize_results(result_dir="results"):
    """Aggregate all baseline CSVs and produce a summary table + comparison plot."""
    all_stats = []
    csv_files = [f for f in os.listdir(result_dir) if f.endswith("_metrics.csv")]

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

    # --- Plot aggregated comparison ---
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    x = summary_df["Baseline"]

    axs[0].bar(x, summary_df["Mean_QoE"], color="cornflowerblue")
    axs[0].set_title("Average QoE")
    axs[0].set_ylabel("QoE")

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

def plot_local_layout(sim):
    """Plot UEs and MEC servers (local zoomed view)."""
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
    """Plot UEs, MEC, and Cloud servers (wide view)."""
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


if __name__ == "__main__":
    # === Baseline Experiments (comparable to research paper) ===
    baselines = [
        ("Always-Local", AlwaysLocal()),
        ("Always-MEC", AlwaysMEC()),
        ("Always-Cloud", AlwaysCloud()),
        ("Greedy-By-Size", GreedyBySize(size_threshold_bits=150e3 * 8)),
    ]

    for name, policy in baselines:
        run_and_plot(name, policy, lam=1.0, T=500)

    summarize_results()

