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
