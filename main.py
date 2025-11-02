import matplotlib.pyplot as plt
import numpy as np
from sim import Simulator
from policy import AlwaysLocal, GreedyBySize
from EnvConfig import EnvConfig


def smooth(vals, k=10):
    """Utility for smoothing noisy curves."""
    if len(vals) < k:
        return vals
    kern = np.ones(k) / k
    return np.convolve(vals, kern, mode='valid')


def run_and_plot(name, policy, lam=1.0, T=1000):
    """Run simulation with a given policy and plot results."""
    sim = Simulator(n_ues=EnvConfig.NUM_UES, lam=lam)
    metrics = sim.run(T=T, policy=policy)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    axs[0, 0].plot(smooth(metrics.qoe))
    axs[0, 0].set_title(f"{name} — QoE")

    axs[0, 1].plot(metrics.battery)
    axs[0, 1].set_title("Avg Battery (J)")

    axs[1, 0].plot(smooth(metrics.latency))
    axs[1, 0].set_title("Latency (s)")

    axs[1, 1].plot(smooth(metrics.offload_ratio))
    axs[1, 1].set_title("Offload ratio")

    for ax in axs.ravel():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Timestep")

    fig.suptitle(f"Baseline: {name}", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run both baselines
    run_and_plot("Always-Local", AlwaysLocal(), lam=1.0, T=500)
    run_and_plot("Greedy-By-Size", GreedyBySize(size_threshold_bits=150e3 * 8), lam=1.0, T=500)
