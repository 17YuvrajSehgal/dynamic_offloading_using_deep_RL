import matplotlib.pyplot as plt
from sim import Simulator
from policy import AlwaysLocal, GreedyBySize

def smooth(vals, k=10):
    if len(vals) < k: return vals
    import numpy as np
    kern = np.ones(k) / k
    return list(np.convolve(vals, kern, mode='valid'))

def run_and_plot(name, policy):
    sim = Simulator(n_ues=20, lam=1.0)
    met = sim.run(T=1000, policy=policy)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs[0,0].plot(smooth(met.qoe));      axs[0,0].set_title(f"{name} — QoE")
    axs[0,1].plot(met.battery);          axs[0,1].set_title("Avg Battery (J)")
    axs[1,0].plot(smooth(met.latency));  axs[1,0].set_title("Latency (s)")
    axs[1,1].plot(smooth(met.offload_ratio)); axs[1,1].set_title("Offload ratio")
    for ax in axs.ravel(): ax.grid(True, alpha=0.3)
    fig.suptitle(f"Baseline: {name}")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    run_and_plot("Always-Local", AlwaysLocal())
    run_and_plot("Greedy-By-Size", GreedyBySize(size_threshold_bits=150e3*8))
