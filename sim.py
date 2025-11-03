from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from models import UE, BaseStation, MECServer, CloudServer, TaskFactory
from policy import Policy, AlwaysLocal, GreedyBySize
from EnvConfig import EnvConfig
from scipy.ndimage import uniform_filter1d


@dataclass
class Metrics:
    qoe: List[float] = field(default_factory=list)
    latency: List[float] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)
    battery: List[float] = field(default_factory=list)
    offload_ratio: List[float] = field(default_factory=list)


class Simulator:
    def __init__(self, n_ues: int = EnvConfig.NUM_UES, lam: float = 0.6):
        """Initialize a 3-layer simulation environment."""
        # === Layers ===
        self.bs = BaseStation()
        self.mec = MECServer()
        self.cloud = CloudServer(fiber_distance_m=EnvConfig.FIBER_DISTANCE)

        # === Devices ===
        self.ues = []
        for i in range(n_ues):
            x = np.random.uniform(5, 100)
            y = np.random.uniform(5, 100)
            self.ues.append(UE(n=i, x_m=x, y_m=y))

        # === Task factory & traffic ===
        self.factory = TaskFactory()
        self.n_ues = n_ues
        self.lam = lam  # task arrival rate (Poisson λ)

    # ----------------------------------------------------------
    def step_once(self, policy: Policy) -> Dict[str, float]:
        """Simulate one timestep with a given policy."""
        arrivals = np.random.poisson(self.lam)
        if arrivals == 0:
            # still drain idle power consumption
            for ue in self.ues:
                ue.drain_idle()
            return {
                "qoe": 0.0,
                "lat": 0.0,
                "eng": 0.0,
                "offload": 0.0,
                "avg_batt": np.mean([u.battery_j for u in self.ues]),
            }

        qoe_sum = lat_sum = eng_sum = offload_ct = 0.0

        for _ in range(arrivals):
            # Randomly pick a UE to generate a task
            ue = np.random.choice(self.ues)
            task = self.factory.sample()
            action = policy.decide(task, ue)

            # Determine execution site
            if action == "local":
                latency = ue.local_latency(task.cpu_cycles)
                energy = ue.local_energy(task.cpu_cycles)
            elif action == "mec":
                latency, energy = ue.offload_to_mec(task, self.bs, self.mec, self.n_ues)
                offload_ct += 1
            else:  # "cloud"
                latency, energy = ue.offload_to_cloud(task, self.bs, self.cloud, self.n_ues)
                offload_ct += 1

            # QoE Calculation (Eq. 18 from paper)
            success = latency <= task.latency_deadline

            if success:
                # Compute normalized energy fraction depending on offload site (α)
                if action == "local":
                    qoe = - energy / ue.battery_j  # e_cl_i / B^n
                elif action == "mec":
                    qoe = - energy / ue.battery_j  # e_m_i / B^n
                elif action == "cloud":
                    qoe = - energy / ue.battery_j  # e_c_i / B^n
            else:
                qoe = EnvConfig.FAIL_PENALTY  # η

            # Update UE state
            ue.battery_j = max(ue.battery_j - energy, 0.0)
            ue.drain_idle()  # apply residual drain every timestep

            qoe_sum += qoe
            lat_sum += latency
            eng_sum += energy

        avg_batt = float(np.mean([u.battery_j for u in self.ues]))
        return {
            "qoe": qoe_sum / max(arrivals, 1),
            "lat": lat_sum / max(arrivals, 1),
            "eng": eng_sum / max(arrivals, 1),
            "offload": offload_ct / max(arrivals, 1),
            "avg_batt": avg_batt,
        }

    # ----------------------------------------------------------
    def run(self, T: int, policy: Policy) -> Metrics:
        m = Metrics()
        for t in range(T):
            results = self.step_once(policy)
            m.qoe.append(results["qoe"])
            m.latency.append(results["lat"])
            m.energy.append(results["eng"])
            m.offload_ratio.append(results["offload"])
            m.battery.append(results["avg_batt"])

        # Optional smoothing for QoE curves
        m.qoe = list(uniform_filter1d(m.qoe, size=10))
        return m
