from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from models import UE, BaseStation, MECServer, CloudServer, TaskFactory
from policy import Policy, AlwaysLocal, GreedyBySize

@dataclass
class Metrics:
    qoe: List[float] = field(default_factory=list)
    latency: List[float] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)
    battery: List[float] = field(default_factory=list)
    offload_ratio: List[float] = field(default_factory=list)

class Simulator:
    def __init__(self, n_ues: int = 10, lam: float = 0.6):
        self.ues = [UE(distance_to_bs_m=np.random.uniform(5, 100)) for _ in range(n_ues)]
        self.bs = BaseStation()
        self.mec = MECServer()
        self.cloud = CloudServer()
        self.factory = TaskFactory()
        self.n_ues = n_ues
        self.lam = lam

    def step_once(self, policy: Policy) -> Dict[str, float]:
        # Poisson arrivals (aggregate to one UE chosen uniformly to keep it simple)
        arrivals = np.random.poisson(self.lam)
        if arrivals == 0:
            return {"qoe": 0.0, "lat": 0.0, "eng": 0.0, "offload": 0.0, "avg_batt": np.mean([u.battery_j for u in self.ues])}

        qoe_sum = lat_sum = eng_sum = offload_ct = 0.0
        for _ in range(arrivals):
            ue = np.random.choice(self.ues)
            task = self.factory.sample()
            act = policy.decide(task, ue)

            if act == "local":
                latency = ue.local_latency(task.cpu_cycles)
                energy  = ue.local_energy(task.cpu_cycles)
            elif act == "mec":
                latency, energy = ue.offload_to_mec(task, self.bs, self.mec, self.n_ues)
                offload_ct += 1
            else:  # "cloud"
                latency, energy = ue.offload_to_cloud(task, self.bs, self.cloud, self.n_ues)
                offload_ct += 1

            # success if latency <= deadline
            success = latency <= task.latency_deadline
            # QoE ~ -energy / remaining_battery; punishment if fail
            # scale factor keeps values visible
            if success:
                denom = max(ue.battery_j, 1e-6)
                qoe = - (energy / denom) * 1e6
            else:
                qoe = -0.1  # penalty (η)

            # Update battery
            ue.battery_j = max(ue.battery_j - energy, 0.0)

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

    def run(self, T: int, policy: Policy) -> Metrics:
        m = Metrics()
        for _ in range(T):
            out = self.step_once(policy)
            m.qoe.append(out["qoe"])
            m.latency.append(out["lat"])
            m.energy.append(out["eng"])
            m.offload_ratio.append(out["offload"])
            m.battery.append(out["avg_batt"])
        return m
