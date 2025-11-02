from dataclasses import dataclass
from typing import Optional, Tuple
import math
import random

# ----------------------------
# Task model (Classes 1/2/3)
# ----------------------------
@dataclass
class Task:
    data_bits: float          # D_i (bits)
    cpu_cycles: float         # φ_i (cycles)
    latency_deadline: float   # T_req_i (seconds)
    cls: int                  # {1: delay-sensitive, 2: energy-sensitive, 3: insensitive}

class TaskFactory:
    """Sampling consistent with the paper's three classes (tunable ranges)."""
    def sample(self) -> Task:
        cls = random.choices([1, 2, 3], weights=[1, 1, 1], k=1)[0]
        if cls == 1:  # small, strict deadline
            D = random.uniform(10e3, 40e3) * 8          # bytes→bits
            phi = D                                    # ≈ 1 cycle/bit (toy; tune)
            T = 0.5e-3 * (D / 8.0)                     # scales with payload (toy)
        elif cls == 2:  # larger compute, some deadline
            D = random.uniform(20e3, 50e3) * 8
            phi = 8 * D
            T = 1e-3 * (D / 8.0)
        else:  # cls=3 big/loose
            D = random.uniform(200e3, 400e3) * 8
            phi = 8 * D
            T = 2e-3 * (D / 8.0)
        return Task(D, phi, T, cls)

# ----------------------------
# Servers
# ----------------------------
@dataclass
class MECServer:
    f_available_hz: float = 1e9   # 1 GHz effective
    def proc_time(self, cpu_cycles: float) -> float:
        return cpu_cycles / self.f_available_hz

@dataclass
class CloudServer:
    f_available_hz: float = 10e9  # 10 GHz effective
    fiber_capacity_bps: float = 100e9  # 100 Gbps (effective after overheads)
    fiber_prop_speed: float = 2e8      # m/s in fiber
    fiber_distance_m: float = 50.0     # BS↔Cloud
    overhead: float = 0.10
    fec: float = 0.20
    wdm_factor: float = math.sqrt(16)  # WDM gain for 16-QAM (toy)

    def proc_time(self, cpu_cycles: float) -> float:
        return cpu_cycles / self.f_available_hz

    def fiber_rate(self) -> float:
        return (self.fiber_capacity_bps / self.wdm_factor) * (1 - self.overhead) * (1 - self.fec)

    def fiber_tx_delay(self, data_bits: float) -> float:
        return data_bits / max(self.fiber_rate(), 1.0)

    def fiber_prop_delay_roundtrip(self) -> float:
        return 2.0 * (self.fiber_distance_m / self.fiber_prop_speed)

# ----------------------------
# Radio access (BS) & channel
# ----------------------------
@dataclass
class BaseStation:
    total_bw_hz: float = 100e6   # 100 MHz
    noise_w: float = 1e-10       # -100 dBW
    shadow_sigma_db: float = 5.9

    def ofdma_subband(self, n_ues: int) -> float:
        return self.total_bw_hz / max(n_ues, 1)

    def path_loss_abg_db(self, fc_ghz: float, d_m: float) -> float:
        # 3GPP Indoor factory SH: PL = 32.4 + 23 log10(d) + 20 log10(fc)
        return 32.4 + 23.0 * math.log10(max(d_m, 1e-3)) + 20.0 * math.log10(max(fc_ghz, 1e-3))

    def shadow_fading_db(self) -> float:
        return random.gauss(0.0, self.shadow_sigma_db)

    def channel_gain_linear(self, fc_ghz: float, d_m: float) -> float:
        pl_db = self.path_loss_abg_db(fc_ghz, d_m) + self.shadow_fading_db()
        return 10 ** (-pl_db / 10.0)

    def uplink_rate_bps(self, p_tx_w: float, h_lin: float, bw_hz: float) -> float:
        snr = (p_tx_w * h_lin) / (self.noise_w)
        return bw_hz * math.log2(1.0 + max(snr, 1e-12))

# ----------------------------
# UE (device)
# ----------------------------
@dataclass
class UE:
    cpu_hz: float = 40e6             # 40 MHz effective
    battery_j: float = 4000.0        # J
    p_tx_w: float = 1.0              # W (tunable)
    f_c_ghz: float = 3.5             # carrier frequency
    distance_to_bs_m: float = 30.0   # UE↔BS distance
    kappa: float = 1e-21             # chip energy coefficient

    # Local execution
    def local_latency(self, cpu_cycles: float) -> float:
        return cpu_cycles / self.cpu_hz

    def local_energy(self, cpu_cycles: float) -> float:
        return self.kappa * (self.cpu_hz ** 2) * cpu_cycles

    # Offload to MEC
    def offload_to_mec(self, task: Task, bs: BaseStation, mec: MECServer, n_ues: int) -> Tuple[float, float]:
        bw = bs.ofdma_subband(n_ues)
        h = bs.channel_gain_linear(self.f_c_ghz, self.distance_to_bs_m)
        r = bs.uplink_rate_bps(self.p_tx_w, h, bw)
        t_tx = task.data_bits / max(r, 1.0)
        t_proc = mec.proc_time(task.cpu_cycles)
        # downlink negligible (small result + high P_tx at MEC) → ignore
        latency = t_tx + t_proc
        energy = self.p_tx_w * t_tx
        return latency, energy

    # Offload to Cloud (via BS fiber)
    def offload_to_cloud(self, task: Task, bs: BaseStation, cloud: CloudServer, n_ues: int) -> Tuple[float, float]:
        bw = bs.ofdma_subband(n_ues)
        h = bs.channel_gain_linear(self.f_c_ghz, self.distance_to_bs_m)
        r = bs.uplink_rate_bps(self.p_tx_w, h, bw)
        t_tx_bs = task.data_bits / max(r, 1.0)
        t_fiber = cloud.fiber_tx_delay(task.data_bits)
        t_prop = cloud.fiber_prop_delay_roundtrip()
        t_proc = cloud.proc_time(task.cpu_cycles)
        latency = t_tx_bs + t_fiber + t_prop + t_proc
        energy = self.p_tx_w * t_tx_bs  # UE spends energy only on the wireless uplink
        return latency, energy
