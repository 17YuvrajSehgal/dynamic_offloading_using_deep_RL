from dataclasses import dataclass
from typing import Optional, Tuple
import math
import random

from EnvConfig import EnvConfig


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
    """
    Factory for generating tasks according to the three task classes
    described in the paper. Supports both random sampling and
    class-specific generation.
    """

    def __init__(self, mode: str = "random", fixed_class: int = None):
        """
        mode: "random" → choose among classes 1/2/3 randomly (default)
              "fixed"  → always generate tasks of `fixed_class`
        fixed_class: int ∈ {1, 2, 3} when mode="fixed"
        """
        self.mode = mode
        self.fixed_class = fixed_class

    def sample(self) -> Task:
        """Generate a single task according to the selected mode."""
        # Choose task class based on mode
        if self.mode == "fixed" and self.fixed_class in [1, 2, 3]:
            cls = self.fixed_class
        else:
            cls = random.choices([1, 2, 3], weights=[1, 1, 1], k=1)[0]

        # --- Class-specific parameter generation ---
        if cls == 1:  # Delay-sensitive: small, strict deadline
            D_i = random.uniform(10e3, 40e3) * 8  # bytes→bits
            phi = D_i * 8                          # 8 cycles/bit (from Table 3)
            T_i = 0.5e-3 * (D_i / 8.0)            # short deadline
        elif cls == 2:  # Energy-sensitive: medium, moderate deadline
            D_i = random.uniform(20e3, 50e3) * 8
            phi = 8 * D_i
            T_i = 1e-3 * (D_i / 8.0)
        else:  # cls == 3 → insensitive (large, loose deadline)
            D_i = random.uniform(200e3, 400e3) * 8
            phi = 8 * D_i
            T_i = 2e-3 * (D_i / 8.0)
        return Task(D_i, phi, T_i, cls)

# ----------------------------
# Servers
# ----------------------------
@dataclass
class MECServer:
    # Since results are small, the paper ignores MEC transmission time and energy.
    tx_power_dbm: float = EnvConfig.MEC_TRANSMISSION_POWER              # (P_tx^m) transmission_power_mec
    f_available_hz: float = EnvConfig.MEC_MAX_COMPUTATION_RESOURCES     # (F^m)_i Max. computation resources 1 GHz

    def proc_time(self, cpu_cycles: float) -> float:
        return cpu_cycles / self.f_available_hz

@dataclass
class CloudServer:
    # Cloud configuration
    f_available_hz: float = EnvConfig.CLOUD_TRANSMISSION_POWER       # (F^S)_i Max. computation resources 10 GHz

    # Optical Fiber Configuration
    fiber_distance_m: float = EnvConfig.CLOUD_FIBER_DISTANCE            # BS↔Cloud
    fiber_capacity_bps: float = EnvConfig.OPTICAL_FIBER_CAPACITY        # (C^f) Capacity 100 Gbps
    wdm_factor: float = math.sqrt(EnvConfig.OPTICAL_FIBER_WDM)          # (WDM) Modulation
    overhead: float = EnvConfig.OPTICAL_FIBER_OVERHEAD                  # (O^f) Overhead 10%
    fec: float = EnvConfig.OPTICAL_FIBER_FEC                            # (F^f) FEC 20%
    refractive_index: float = EnvConfig.OPTICAL_FIBER_REFRACTIVE_INDEX  # (ρ) Refractive Index

    @property
    def fiber_prop_speed(self) -> float:
        # v = c / ρ (c => speed of light 3e8 m/s)
        return 3e8 / self.refractive_index

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
    total_bw_hz: float = EnvConfig.BS_TOTAL_CHANNEL_BANDWIDTH   # (W) Total channel bandwidth 100 MHz
    noise_w: float = EnvConfig.BS_NOISE                         # (N_0) AWGN noise power
    shadow_sigma_db: float = EnvConfig.BS_SHADOW_SIGMA_DB       # Shadow fading std dev

    def ofdma_subband(self, n_ues: int) -> float:
        """Calculate bandwidth per UE using OFDMA."""
        return self.total_bw_hz / max(n_ues, 1)

    def path_loss_abg_db(self, fc_ghz: float, d_m: float) -> float:
        """3GPP Indoor factory SH path loss model."""
        # PL = 32.4 + 23*log10(d) + 20*log10(fc)
        return 32.4 + 23.0 * math.log10(max(d_m, 1e-3)) + 20.0 * math.log10(max(fc_ghz, 1e-3))

    def shadow_fading_db(self) -> float:
        """Generate shadow fading sample."""
        return random.gauss(0.0, self.shadow_sigma_db)

    def channel_gain_linear(
        self, 
        fc_ghz: float, 
        d_m: float, 
        quality_multiplier: float = 1.0
    ) -> float:
        """
        Calculate channel gain with optional quality multiplier for scenarios.
        
        Args:
            fc_ghz: Carrier frequency in GHz
            d_m: Distance in meters
            quality_multiplier: Channel quality multiplier (1.0 = normal, <1.0 = degraded)
                               Used in Scenario 2 for channel degradation periods
        
        Returns:
            Linear channel gain
        """
        pl_db = self.path_loss_abg_db(fc_ghz, d_m) + self.shadow_fading_db()
        gain_linear = 10 ** (-pl_db / 10.0)
        
        # Apply quality multiplier (for Scenario 2: channel degradation)
        # quality_multiplier < 1.0 means degraded channel
        return gain_linear * quality_multiplier
    
    def uplink_rate_bps(self, p_tx_w: float, h_lin: float, bw_hz: float) -> float:
        """Calculate uplink rate using Shannon capacity formula."""
        snr = (p_tx_w * h_lin) / self.noise_w
        return bw_hz * math.log2(1.0 + max(snr, 1e-12))

# ----------------------------
# UE (device)
# ----------------------------
@dataclass
class UE:
    n: int = 0                                                          # UE index
    x_m: float = 0.0                                                    # X coordinate (meters)
    y_m: float = 0.0                                                    # Y coordinate (meters)
    cpu_hz: float = EnvConfig.UE_MAX_COMPUTATION_RESOURCES              # (F^n) CPU frequency
    kappa: float = EnvConfig.UE_KAPPA                                   # (κ^n) Energy coefficient
    residual_j_per_t: float = EnvConfig.UE_RESIDUAL_J_PER_T             # (b_r^n) Idle consumption
    battery_j: float = EnvConfig.UE_MAX_BATTERY                         # (B^n) Battery capacity
    p_tx_w: float = EnvConfig.UE_TRANSMISSION_POWER                     # (P_tx^n) TX power (Watts)
    f_c_ghz: float = EnvConfig.UE_CARRIER_FREQUENCY                     # (f_c) Carrier frequency

    @property
    def distance_to_bs_m(self) -> float:
        """Calculate Euclidean distance to BS at (0, 0)."""
        return math.sqrt(self.x_m ** 2 + self.y_m ** 2)

    def drain_idle(self):
        """Apply idle battery drain for one timestep."""
        if self.battery_j > 0:
            self.battery_j = max(self.battery_j - self.residual_j_per_t, 0)

    # Local execution
    def local_latency(self, cpu_cycles: float, active_tasks: int = 1) -> float:
        """Calculate latency for local task execution."""
        return cpu_cycles / (self.cpu_hz / active_tasks)

    def local_energy(self, cpu_cycles: float) -> float:
        """Calculate energy for local task execution (DVFS model)."""
        return self.kappa * (self.cpu_hz ** 2) * cpu_cycles

    # Offload to MEC
    def offload_to_mec(
        self, 
        task: Task, 
        bs: BaseStation, 
        mec: MECServer, 
        n_ues: int,
        channel_quality_multiplier: float = 1.0
    ) -> Tuple[float, float]:
        """
        Offload task to MEC server.
        
        Args:
            task: Task to offload
            bs: Base station
            mec: MEC server
            n_ues: Number of active UEs (for bandwidth sharing)
            channel_quality_multiplier: Channel quality multiplier for scenarios
        
        Returns:
            (latency, energy) tuple
        """
        bw = bs.ofdma_subband(n_ues)
        h = bs.channel_gain_linear(self.f_c_ghz, self.distance_to_bs_m, channel_quality_multiplier)
        r = bs.uplink_rate_bps(self.p_tx_w, h, bw)
        
        t_tx = task.data_bits / max(r, 1.0)
        t_proc = mec.proc_time(task.cpu_cycles)
        
        # Downlink negligible (small result + high P_tx at MEC) → ignore
        latency = t_tx + t_proc
        energy = self.p_tx_w * t_tx
        
        return latency, energy

    # Offload to Cloud (via BS + fiber)
    def offload_to_cloud(
        self, 
        task: Task, 
        bs: BaseStation, 
        cloud: CloudServer, 
        n_ues: int,
        channel_quality_multiplier: float = 1.0
    ) -> Tuple[float, float]:
        """
        Offload task to cloud server via BS and fiber link.
        
        Args:
            task: Task to offload
            bs: Base station
            cloud: Cloud server
            n_ues: Number of active UEs (for bandwidth sharing)
            channel_quality_multiplier: Channel quality multiplier for scenarios
        
        Returns:
            (latency, energy) tuple
        """
        bw = bs.ofdma_subband(n_ues)
        h = bs.channel_gain_linear(self.f_c_ghz, self.distance_to_bs_m, channel_quality_multiplier)
        r = bs.uplink_rate_bps(self.p_tx_w, h, bw)
        
        # Wireless uplink: UE → BS
        t_tx_bs = task.data_bits / max(r, 1.0)
        
        # Fiber link: BS → Cloud
        t_fiber = cloud.fiber_tx_delay(task.data_bits)
        t_prop = cloud.fiber_prop_delay_roundtrip()
        
        # Cloud processing
        t_proc = cloud.proc_time(task.cpu_cycles)
        
        # Total latency
        latency = t_tx_bs + t_fiber + t_prop + t_proc
        
        # UE energy: only wireless uplink (fiber is BS→Cloud)
        energy = self.p_tx_w * t_tx_bs
        
        return latency, energy
