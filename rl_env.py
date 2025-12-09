import numpy as np
import torch
from typing import Tuple, Optional

from EnvConfig import EnvConfig
from models import UE, BaseStation, MECServer, CloudServer, TaskFactory, Task


class OffloadEnv:
    """
    Single-UE RL environment.

    One task arrives per step; agent chooses an action:
        0 = local, 1 = MEC, 2 = Cloud.

    Reward follows the QoE definition used in Simulator.step_once:
    successful tasks get a value in [FAIL_PENALTY, 0],
    failures get FAIL_PENALTY.
    """

    def __init__(
        self,
        ue: UE,
        bs: BaseStation,
        mec: MECServer,
        cloud: CloudServer,
        max_steps: int = EnvConfig.TOTAL_TIME_T,
        task_mode: str = "random",
    ):
        self.ue = ue
        self.bs = bs
        self.mec = mec
        self.cloud = cloud
        self.max_steps = max_steps
        self.task_factory = TaskFactory(mode=task_mode)
        self.step_count = 0

        # Track current task
        self.current_task: Optional[Task] = None

        # Normalization constants (rough, based on TaskFactory ranges & config)
        self.batt_max = EnvConfig.UE_MAX_BATTERY
        # Max distance from (0,0) in Simulator is sqrt(100^2 + 100^2)
        self.dist_max = float(np.sqrt(100.0 ** 2 + 100.0 ** 2))
        # Max task size ≈ class-3 upper bound
        self.D_max = 400e3 * 8.0
        # Max deadline ≈ class-3 upper bound
        # T_i = 2e-3 * (D/8) so for max D:
        self.T_max = 2e-3 * (400e3)

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """
        Resets UE battery and episode counters, samples the first task,
        and returns the initial state.
        """
        self.ue.battery_j = EnvConfig.UE_MAX_BATTERY
        self.step_count = 0
        self.current_task = self.task_factory.sample()
        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        action: 0=local, 1=MEC, 2=Cloud

        Returns:
            next_state, reward, done, info
        """
        assert action in (0, 1, 2), f"Invalid action {action}"

        self.step_count += 1
        ue = self.ue
        task = self.current_task

        if task is None:
            # Should not normally happen; treat as terminal
            return self._build_state(), 0.0, True, {"note": "no_task"}

        # If UE already dead, give penalty and end episode
        if ue.battery_j <= 0:
            return self._build_state(), EnvConfig.FAIL_PENALTY, True, {"dead": True}

        # --- compute latency and energy using existing model methods ---
        if action == 0:
            latency = ue.local_latency(task.cpu_cycles)
            energy = ue.local_energy(task.cpu_cycles)
        elif action == 1:
            latency, energy = ue.offload_to_mec(
                task, self.bs, self.mec, n_ues=EnvConfig.NUM_UES
            )
        else:
            latency, energy = ue.offload_to_cloud(
                task, self.bs, self.cloud, n_ues=EnvConfig.NUM_UES
            )

        # QoE-style reward
        success = latency <= task.latency_deadline
        if success:
            qoe_raw = -(energy / self.batt_max) * 1000.0
            reward = max(EnvConfig.FAIL_PENALTY, min(0.0, qoe_raw))
        else:
            reward = EnvConfig.FAIL_PENALTY

        # Update UE battery + idle drain
        ue.battery_j = max(ue.battery_j - energy, 0.0)
        ue.drain_idle()

        done = (ue.battery_j <= 0.0) or (self.step_count >= self.max_steps)

        # Prepare next state
        self.current_task = self.task_factory.sample()
        next_state = self._build_state()
        info = {
            "latency": latency,
            "energy": energy,
            "success": success,
            "battery": ue.battery_j,
            "task_class": task.cls,
        }
        return next_state, float(reward), bool(done), info

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------
    def _build_state(self) -> np.ndarray:
        ue = self.ue
        task = self.current_task

        # If no task or battery empty, return zeros
        if task is None or ue.battery_j <= 0.0:
            return np.zeros(10, dtype=np.float32)

        # Battery
        b = ue.battery_j / self.batt_max

        # CPU resources: currently constant, but normalize anyway
        r_ue = ue.cpu_hz / EnvConfig.UE_MAX_COMPUTATION_RESOURCES
        r_mec = self.mec.f_available_hz / EnvConfig.MEC_MAX_COMPUTATION_RESOURCES
        # cheap proxy normalizer for cloud
        r_cloud = self.cloud.f_available_hz / EnvConfig.CLOUD_TRANSMISSION_POWER

        # Channel gain: map path-loss-based gain to a roughly [0,1] range
        h_lin = self.bs.channel_gain_linear(ue.f_c_ghz, ue.distance_to_bs_m)
        # use log scale (higher gain → lower value)
        h_norm = np.clip(-np.log10(h_lin + 1e-12) / 10.0, 0.0, 1.0)

        # Task features
        D_norm = task.data_bits / self.D_max
        T_norm = task.latency_deadline / self.T_max

        cls_oh = np.zeros(3, dtype=np.float32)
        cls_oh[task.cls - 1] = 1.0

        state = np.array(
            [b, r_ue, r_mec, r_cloud, h_norm, D_norm, T_norm, *cls_oh],
            dtype=np.float32,
        )
        return state

    # Small helper to move the current state to a torch tensor on a target device.
    def state_tensor(self, device: str = "cpu") -> torch.Tensor:
        return torch.tensor(self._build_state(), dtype=torch.float32, device=device)
