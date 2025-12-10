import numpy as np
import torch
from typing import Tuple, Optional, List

from EnvConfig import EnvConfig
from models import UE, BaseStation, MECServer, CloudServer, TaskFactory, Task


class OffloadEnv:
    """
    Single-UE RL environment with Poisson task arrivals.

    Tasks arrive according to a Poisson process (λ = TASK_ARRIVAL_RATE).
    Agent chooses an action for each arriving task:
        0 = local, 1 = MEC, 2 = Cloud.

    Reward follows the QoE definition from paper Equation 18:
    - Successful tasks: QoE = -E_consumed / B_n (current battery)
    - Failed tasks: QoE = η (FAIL_PENALTY = -0.1)
    """

    def __init__(
        self,
        ue: UE,
        bs: BaseStation,
        mec: MECServer,
        cloud: CloudServer,
        max_steps: int = EnvConfig.TOTAL_TIME_T,
        task_mode: str = "random",
        task_arrival_rate: float = None,
    ):
        self.ue = ue
        self.bs = bs
        self.mec = mec
        self.cloud = cloud
        self.max_steps = max_steps
        self.task_factory = TaskFactory(mode=task_mode)
        self.step_count = 0
        
        # Use same arrival rate as baselines
        self.lam = task_arrival_rate if task_arrival_rate is not None else EnvConfig.TASK_ARRIVAL_RATE

        # Track current tasks for this timestep
        self.current_tasks: List[Task] = []
        self.task_index = 0  # Index of current task being processed

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
        Resets UE battery and episode counters, samples tasks for first timestep,
        and returns the initial state.
        """
        self.ue.battery_j = EnvConfig.UE_MAX_BATTERY
        self.step_count = 0
        
        # Sample tasks for first timestep using Poisson arrivals
        num_arrivals = np.random.poisson(self.lam)
        self.current_tasks = [self.task_factory.sample() for _ in range(num_arrivals)]
        self.task_index = 0
        
        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        action: 0=local, 1=MEC, 2=Cloud

        Returns:
            next_state, reward, done, info
        """
        assert action in (0, 1, 2), f"Invalid action {action}"

        ue = self.ue
        
        # Check if we have tasks to process in current timestep
        if self.task_index >= len(self.current_tasks):
            # No more tasks in this timestep, move to next timestep
            self.step_count += 1
            
            # Apply idle drain for this timestep
            ue.drain_idle()
            
            # Check if episode is done
            done = (ue.battery_j <= 0.0) or (self.step_count >= self.max_steps)
            
            if done:
                return self._build_state(), 0.0, True, {"note": "episode_end", "battery": ue.battery_j}
            
            # Sample new tasks for next timestep
            num_arrivals = np.random.poisson(self.lam)
            self.current_tasks = [self.task_factory.sample() for _ in range(num_arrivals)]
            self.task_index = 0
            
            # If no tasks arrived, return zero reward and continue
            if len(self.current_tasks) == 0:
                return self._build_state(), 0.0, False, {"note": "no_arrivals", "battery": ue.battery_j}
        
        # Get current task
        task = self.current_tasks[self.task_index]
        self.task_index += 1

        # If UE already dead, give penalty
        if ue.battery_j <= 0:
            reward = EnvConfig.FAIL_PENALTY
            done = True

            info = {
                "latency": 0.0,
                "energy": 0.0,
                "success": False,
                "battery": ue.battery_j,
                "task_class": task.cls,
                "deadline": task.latency_deadline,
                "dead": True,
            }
            return self._build_state(), float(reward), bool(done), info

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

        # ---------------- QoE Reward (Paper Equation 18) ----------------
        # Successful tasks: QoE = -E_consumed / B_n (current battery)
        # Failed tasks: QoE = η (FAIL_PENALTY)
        success = latency <= task.latency_deadline
        
        if success:
            # Use CURRENT battery B_n as denominator (as per paper Eq. 18)
            if ue.battery_j > 0:
                reward = -(energy / ue.battery_j)
            else:
                # Edge case: battery is exactly 0
                reward = EnvConfig.FAIL_PENALTY
        else:
            # Missed deadline → fixed penalty
            reward = EnvConfig.FAIL_PENALTY  # η = −0.1
        # ------------------------------------------------------------------------------

        # Update UE battery (no idle drain here, done at timestep boundary)
        ue.battery_j = max(ue.battery_j - energy, 0.0)

        # Check if done after processing this task
        done = (ue.battery_j <= 0.0) or (self.step_count >= self.max_steps and self.task_index >= len(self.current_tasks))

        # Build next state
        next_state = self._build_state()
        info = {
            "latency": latency,
            "energy": energy,
            "success": success,
            "battery": ue.battery_j,
            "task_class": task.cls,
            "deadline": task.latency_deadline,
            "timestep": self.step_count,
            "task_index": self.task_index - 1,
            "total_tasks_in_timestep": len(self.current_tasks),
        }
        return next_state, float(reward), bool(done), info

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------
    def _build_state(self) -> np.ndarray:
        ue = self.ue
        
        # If battery empty, return zeros
        if ue.battery_j <= 0.0:
            return np.zeros(10, dtype=np.float32)
        
        # If no more tasks in current timestep, use zeros for task features
        if self.task_index >= len(self.current_tasks):
            # Battery and resource states
            b = ue.battery_j / self.batt_max
            r_ue = ue.cpu_hz / EnvConfig.UE_MAX_COMPUTATION_RESOURCES
            r_mec = self.mec.f_available_hz / EnvConfig.MEC_MAX_COMPUTATION_RESOURCES
            r_cloud = self.cloud.f_available_hz / EnvConfig.CLOUD_TRANSMISSION_POWER
            h_lin = self.bs.channel_gain_linear(ue.f_c_ghz, ue.distance_to_bs_m)
            h_norm = np.clip(-np.log10(h_lin + 1e-12) / 10.0, 0.0, 1.0)
            
            # No task features
            return np.array(
                [b, r_ue, r_mec, r_cloud, h_norm, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
        
        task = self.current_tasks[self.task_index]

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
