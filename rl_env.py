import numpy as np
import torch
from typing import Tuple, Optional, List

from EnvConfig import EnvConfig
from models import UE, BaseStation, MECServer, CloudServer, TaskFactory, Task


class OffloadEnv:
    """
    Single-UE RL environment with Poisson task arrivals and scenario support.

    Tasks arrive according to a Poisson process (λ = TASK_ARRIVAL_RATE).
    Agent chooses an action for each arriving task:
        0 = local, 1 = MEC, 2 = Cloud.

    Reward follows the QoE definition from paper Equation 18:
    - Successful tasks: QoE = -E_consumed / B_n (current battery)
    - Failed tasks: QoE = η (FAIL_PENALTY = -0.1)
    
    Supports scenarios with:
    - Time-varying MEC availability
    - Time-varying channel quality
    - Task class distribution
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
        scenario_config=None,  # ScenarioConfig instance
    ):
        self.ue = ue
        self.bs = bs
        self.mec = mec
        self.cloud = cloud
        self.max_steps = max_steps
        self.task_factory = TaskFactory(mode=task_mode)
        self.step_count = 0
        
        # Scenario configuration
        self.scenario_config = scenario_config
        
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
        self.current_tasks = self._sample_tasks(num_arrivals)
        self.task_index = 0
        
        return self._build_state()
    
    def _sample_tasks(self, num_tasks: int) -> List[Task]:
        """Sample tasks according to scenario configuration or default factory."""
        if self.scenario_config is not None:
            # Use scenario's task distribution
            tasks = []
            for _ in range(num_tasks):
                task_class = self.scenario_config.sample_task_class()
                # Create task factory for specific class
                factory = TaskFactory(mode="fixed", fixed_class=task_class)
                tasks.append(factory.sample())
            return tasks
        else:
            # Use default task factory
            return [self.task_factory.sample() for _ in range(num_tasks)]
    
    def _is_mec_available(self) -> bool:
        """Check if MEC is available at current timestep."""
        if self.scenario_config is None:
            return True  # Always available if no scenario
        return self.scenario_config.is_mec_available(self.step_count)
    
    def _get_channel_quality_multiplier(self) -> float:
        """Get channel quality multiplier for current timestep."""
        if self.scenario_config is None:
            return 1.0  # Normal quality if no scenario
        return self.scenario_config.get_channel_quality_multiplier(self.step_count)

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
            self.current_tasks = self._sample_tasks(num_arrivals)
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

        # --- Check MEC availability (Scenario 1) ---
        mec_available = self._is_mec_available()
        if action == 1 and not mec_available:
            # MEC requested but unavailable → treat as failed task
            latency = task.latency_deadline * 10.0  # Artificially high latency
            energy = 0.0  # No energy consumed if can't offload
            success = False
            reward = EnvConfig.FAIL_PENALTY
            
            info = {
                "latency": latency,
                "energy": energy,
                "success": success,
                "battery": ue.battery_j,
                "task_class": task.cls,
                "deadline": task.latency_deadline,
                "mec_unavailable": True,
                "timestep": self.step_count,
                "task_index": self.task_index - 1,
                "total_tasks_in_timestep": len(self.current_tasks),
            }
            
            # Check if done
            done = (ue.battery_j <= 0.0) or (self.step_count >= self.max_steps and self.task_index >= len(self.current_tasks))
            return self._build_state(), float(reward), bool(done), info

        # --- Compute latency and energy using existing model methods ---
        if action == 0:
            latency = ue.local_latency(task.cpu_cycles)
            energy = ue.local_energy(task.cpu_cycles)
        elif action == 1:
            # Apply channel quality multiplier (Scenario 2)
            channel_multiplier = self._get_channel_quality_multiplier()
            latency, energy = ue.offload_to_mec(
                task, self.bs, self.mec, n_ues=EnvConfig.NUM_UES,
                channel_quality_multiplier=channel_multiplier
            )
        else:  # action == 2
            # Apply channel quality multiplier (Scenario 2)
            channel_multiplier = self._get_channel_quality_multiplier()
            latency, energy = ue.offload_to_cloud(
                task, self.bs, self.cloud, n_ues=EnvConfig.NUM_UES,
                channel_quality_multiplier=channel_multiplier
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
            "mec_available": mec_available,
            "channel_quality": self._get_channel_quality_multiplier(),
        }
        return next_state, float(reward), bool(done), info

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------
    def _build_state(self) -> np.ndarray:
        ue = self.ue
        
        # If battery empty, return zeros
        if ue.battery_j <= 0.0:
            return np.zeros(12, dtype=np.float32)  # Increased size for scenario features
        
        # If no more tasks in current timestep, use zeros for task features
        if self.task_index >= len(self.current_tasks):
            # Battery and resource states
            b = ue.battery_j / self.batt_max
            r_ue = ue.cpu_hz / EnvConfig.UE_MAX_COMPUTATION_RESOURCES
            r_mec = self.mec.f_available_hz / EnvConfig.MEC_MAX_COMPUTATION_RESOURCES
            r_cloud = self.cloud.f_available_hz / EnvConfig.CLOUD_TRANSMISSION_POWER
            h_lin = self.bs.channel_gain_linear(ue.f_c_ghz, ue.distance_to_bs_m)
            # Apply channel quality multiplier
            h_lin *= self._get_channel_quality_multiplier()
            h_norm = np.clip(-np.log10(h_lin + 1e-12) / 10.0, 0.0, 1.0)
            
            # Scenario features
            mec_avail = 1.0 if self._is_mec_available() else 0.0
            ch_quality = self._get_channel_quality_multiplier()
            
            # No task features
            return np.array(
                [b, r_ue, r_mec, r_cloud, h_norm, 0.0, 0.0, 0.0, 0.0, 0.0, mec_avail, ch_quality],
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
        # Apply channel quality multiplier (Scenario 2)
        h_lin *= self._get_channel_quality_multiplier()
        # use log scale (higher gain → lower value)
        h_norm = np.clip(-np.log10(h_lin + 1e-12) / 10.0, 0.0, 1.0)

        # Task features
        D_norm = task.data_bits / self.D_max
        T_norm = task.latency_deadline / self.T_max

        cls_oh = np.zeros(3, dtype=np.float32)
        cls_oh[task.cls - 1] = 1.0
        
        # Scenario features
        mec_avail = 1.0 if self._is_mec_available() else 0.0
        ch_quality = self._get_channel_quality_multiplier()

        state = np.array(
            [b, r_ue, r_mec, r_cloud, h_norm, D_norm, T_norm, *cls_oh, mec_avail, ch_quality],
            dtype=np.float32,
        )
        return state

    # Small helper to move the current state to a torch tensor on a target device.
    def state_tensor(self, device: str = "cpu") -> torch.Tensor:
        return torch.tensor(self._build_state(), dtype=torch.float32, device=device)
