class EnvConfig:
    """
    Configuration parameters matching the paper:
    'Deep Reinforcement Learning techniques for dynamic task offloading in the 5G edge-cloud continuum'
    by Nieto et al., Journal of Cloud Computing (2024)
    
    All values from Table 2 (System Parameters)
    """

    # ===== CLOUD SERVER CONFIG =====
    # F^c: Cloud computation resources (Table 2)
    CLOUD_TRANSMISSION_POWER = 10e9  # 10 GHz
    CLOUD_FIBER_DISTANCE = 50.0      # 50 meters (distance from BS to Cloud)

    # ===== OPTICAL FIBER CONFIG =====
    # Based on paper's fiber link model
    OPTICAL_FIBER_REFRACTIVE_INDEX = 1.5  # rho: refractive index
    OPTICAL_FIBER_FEC = 0.20              # F^f: FEC overhead (20%)
    OPTICAL_FIBER_OVERHEAD = 0.10         # O^f: Protocol overhead (10%)
    OPTICAL_FIBER_WDM = 16                # WDM modulation factor (16-QAM)
    OPTICAL_FIBER_CAPACITY = 100e9        # C^f: 100 Gbps capacity

    # ===== MEC SERVER CONFIG =====
    # F^m: MEC computation resources (Table 2)
    MEC_MAX_COMPUTATION_RESOURCES = 1e9  # 1 GHz
    
    # P_tx^m: MEC/BS transmission power (Table 2)
    # Paper says 100 dBm, but this seems like an error as it would be
    # 10^7 Watts which is unrealistic. We use 40 dBm = 10 W instead.
    MEC_TRANSMISSION_POWER_DBM = 40       # 40 dBm = 10 W (reasonable for small cell BS)
    MEC_TRANSMISSION_POWER = 10 ** ((40 - 30) / 10)  # Convert to Watts

    # ===== BASE STATION CONFIG =====
    # W: Total channel bandwidth (Table 2)
    BS_TOTAL_CHANNEL_BANDWIDTH = 100e6   # 100 MHz
    
    # N_0: AWGN noise power (Table 2: -100 dBm)
    # Convert: -100 dBm = 10^((-100-30)/10) = 10^(-13) W
    BS_NOISE_DBM = -100
    BS_NOISE = 10 ** ((BS_NOISE_DBM - 30) / 10)  # = 1e-13 W
    
    # sigma_SF: Shadow fading standard deviation (Table 2)
    # 3GPP TR 38.901 for indoor factory SH
    BS_SHADOW_SIGMA_DB = 5.9

    # ===== USER EQUIPMENT (UE) CONFIG =====
    # Number of UEs in simulation
    NUM_UES = 20
    
    # F^n: UE computation resources (Table 2)
    UE_MAX_COMPUTATION_RESOURCES = 40e6  # 40 MHz
    
    # kappa^n: Energy coefficient of chip (Table 2)
    UE_KAPPA = 1e-21
    
    # b_r^n: Residual consumption per timestep (Table 2)
    UE_RESIDUAL_J_PER_T = 0.1  # 0.1 J per timestep
    
    # B^n: Maximum battery capacity (Table 2)
    UE_MAX_BATTERY = 4000  # 4000 J
    
    # P_tx^n: UE transmission power (Table 2: 30 dBm)
    # Convert: 30 dBm = 10^((30-30)/10) = 1 W
    UE_TRANSMISSION_POWER_DBM = 30
    UE_TRANSMISSION_POWER = 10 ** ((UE_TRANSMISSION_POWER_DBM - 30) / 10)  # = 1.0 W
    
    # f_c: Carrier frequency (Table 2)
    UE_CARRIER_FREQUENCY = 3.5  # 3.5 GHz

    # ===== QoE CONFIG =====
    # eta: Penalty for failed tasks (Equation 18)
    FAIL_PENALTY = -0.1

    # ===== SIMULATION CONFIG =====
    # T: Total timesteps (from paper's experiments)
    TOTAL_TIME_T = 2000
    
    # lambda: Task arrival rate - Poisson distribution (from paper)
    TASK_ARRIVAL_RATE = 10  # average 10 tasks per timestep
    
    # ===== TASK GENERATION CONFIG (Table 3) =====
    # Class 1: Delay-sensitive tasks
    TASK_CLASS1_DATA_SIZE_MIN = 10e3   # 10 KB
    TASK_CLASS1_DATA_SIZE_MAX = 40e3   # 40 KB
    TASK_CLASS1_CYCLES_MULTIPLIER = 8   # phi_i = 8 * D_i (from Table 3)
    TASK_CLASS1_DEADLINE_FACTOR = 0.5e-3  # T_req = 0.5 ms * D_i (most strict)
    
    # Class 2: Energy-sensitive tasks
    TASK_CLASS2_DATA_SIZE_MIN = 20e3   # 20 KB
    TASK_CLASS2_DATA_SIZE_MAX = 50e3   # 50 KB
    TASK_CLASS2_CYCLES_MULTIPLIER = 8   # phi_i = 8 * D_i
    TASK_CLASS2_DEADLINE_FACTOR = 1e-3  # T_req = 1 ms * D_i (moderate)
    
    # Class 3: Insensitive tasks
    TASK_CLASS3_DATA_SIZE_MIN = 200e3  # 200 KB
    TASK_CLASS3_DATA_SIZE_MAX = 400e3  # 400 KB
    TASK_CLASS3_CYCLES_MULTIPLIER = 8   # phi_i = 8 * D_i
    TASK_CLASS3_DEADLINE_FACTOR = 2e-3  # T_req = 2 ms * D_i (relaxed)

    # ===== ACTOR-CRITIC HYPERPARAMETERS (from paper) =====
    # Network architecture: 2 hidden layers, 256 neurons each, TanH activation
    AC_HIDDEN_SIZE = 256
    AC_NUM_HIDDEN_LAYERS = 2
    
    # Learning rates (from paper)
    AC_LEARNING_RATE_ACTOR = 1e-5   # alpha_actor = 10^-5
    AC_LEARNING_RATE_CRITIC = 1e-4  # alpha_critic = 10^-4
    
    # Discount factor (standard value, paper doesn't specify)
    AC_GAMMA = 0.99
    
    # Entropy coefficient for exploration (not in paper, but standard)
    AC_ENTROPY_COEFF = 0.01


