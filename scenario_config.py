"""
scenario_config.py

Defines the scenarios from the paper:
- Scenario 1: MEC unavailable for certain time periods
- Scenario 2: Channel quality degradation

Each scenario specifies:
- MEC availability schedule
- Channel quality modifiers
- Task distribution (% of each class)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario."""
    name: str
    description: str
    
    # MEC availability: list of (start_timestep, end_timestep, availability_percent)
    # availability_percent: 0-100, where 80-100 means "available"
    mec_availability_schedule: List[Tuple[int, int, float]]
    
    # Channel quality modifier: list of (start_timestep, end_timestep, multiplier)
    # multiplier: 1.0 = normal, < 1.0 = degraded, > 1.0 = enhanced
    channel_quality_schedule: List[Tuple[int, int, float]]
    
    # Task class distribution: [prob_class1, prob_class2, prob_class3]
    # Must sum to 1.0
    task_distribution: List[float]
    
    # Total simulation timesteps
    total_timesteps: int = 2000
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate task distribution
        if abs(sum(self.task_distribution) - 1.0) > 1e-6:
            raise ValueError(f"Task distribution must sum to 1.0, got {sum(self.task_distribution)}")
        
        if len(self.task_distribution) != 3:
            raise ValueError(f"Task distribution must have exactly 3 values, got {len(self.task_distribution)}")
    
    def get_mec_availability(self, timestep: int) -> float:
        """Get MEC availability percentage at given timestep."""
        for start, end, availability in self.mec_availability_schedule:
            if start <= timestep < end:
                return availability
        # Default: fully available
        return 100.0
    
    def is_mec_available(self, timestep: int, threshold: float = 80.0) -> bool:
        """Check if MEC is considered 'available' at given timestep."""
        return self.get_mec_availability(timestep) >= threshold
    
    def get_channel_quality_multiplier(self, timestep: int) -> float:
        """Get channel quality multiplier at given timestep."""
        for start, end, multiplier in self.channel_quality_schedule:
            if start <= timestep < end:
                return multiplier
        # Default: normal quality
        return 1.0
    
    def sample_task_class(self) -> int:
        """Sample a task class (1, 2, or 3) according to distribution."""
        return int(np.random.choice([1, 2, 3], p=self.task_distribution))


# ============================================================================
# SCENARIO 1: MEC Unavailable for Certain Time Periods
# ============================================================================
# From paper: "the MEC server is available during a big part of the execution 
# time, considering it is available when 80%-100% of its computing resources 
# are free to use. It is available between timesteps 0-500, 750-1250 and 
# 1500-2000. However, there is an important service drop between timesteps 
# 500-750 and 1250-1500, resulting in an unavailable MEC server."

SCENARIO_1_BASE = ScenarioConfig(
    name="Scenario 1 - Base",
    description="MEC unavailable during timesteps 500-750 and 1250-1500",
    mec_availability_schedule=[
        (0, 500, 100.0),      # Available: 0-500
        (500, 750, 0.0),      # UNAVAILABLE: 500-750
        (750, 1250, 100.0),   # Available: 750-1250
        (1250, 1500, 0.0),    # UNAVAILABLE: 1250-1500
        (1500, 2000, 100.0),  # Available: 1500-2000
    ],
    channel_quality_schedule=[
        (0, 2000, 1.0),  # Normal channel quality throughout
    ],
    task_distribution=[0.33, 0.34, 0.33],  # Equal distribution (default)
    total_timesteps=2000,
)

# Scenario 1 - Distribution A: 90% Class 1 (delay-sensitive)
SCENARIO_1_CLASS1_90 = ScenarioConfig(
    name="Scenario 1 - 90% Class 1",
    description="MEC unavailable 500-750, 1250-1500; 90% delay-sensitive tasks",
    mec_availability_schedule=SCENARIO_1_BASE.mec_availability_schedule,
    channel_quality_schedule=SCENARIO_1_BASE.channel_quality_schedule,
    task_distribution=[0.90, 0.05, 0.05],  # 90% Class 1, 5% Class 2, 5% Class 3
    total_timesteps=2000,
)

# Scenario 1 - Distribution B: 90% Class 2 (energy-sensitive)
SCENARIO_1_CLASS2_90 = ScenarioConfig(
    name="Scenario 1 - 90% Class 2",
    description="MEC unavailable 500-750, 1250-1500; 90% energy-sensitive tasks",
    mec_availability_schedule=SCENARIO_1_BASE.mec_availability_schedule,
    channel_quality_schedule=SCENARIO_1_BASE.channel_quality_schedule,
    task_distribution=[0.05, 0.90, 0.05],  # 5% Class 1, 90% Class 2, 5% Class 3
    total_timesteps=2000,
)

# Scenario 1 - Distribution C: 90% Class 3 (insensitive)
SCENARIO_1_CLASS3_90 = ScenarioConfig(
    name="Scenario 1 - 90% Class 3",
    description="MEC unavailable 500-750, 1250-1500; 90% insensitive tasks",
    mec_availability_schedule=SCENARIO_1_BASE.mec_availability_schedule,
    channel_quality_schedule=SCENARIO_1_BASE.channel_quality_schedule,
    task_distribution=[0.05, 0.05, 0.90],  # 5% Class 1, 5% Class 2, 90% Class 3
    total_timesteps=2000,
)

# Scenario 1 - Distribution D: Random/Equal distribution
SCENARIO_1_RANDOM = ScenarioConfig(
    name="Scenario 1 - Random Distribution",
    description="MEC unavailable 500-750, 1250-1500; equal task distribution",
    mec_availability_schedule=SCENARIO_1_BASE.mec_availability_schedule,
    channel_quality_schedule=SCENARIO_1_BASE.channel_quality_schedule,
    task_distribution=[0.33, 0.34, 0.33],  # Approximately equal
    total_timesteps=2000,
)


# ============================================================================
# SCENARIO 2: Channel Quality Degradation
# ============================================================================
# From paper: "In this scenario, the MEC server is always available... 
# the wireless channel is unstable during a certain period of time"
# (Details would come from paper's description of Scenario 2)

SCENARIO_2_BASE = ScenarioConfig(
    name="Scenario 2 - Base",
    description="MEC always available, channel degradation at certain periods",
    mec_availability_schedule=[
        (0, 2000, 100.0),  # Always available
    ],
    channel_quality_schedule=[
        # Placeholder: adjust based on paper's Figure 8
        (0, 500, 1.0),        # Normal: 0-500
        (500, 750, 0.3),      # DEGRADED: 500-750 (30% of normal)
        (750, 1250, 1.0),     # Normal: 750-1250
        (1250, 1500, 0.3),    # DEGRADED: 1250-1500
        (1500, 2000, 1.0),    # Normal: 1500-2000
    ],
    task_distribution=[0.33, 0.34, 0.33],
    total_timesteps=2000,
)

# Add more Scenario 2 variants as needed (90% Class 1, 90% Class 2, etc.)


# ============================================================================
# Scenario Registry
# ============================================================================

ALL_SCENARIOS = {
    # Scenario 1 variants
    "s1_base": SCENARIO_1_BASE,
    "s1_class1_90": SCENARIO_1_CLASS1_90,
    "s1_class2_90": SCENARIO_1_CLASS2_90,
    "s1_class3_90": SCENARIO_1_CLASS3_90,
    "s1_random": SCENARIO_1_RANDOM,
    
    # Scenario 2 variants
    "s2_base": SCENARIO_2_BASE,
}


def get_scenario(scenario_key: str) -> ScenarioConfig:
    """Get scenario configuration by key."""
    if scenario_key not in ALL_SCENARIOS:
        raise ValueError(
            f"Unknown scenario: {scenario_key}. "
            f"Available: {list(ALL_SCENARIOS.keys())}"
        )
    return ALL_SCENARIOS[scenario_key]


def list_scenarios() -> None:
    """Print all available scenarios."""
    print("\nAvailable Scenarios:")
    print("=" * 80)
    for key, config in ALL_SCENARIOS.items():
        print(f"\n[{key}]")
        print(f"  Name: {config.name}")
        print(f"  Description: {config.description}")
        print(f"  Task Distribution: Class1={config.task_distribution[0]:.0%}, "
              f"Class2={config.task_distribution[1]:.0%}, "
              f"Class3={config.task_distribution[2]:.0%}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test scenarios
    list_scenarios()
    
    # Test Scenario 1
    print("\n\nTesting Scenario 1 - Class 1 90%:")
    s1 = get_scenario("s1_class1_90")
    
    # Check MEC availability at key timesteps
    test_timesteps = [0, 250, 500, 625, 750, 1000, 1250, 1375, 1500, 1750]
    for t in test_timesteps:
        avail = s1.get_mec_availability(t)
        is_avail = s1.is_mec_available(t)
        print(f"  t={t:4d}: MEC availability = {avail:5.1f}%, available = {is_avail}")
    
    # Sample some task classes
    print("\n  Sampling 20 task classes:")
    samples = [s1.sample_task_class() for _ in range(20)]
    print(f"  {samples}")
    print(f"  Class 1: {samples.count(1)}/20, Class 2: {samples.count(2)}/20, Class 3: {samples.count(3)}/20")
