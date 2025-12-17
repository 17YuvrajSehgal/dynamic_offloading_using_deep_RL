"""
Script to run all scenarios with ONLY the greedy policy baseline.
Outputs CSV files compatible with plot_scenario_dashboard_bars.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import os
from pathlib import Path


class GreedyPolicyRunner:
    """
    Runs greedy policy across all scenarios.
    Outputs CSV files with columns: QoE, Latency, Battery, Success
    """

    def __init__(self, config: Dict):
        self.config = config
        self.results = {}

    def greedy_decision(self, state: Dict) -> int:
        """
        Greedy policy: Choose action that minimizes immediate cost

        Actions:
        0 - Local execution
        1 - Offload to MEC
        2 - Offload to Cloud

        Returns the action with lowest immediate energy/latency cost
        """
        battery = state['battery']
        mec_availability = state['mec_resources']
        cloud_availability = state['cloud_resources']
        channel_gain = state['channel_gain']
        task_class = state['task_class']
        task_size = state['task_size']
        latency_req = state['latency_req']

        # Calculate immediate cost for each action
        costs = {}

        # Local execution cost
        if battery > 100:  # Minimum battery threshold
            local_energy = task_size * 0.001  # Energy per KB
            local_latency = task_size * 0.0001  # Simple latency model
            costs[0] = local_energy / battery + (local_latency / latency_req if latency_req > 0 else 0)
        else:
            costs[0] = float('inf')

        # MEC offloading cost
        if mec_availability > 0.2 and channel_gain > -100:  # MEC available and channel OK
            mec_energy = task_size * 0.0005  # Lower energy for transmission
            mec_latency = task_size * 0.00005 + 0.01  # Transmission + processing delay
            costs[1] = mec_energy / battery + (mec_latency / latency_req if latency_req > 0 else 0)
        else:
            costs[1] = float('inf')

        # Cloud offloading cost
        if channel_gain > -100:  # Channel OK
            cloud_energy = task_size * 0.0005
            cloud_latency = task_size * 0.00005 + 0.05  # Higher latency than MEC
            # Penalize cloud for delay-sensitive tasks (class 1)
            latency_penalty = 3.0 if task_class == 1 else 1.0
            costs[2] = cloud_energy / battery + latency_penalty * (cloud_latency / latency_req if latency_req > 0 else 0)
        else:
            costs[2] = float('inf')

        # Return action with minimum cost
        return min(costs, key=costs.get)

    def run_scenario_case(self, scenario: str, case: str) -> pd.DataFrame:
        """
        Run greedy policy for a specific scenario-case combination

        Args:
            scenario: 's1' or 's2'
            case: 'class1_90', 'class2_90', 'class3_90', or 'random'

        Returns:
            DataFrame with columns: Timestep, QoE, Latency, Battery, Success
        """
        print(f"Running {scenario}_{case} with greedy policy...")

        num_timesteps = 2000
        num_ues = 20

        # Initialize metrics storage
        timestep_data = []

        # Initialize UE states
        ue_states = [{'battery': 4000.0, 'id': i} for i in range(num_ues)]

        for t in range(num_timesteps):
            # Get environment state based on scenario
            env_state = self._get_environment_state(scenario, t)

            timestep_qoe = []
            timestep_latency = []
            timestep_success = []

            for ue_id in range(num_ues):
                # Generate task based on case
                task = self._generate_task(case)

                # Get current state for this UE
                state = self._get_ue_state(ue_states[ue_id], env_state, task)

                # Make greedy decision
                action = self.greedy_decision(state)

                # Execute action and get metrics
                qoe, latency, success, energy_consumed = self._execute_action(
                    state, action, ue_states[ue_id]['battery']
                )

                # Update UE battery
                ue_states[ue_id]['battery'] = max(0, ue_states[ue_id]['battery'] - energy_consumed - 0.1)

                # Record metrics
                timestep_qoe.append(qoe)
                timestep_latency.append(latency)
                timestep_success.append(success)

            # Average battery across all UEs at this timestep
            avg_battery = np.mean([ue['battery'] for ue in ue_states])

            # Aggregate timestep metrics
            timestep_data.append({
                'Timestep': t,
                'QoE': np.mean(timestep_qoe),
                'Latency': np.mean(timestep_latency),
                'Battery': avg_battery,
                'Success': np.mean(timestep_success)
            })

        # Create DataFrame
        df = pd.DataFrame(timestep_data)

        print(f"  Completed: Avg QoE={df['QoE'].mean():.4f}, Final Battery={df['Battery'].iloc[-1]:.2f}, Success Rate={df['Success'].mean():.2%}")

        return df

    def _get_environment_state(self, scenario: str, timestep: int) -> Dict:
        """Get environment state based on scenario and timestep"""
        if scenario == 's1':
            # Scenario 1: MEC unavailable at certain times
            if (500 <= timestep < 750) or (1250 <= timestep < 1500):
                mec_resources = 0.0
            else:
                mec_resources = np.random.uniform(0.7, 1.0)
            channel_gain = np.random.normal(0, 5.9)  # Shadow fading

        else:  # s2
            # Scenario 2: Communications failure
            if 500 <= timestep < 1000:
                channel_gain = -200  # Very poor channel
                mec_resources = 0.0
            else:
                channel_gain = np.random.normal(0, 5.9)
                mec_resources = np.random.uniform(0.7, 1.0)

        return {
            'mec_resources': mec_resources,
            'cloud_resources': 1.0,  # Cloud always available (high resources)
            'channel_gain': channel_gain
        }

    def _generate_task(self, case: str) -> Dict:
        """Generate task based on case distribution"""
        # Determine task distribution
        if case == 'class1_90':
            class_probs = [0.9, 0.05, 0.05]
        elif case == 'class2_90':
            class_probs = [0.05, 0.9, 0.05]
        elif case == 'class3_90':
            class_probs = [0.05, 0.05, 0.9]
        else:  # random
            class_probs = [1/3, 1/3, 1/3]

        task_class = np.random.choice([1, 2, 3], p=class_probs)

        # Task parameters based on class (from paper)
        if task_class == 1:
            # Delay-sensitive: small size, strict latency
            data_size = np.random.uniform(10, 40) * 1000  # KB -> bytes
            latency_req = 0.5 * 1e-3 * data_size / 1000  # Strict requirement
        elif task_class == 2:
            # Energy-sensitive: medium size, moderate latency
            data_size = np.random.uniform(20, 50) * 1000
            latency_req = 1e-3 * data_size / 1000
        else:  # class 3
            # Insensitive: large size, relaxed latency
            data_size = np.random.uniform(200, 400) * 1000
            latency_req = 2 * 1e-3 * data_size / 1000

        return {
            'class': task_class,
            'size': data_size / 1000,  # Convert to KB for calculations
            'latency_req': latency_req,
            'cpu_cycles': data_size * 8
        }

    def _get_ue_state(self, ue_state: Dict, env_state: Dict, task: Dict) -> Dict:
        """Combine UE, environment, and task into state"""
        return {
            'battery': ue_state['battery'],
            'mec_resources': env_state['mec_resources'],
            'cloud_resources': env_state['cloud_resources'],
            'channel_gain': env_state['channel_gain'],
            'task_class': task['class'],
            'task_size': task['size'],
            'latency_req': task['latency_req']
        }

    def _execute_action(self, state: Dict, action: int, current_battery: float) -> tuple:
        """
        Execute action and return (qoe, latency, success, energy_consumed)
        """
        task_size = state['task_size']
        latency_req = state['latency_req']

        if action == 0:  # Local execution
            energy_consumed = task_size * 0.001  # Higher local energy
            latency = task_size * 0.0001
            # Success depends on battery and latency
            success = (current_battery > energy_consumed) and (latency <= latency_req)

        elif action == 1:  # MEC offloading
            energy_consumed = task_size * 0.0005  # Lower transmission energy
            latency = task_size * 0.00005 + 0.01
            # Success depends on MEC availability and channel
            success = (state['mec_resources'] > 0.2) and (state['channel_gain'] > -100) and (latency <= latency_req)

        else:  # Cloud offloading
            energy_consumed = task_size * 0.0005
            latency = task_size * 0.00005 + 0.05  # Higher cloud latency
            # Success depends on channel and latency requirements
            success = (state['channel_gain'] > -100) and (latency <= latency_req * 1.5)  # Cloud has more relaxed requirements

        # Calculate QoE based on paper's definition
        if success:
            # QoE is inversely proportional to energy consumed, normalized by battery
            qoe = 1.0 - (energy_consumed / (current_battery + 1))
        else:
            qoe = -1.0  # Failure penalty

        return (qoe, latency, 1.0 if success else 0.0, energy_consumed)

    def run_all_scenarios(self, output_dir: str = 'results/scenarios'):
        """Run greedy policy for all scenarios and cases"""

        scenarios = ['s1', 's2']
        cases = ['class1_90', 'class2_90', 'class3_90', 'random']

        print("="*60)
        print("Running Greedy Policy Baseline for All Scenarios")
        print("="*60)

        for scenario in scenarios:
            print(f"\n### SCENARIO {scenario.upper()} ###")

            for case in cases:
                # Run the scenario-case combination
                df = self.run_scenario_case(scenario, case)

                # Save CSV file
                case_dir = Path(output_dir) / f"{scenario}_{case}"
                case_dir.mkdir(parents=True, exist_ok=True)

                csv_path = case_dir / "greedy_metrics.csv"
                df.to_csv(csv_path, index=False)

                print(f"  ✅ Saved: {csv_path}")

        print("\n" + "="*60)
        print("Greedy Policy Baseline Complete!")
        print("="*60)
        print(f"\nResults saved to: {output_dir}/")
        print("\nYou can now run your plotting script:")
        print(f"  python plot_scenario_dashboard_bars.py --scenario s1 --results-dir {output_dir}")
        print(f"  python plot_scenario_dashboard_bars.py --scenario s2 --results-dir {output_dir}")


def main():
    """Main execution function"""

    # Configuration
    config = {
        'num_ues': 20,
        'num_timesteps': 2000,
        'battery_capacity': 4000.0
    }

    # Initialize runner
    runner = GreedyPolicyRunner(config)

    # Run all scenarios and save CSV files
    runner.run_all_scenarios(output_dir='results/scenarios')


if __name__ == "__main__":
    main()
