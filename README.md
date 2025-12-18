### Deep RL for Dynamic Task Offloading in the 5G Edge–Cloud Continuum

This repository contains a PyTorch implementation of the experiments from:

> **Nieto, G., de la Iglesia, I., Lopez-Novoa, U. et al.**  
> *Deep Reinforcement Learning techniques for dynamic task offloading in the 5G edge-cloud continuum.*  
> Journal of Cloud Computing 13, 94 (2024). https://doi.org/10.1186/s13677-024-00658-0

The code simulates a 5G edge–cloud system with a single user equipment (UE), a base station (BS), a MEC server, and a cloud server. A Deep Reinforcement Learning (DRL) agent learns when to execute tasks **locally**, offload to **MEC**, or offload to the **cloud**, under varying **MEC availability** and **communication failures**.

---

### Repository Structure

- **`offload_rl/`** – Core simulation and RL code
  - `EnvConfig.py` – System and experiment parameters (as in the paper’s tables)
  - `models.py` – UE, BS, MEC, Cloud and task generation models
  - `rl_env.py` – `OffloadEnv` RL environment with scenario support and QoE reward
  - `ac_agent.py` – Actor–Critic agent (SeparatedNetworks AC) used in the paper
  - `policy.py` – (Legacy) baseline policy definitions, kept for completeness
- **Scenario & experiment control**
  - `scenario_config.py` – Formal definitions of **Scenario 1** (MEC unavailable) and **Scenario 2** (communication failure), plus task-class distributions
  - `run_scenario.py` – Train/evaluate RL on a **single** scenario
  - `run_baselines_scenario.py` – Run baseline policies on a scenario
  - `run_all_scenarios.py` – Master script to run **all scenarios** (baselines + RL + plots)
  - `train_rl.py` – Stand-alone RL training script on the base environment (no scenario logic)
  - `run_rl_nibi*.slurm` – Example SLURM scripts for running on an HPC cluster
- **Plotting & analysis**
  - `plot_scenario_results.py` – Aggregate and plot Scenario 1 results (Figure-7-style)
  - `plot_scenario_2_results.py` – Aggregate and plot Scenario 2 results
  - `plot_scenario_dashboard_bars.py` – Extra summary dashboard/bar plots
  - `SCENARIO_GUIDE.md` – Detailed guide for running & plotting Scenario 1
  - `SCENARIO_2_README.md` – Scenario 2 description and analysis tips
- **Results & logs**
  - `results/scenarios/` – Output directory for per-scenario CSVs, trained models and figures
  - `logs/` – Cluster log files when running via SLURM
- **Other**
  - `requirements.txt` – Python dependencies
  - `Deep_Reinforcement_Learning_techniques_for_dynamic.pdf` – Paper PDF

---

### Main Idea & Learning Setup

- **Goal**: Learn a **dynamic offloading policy** that maximizes an energy-aware **Quality of Experience (QoE)** while respecting task deadlines and battery constraints.
- **Actions**:  
  - `0` – Execute task locally on the UE  
  - `1` – Offload to MEC  
  - `2` – Offload to cloud
- **Tasks** (three classes, as in the paper):
  - **Class 1** – Delay-sensitive, small data, strict deadlines
  - **Class 2** – Energy-sensitive, medium size, moderate deadlines
  - **Class 3** – Insensitive, large data, relaxed deadlines
- **Reward / QoE** (Equation 18 in the paper, implemented in `OffloadEnv`):
  - Successful task: $ \mathrm{QoE} = - \frac{E_{\mathrm{consumed}}}{B_n} $ (energy cost relative to current battery)
  - Failed task: fixed penalty $ \eta = -0.1 $

- **Agent**:
  - On-policy **Actor–Critic** with separated actor and critic networks
  - Two hidden layers, 256 units each, `Tanh` activations
  - Hyperparameters follow the paper (learning rates, discount factor, etc., see `EnvConfig.py`)

Baselines are **non-learning** policies:
- Always Local, Always MEC, Always Cloud
- Uniform Random
- Greedy-by-Size (offload if task size ≥ threshold)

---

### Experiment Scenarios

Scenarios are defined in `scenario_config.py` and implemented in both the RL environment (`offload_rl/rl_env.py`) and baseline simulator (`run_baselines_scenario.py`).

- **Scenario 1 – MEC Unavailability**
  - MEC server resources drop to **0%** in the intervals **[500, 750)** and **[1250, 1500)**; otherwise available (≥80% resources).
  - Communication is always available.
  - Variants (task-class distributions):
    - `s1_class1_90` – 90% delay-sensitive (Class 1)
    - `s1_class2_90` – 90% energy-sensitive (Class 2)
    - `s1_class3_90` – 90% insensitive (Class 3)
    - `s1_random` – ~equal mix of all three classes

- **Scenario 2 – Communication Failure**
  - MEC server is always available (80–100% resources).
  - Wireless link **completely disappears** between timesteps **500 and 1000** → UE cannot offload to MEC or cloud; only local execution is possible.
  - Variants mirror Scenario 1 distributions:
    - `s2_class1_90`, `s2_class2_90`, `s2_class3_90`, `s2_random`

Each scenario runs for **2000 timesteps** and matches the timelines and behavior described in the paper.

---

### Baselines vs. RL Agent

For each scenario and distribution, the following are evaluated:

- **Baselines** (via `run_baselines_scenario.py` or `run_all_scenarios.py`):
  - `local` – Always execute on the UE
  - `mec` – Always offload to MEC (subject to availability/communication)
  - `cloud` – Always offload to cloud
  - `random` – Random choice among local/MEC/cloud
  - `greedy_by_size` – Offload large tasks to MEC, process small tasks locally

- **RL Agent** (via `run_scenario.py` or `run_all_scenarios.py`):
  - Actor–Critic agent observes state features such as battery level, task properties, MEC availability, and communication status and outputs an offloading decision each step.

Metrics saved per timestep (for both baselines and RL):
- `QoE`, `Latency`, `Energy`, `Battery`, `Action`, `TaskClass`, `Success`, `OffloadRatio`

---

### Installation

- **Python version**: 3.9+ (recommended)

1. Create and activate a virtual environment (optional but recommended):

```bash
cd dynamic_offloading_using_deep_RL
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Check GPU availability:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

### How to Reproduce the Main Experiments

The easiest way to reproduce the paper-style experiments (including baselines and RL) is to use `run_all_scenarios.py`. All commands below assume you are in the project root.

#### 1. Reproduce Scenario 1 (MEC Unavailability) – All Variants

Runs **baselines + RL training + RL evaluation + plots** for:
`s1_class1_90`, `s1_class2_90`, `s1_class3_90`, `s1_random`.

```bash
python run_all_scenarios.py --scenario-set s1 --episodes 500 --all
```

Key outputs:
- Per-scenario directories in `results/scenarios/s1_*/`:
  - `actor.pt`, `critic.pt` – trained RL models
  - `{local,mec,cloud,random,greedy_by_size}_metrics.csv` – baseline metrics
  - `rl_agent_metrics.csv` – RL evaluation metrics
- Aggregated plot:
  - `results/scenarios/scenario_1_complete_figure.png` (Figure-7-style grid with QoE, battery, and decisions across all Scenario 1 variants).

#### 2. Reproduce Scenario 2 (Communication Failure) – All Variants

```bash
python run_all_scenarios.py --scenario-set s2 --episodes 500 --all
```

Outputs:
- Per-scenario results in `results/scenarios/s2_*/` (same structure as Scenario 1).
- Aggregated plot:
  - `results/scenarios/scenario_2_complete_figure.png`.

#### 3. Quick Sanity Run (Fewer Episodes)

To check everything works end-to-end without waiting for full training:

```bash
python run_all_scenarios.py --scenario-set s1 --episodes 50 --all
```

#### 4. Run Only Baselines (No RL)

Useful for quickly reproducing baseline curves:

```bash
python run_all_scenarios.py --scenario-set s1 --baselines --plot
```

or for a single scenario:

```bash
python run_baselines_scenario.py --scenario s1_class1_90 --all
```

#### 5. Train & Evaluate RL on a Single Scenario

Example: Scenario 1 with 90% Class 1 tasks.

```bash
# Train
python run_scenario.py --scenario s1_class1_90 --train --episodes 500

# Evaluate (after training)
python run_scenario.py --scenario s1_class1_90 --eval
```

---

### Plotting and Post-Processing

- **Generate Scenario 1 plots from existing CSVs**:

```bash
python plot_scenario_results.py --scenario s1 --results-dir results/scenarios
```

- **Generate Scenario 2 plots from existing CSVs**:

```bash
python plot_scenario_2_results.py --results-dir results/scenarios
```

The main publication-style figures are:
- `results/scenarios/scenario_1_complete_figure.png`
- `results/scenarios/scenario_2_complete_figure.png`

You can also open individual CSV files with pandas for custom analysis, e.g.:

```python
import pandas as pd

rl = pd.read_csv("results/scenarios/s1_class1_90/rl_agent_metrics.csv")
local = pd.read_csv("results/scenarios/s1_class1_90/local_metrics.csv")

print("RL mean QoE:", rl["QoE"].mean())
print("Local mean QoE:", local["QoE"].mean())
```

---

### Cluster / SLURM Usage (Optional)

On an HPC cluster with SLURM, you can use the provided scripts as templates:

```bash
sbatch run_rl_nibi.slurm
```

Adjust, for example, the scenario set and number of episodes in `run_rl_nibi.slurm`:

```bash
SCENARIO_SET="s1"     # s1, s2, s1_base, all
EPISODES=500          # as in the paper
```

Monitor job logs under `logs/` and GPU status via standard cluster tools.

---

### How to Cite

If you use this code or reproduce these experiments in academic work, please cite the original paper:

> Nieto, G., de la Iglesia, I., Lopez-Novoa, U. et al. **Deep Reinforcement Learning techniques for dynamic task offloading in the 5G edge-cloud continuum.** *J Cloud Comp* 13, 94 (2024). https://doi.org/10.1186/s13677-024-00658-0


