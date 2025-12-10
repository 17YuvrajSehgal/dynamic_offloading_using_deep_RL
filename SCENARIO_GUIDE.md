# Scenario Simulation Guide

This guide explains how to run all scenarios with unified parameters and generate publication-quality plots.

## 🚀 Quick Start

### Run ALL Scenario 1 variants (recommended)

```bash
# Submit to cluster - runs all 4 variants automatically
sbatch run_rl_nibi.slurm

# Or run locally
python run_all_scenarios.py --scenario-set s1 --episodes 500 --all
```

This will:
1. ✅ Train RL agents on all 4 Scenario 1 variants (Class 1 90%, Class 2 90%, Class 3 90%, Random)
2. ✅ Evaluate each trained agent
3. ✅ Generate Figure 7 style plots automatically
4. ✅ Save all results to `results/scenarios/`

---

## 📊 Generated Plots

The system automatically generates:

### **`scenario_1_complete_figure.png`**
A 4×3 grid matching Figure 7 from the paper:
- **Rows**: Class 1, Class 2, Class 3, Random distributions
- **Columns**: QoE, Battery, Decisions
- **Legend**: Shows all baselines + RL agent performance

Example structure:
```
              QoE           Battery        Decisions
Class 1     [plot a]      [plot b]       [plot c]
Class 2     [plot d]      [plot e]       [plot f]
Class 3     [plot g]      [plot h]       [plot i]
Random      [plot j]      [plot k]       [plot l]
```

---

## 🔧 Configuration

### Edit `run_rl_nibi.slurm` (lines 22-30)

```bash
# Scenario set to run
SCENARIO_SET="s1"        # Options: s1, s2, s1_base, all

# Training parameters
EPISODES=500             # Episodes per scenario (paper uses 500)

# Actions to perform
DO_TRAIN=true            # Train agents
DO_EVAL=true             # Evaluate agents
DO_PLOT=true             # Generate plots
```

### Available Scenario Sets

| Set | Scenarios | Description |
|-----|-----------|-------------|
| `s1` | s1_class1_90, s1_class2_90, s1_class3_90, s1_random | **All Scenario 1 variants** (recommended) |
| `s1_base` | s1_base | Base scenario with equal distribution |
| `s2` | s2_base | Scenario 2 (channel degradation) |
| `all` | All above | Every scenario |

---

## 💻 Command-Line Usage

### 1. Run All Scenarios (Full Pipeline)

```bash
python run_all_scenarios.py --scenario-set s1 --episodes 500 --all
```

Equivalent to:
```bash
python run_all_scenarios.py --scenario-set s1 --episodes 500 --train --eval --plot
```

### 2. Run Specific Scenarios

```bash
# Only Class 1 and Class 2
python run_all_scenarios.py \
    --scenarios s1_class1_90 s1_class2_90 \
    --episodes 300 --all
```

### 3. Evaluate Existing Models (No Training)

```bash
python run_all_scenarios.py --scenario-set s1 --eval --plot
```

### 4. Quick Test (Fewer Episodes)

```bash
python run_all_scenarios.py --scenario-set s1 --episodes 50 --all
```

### 5. Training Only (No Evaluation)

```bash
python run_all_scenarios.py --scenario-set s1 --episodes 500 --train
```

### 6. Generate Plots from Existing Results

```bash
python plot_scenario_results.py --scenario s1 --results-dir results/scenarios
```

---

## 💾 Output Structure

After running, you'll have:

```
results/scenarios/
├── s1_class1_90/
│   ├── actor.pt                      # Trained actor network
│   ├── critic.pt                     # Trained critic network
│   └── rl_agent_metrics.csv          # Evaluation results (timestep, QoE, Battery, etc.)
├── s1_class2_90/
│   ├── actor.pt
│   ├── critic.pt
│   └── rl_agent_metrics.csv
├── s1_class3_90/
│   ├── actor.pt
│   ├── critic.pt
│   └── rl_agent_metrics.csv
├── s1_random/
│   ├── actor.pt
│   ├── critic.pt
│   └── rl_agent_metrics.csv
└── scenario_1_complete_figure.png  # 📊 Publication-ready plot
```

---

## 📋 Results Summary

At the end of execution, you'll see:

```
================================================================================
ALL SCENARIOS COMPLETE
================================================================================
Total time: 127.3 minutes

Results Summary:
--------------------------------------------------------------------------------
Scenario             Mean QoE  Final Battery  Success Rate
--------------------------------------------------------------------------------
s1_class1_90        -0.023456       2347.82 J         87.3%
s1_class2_90        -0.018932       2891.45 J         92.1%
s1_class3_90        -0.012345       1456.23 J         94.8%
s1_random           -0.019876       2234.56 J         89.5%
--------------------------------------------------------------------------------

Results saved to: /path/to/results/scenarios/
================================================================================
```

---

## 🔍 Monitoring Your Job

### Check Job Status
```bash
squeue -u $USER
```

### View Real-Time Output
```bash
tail -f logs/rl_all_<JOBID>.out
```

### Check for Errors
```bash
tail -f logs/rl_all_<JOBID>.err
```

### Monitor GPU Usage (if running)
```bash
ssh <node_name>
nvidia-smi -l 1  # Updates every 1 second
```

---

## ⏱️ Time Estimates

| Configuration | Time per Scenario | Total Time (4 scenarios) |
|---------------|-------------------|-------------------------|
| 50 episodes (testing) | ~5 minutes | ~20 minutes |
| 100 episodes | ~10 minutes | ~40 minutes |
| 300 episodes | ~30 minutes | ~2 hours |
| 500 episodes (paper) | ~45 minutes | ~3 hours |
| 1000 episodes | ~90 minutes | ~6 hours |

**Note**: Times are approximate and depend on GPU type.

---

## 🚫 Troubleshooting

### Problem: "No such file or directory: run_all_scenarios.py"

**Solution**: Make sure you're in the project root directory:
```bash
cd /path/to/dynamic_offloading_using_deep_RL
ls run_all_scenarios.py  # Should exist
```

### Problem: "ImportError: No module named 'scenario_config'"

**Solution**: Activate your virtual environment:
```bash
source .venv/bin/activate
```

### Problem: "CUDA out of memory"

**Solution**: Reduce batch size or use CPU:
```bash
python run_all_scenarios.py --scenario-set s1 --episodes 500 --all --device cpu
```

### Problem: "Training is slow"

**Solution**: Check GPU is being used:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))
```

### Problem: "Plots not generated"

**Solution**: Generate manually:
```bash
python plot_scenario_results.py --scenario s1 --results-dir results/scenarios
```

---

## 📖 Key Scripts

| Script | Purpose |
|--------|--------|
| `run_all_scenarios.py` | **Master script** - runs all scenarios with unified parameters |
| `run_scenario.py` | Single scenario runner (used internally by `run_all_scenarios.py`) |
| `plot_scenario_results.py` | Generate publication-quality plots |
| `run_rl_nibi.slurm` | SLURM submission script for cluster |
| `scenario_config.py` | Scenario definitions (MEC availability, task distributions) |
| `rl_env.py` | RL environment with scenario support |

---

## 🎯 Next Steps

1. **Run all scenarios**:
   ```bash
   sbatch run_rl_nibi.slurm
   ```

2. **Wait for completion** (~3 hours for 500 episodes)

3. **Check results**:
   ```bash
   ls results/scenarios/
   open results/scenarios/scenario_1_complete_figure.png
   ```

4. **Analyze metrics**:
   ```python
   import pandas as pd
   df = pd.read_csv('results/scenarios/s1_class1_90/rl_agent_metrics.csv')
   print(df[['QoE', 'Battery', 'Success']].describe())
   ```

5. **Compare with paper's Figure 7** ✅

---

## 📧 Questions?

If you encounter issues:
1. Check the logs in `logs/rl_all_<JOBID>.out`
2. Verify environment setup with `python --version` and `pip list`
3. Test with fewer episodes first: `--episodes 10`

Happy experimenting! 🚀
