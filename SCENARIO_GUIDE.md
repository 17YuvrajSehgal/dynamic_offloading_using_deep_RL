# Scenario Simulation Guide

This guide explains how to run all scenarios with unified parameters, including **baseline comparisons**, and generate publication-quality plots matching the paper's Figure 7.

## 🚀 Quick Start

### Run ALL Scenario 1 variants with baselines (recommended)

```bash
# Submit to cluster - runs all 4 variants + baselines automatically
sbatch run_rl_nibi.slurm

# Or run locally
python run_all_scenarios.py --scenario-set s1 --episodes 500 --all
```

This will:
1. ✅ Run **baselines** (local, MEC, cloud, random) on all 4 scenarios
2. ✅ Train **RL agents** on all 4 Scenario 1 variants (Class 1 90%, Class 2 90%, Class 3 90%, Random)
3. ✅ Evaluate each trained RL agent
4. ✅ Generate **Figure 7 style plots** with all comparisons automatically
5. ✅ Save all results to `results/scenarios/`

---

## 📊 Generated Plots

The system automatically generates:

### **`scenario_1_complete_figure.png`**
A 4×3 grid matching Figure 7 from the paper:
- **Rows**: Class 1, Class 2, Class 3, Random distributions
- **Columns**: QoE, Battery, Decisions
- **Methods compared**: Local, MEC, Cloud, Random, **RL Agent** (SeparatedNetworks AC)
- **Visible features**: MEC unavailability periods, battery depletion, decision patterns

Example structure:
```
              QoE           Battery        Decisions
Class 1     [plot a]      [plot b]       [plot c]
Class 2     [plot d]      [plot e]       [plot f]
Class 3     [plot g]      [plot h]       [plot i]
Random      [plot j]      [plot k]       [plot l]

Legend: — Join-Networks AC  — DDDQN  — cloud  — mec  — random  — local  — SeparatedNetworks AC
```

---

## 🔧 Configuration

### Edit `run_rl_nibi.slurm` (lines 22-32)

```bash
# Scenario set to run
SCENARIO_SET="s1"        # Options: s1, s2, s1_base, all

# Training parameters
EPISODES=500             # Episodes per scenario (paper uses 500)

# Actions to perform
DO_TRAIN=true            # Train RL agents
DO_EVAL=true             # Evaluate RL agents
DO_PLOT=true             # Generate plots
# Note: Baselines are ALWAYS run when using --all
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

### 1. Run Everything (Baselines + RL + Plots)

```bash
python run_all_scenarios.py --scenario-set s1 --episodes 500 --all
```

This runs:
- ✅ Local, MEC, Cloud, Random baselines
- ✅ RL training (500 episodes per scenario)
- ✅ RL evaluation
- ✅ Plot generation

### 2. Run Only Baselines (No RL)

```bash
python run_all_scenarios.py --scenario-set s1 --baselines --plot
```

### 3. Run Only RL (Skip Baselines)

```bash
python run_all_scenarios.py --scenario-set s1 --episodes 500 --train --eval --plot --no-baselines
```

### 4. Run Specific Baseline on One Scenario

```bash
# Run only MEC baseline on Class 1 scenario
python run_baselines_scenario.py --scenario s1_class1_90 --policy mec

# Run all baselines on Class 2 scenario
python run_baselines_scenario.py --scenario s1_class2_90 --all
```

### 5. Quick Test (Fewer Episodes)

```bash
python run_all_scenarios.py --scenario-set s1 --episodes 50 --all
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
│   ├── actor.pt                      # RL: Trained actor network
│   ├── critic.pt                     # RL: Trained critic network
│   ├── rl_agent_metrics.csv          # RL evaluation results
│   ├── local_metrics.csv             # Local baseline results
│   ├── mec_metrics.csv               # MEC baseline results
│   ├── cloud_metrics.csv             # Cloud baseline results
│   └── random_metrics.csv            # Random baseline results
├── s1_class2_90/
│   ├── actor.pt
│   ├── critic.pt
│   ├── rl_agent_metrics.csv
│   ├── local_metrics.csv
│   ├── mec_metrics.csv
│   ├── cloud_metrics.csv
│   └── random_metrics.csv
├── s1_class3_90/
│   ├── ... (same structure)
├── s1_random/
│   ├── ... (same structure)
└── scenario_1_complete_figure.png  # 📊 Publication-ready Figure 7!
```

Each CSV file contains:
- `QoE`: Quality of Experience per task
- `Latency`: Task execution latency
- `Energy`: Energy consumed
- `Battery`: Remaining battery
- `Action`: Decision made (0=local, 1=MEC, 2=cloud)
- `TaskClass`: Task class (1, 2, or 3)
- `Success`: Whether task met deadline

---

## 📋 Results Summary

At the end of execution, you'll see:

```
================================================================================
ALL SCENARIOS COMPLETE
================================================================================
Total time: 187.5 minutes

RL Agent Results Summary:
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

**Plus** individual summaries for each baseline policy!

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
| Baselines only | ~2 minutes | ~8 minutes |
| 50 episodes + baselines | ~7 minutes | ~28 minutes |
| 100 episodes + baselines | ~12 minutes | ~48 minutes |
| 300 episodes + baselines | ~32 minutes | ~2.2 hours |
| 500 episodes + baselines (paper) | ~47 minutes | ~3.2 hours |
| 1000 episodes + baselines | ~92 minutes | ~6.2 hours |

**Note**: Times are approximate and depend on GPU type.

---

## 🎯 Comparing with Paper Results

Your generated plots will show:

### **QoE Plots (Column 1)**
- ✅ Multiple colored lines for each method (local, MEC, cloud, random, RL)
- ✅ Clear performance differences
- ✅ Impact of MEC unavailability (visible as QoE drops)
- ✅ RL agent adaptation over time

### **Battery Plots (Column 2)**
- ✅ Battery depletion curves for each method
- ✅ Markers showing timestep intervals
- ✅ Different slopes showing energy efficiency
- ✅ Early depletion for energy-intensive methods (e.g., local for Class 3)

### **Decision Plots (Column 3)**
- ✅ Color-coded heatmap showing decisions over time
- ✅ Blue = Local, Orange = MEC, Green = Cloud, Red = Dead
- ✅ Visible adaptation during MEC unavailability periods
- ✅ Multiple UE visualization (simulated from single-agent decisions)

---

## 🚫 Troubleshooting

### Problem: "No such file or directory: run_all_scenarios.py"

**Solution**: Make sure you're in the project root directory:
```bash
cd /path/to/dynamic_offloading_using_deep_RL
ls run_all_scenarios.py  # Should exist
```

### Problem: "Baseline results missing in plots"

**Solution**: Run baselines explicitly:
```bash
python run_all_scenarios.py --scenario-set s1 --baselines
```

Then regenerate plots:
```bash
python plot_scenario_results.py --scenario s1
```

### Problem: "ImportError: No module named 'scenario_config'"

**Solution**: Activate your virtual environment:
```bash
source .venv/bin/activate
```

### Problem: "CUDA out of memory"

**Solution**: Use CPU for training:
```bash
python run_all_scenarios.py --scenario-set s1 --episodes 500 --all --device cpu
```

### Problem: "Baselines are too slow"

**Solution**: Baselines are fast (~2 min per scenario). If slow, check:
```bash
# Verify scenario timesteps
python -c "from scenario_config import get_scenario; print(get_scenario('s1_class1_90').total_timesteps)"
```

---

## 📖 Key Scripts

| Script | Purpose |
|--------|--------|
| `run_all_scenarios.py` | **Master script** - runs RL + baselines + plots for all scenarios |
| `run_baselines_scenario.py` | Run individual baseline policies on scenarios |
| `run_scenario.py` | Single RL scenario runner (used internally) |
| `plot_scenario_results.py` | Generate publication-quality plots |
| `run_rl_nibi.slurm` | SLURM submission script for cluster |
| `scenario_config.py` | Scenario definitions (MEC availability, task distributions) |
| `rl_env.py` | RL environment with scenario support |

---

## 🎯 Next Steps

1. **Run everything**:
   ```bash
   sbatch run_rl_nibi.slurm
   ```

2. **Wait for completion** (~3.2 hours for 500 episodes + baselines)

3. **Check results**:
   ```bash
   ls results/scenarios/s1_class1_90/
   # Should see: rl_agent_metrics.csv, local_metrics.csv, mec_metrics.csv, etc.
   ```

4. **View plots**:
   ```bash
   open results/scenarios/scenario_1_complete_figure.png
   ```

5. **Analyze and compare with paper's Figure 7** ✅

6. **Generate custom analysis**:
   ```python
   import pandas as pd
   
   # Load RL agent results
   rl = pd.read_csv('results/scenarios/s1_class1_90/rl_agent_metrics.csv')
   
   # Load baseline results
   local = pd.read_csv('results/scenarios/s1_class1_90/local_metrics.csv')
   mec = pd.read_csv('results/scenarios/s1_class1_90/mec_metrics.csv')
   cloud = pd.read_csv('results/scenarios/s1_class1_90/cloud_metrics.csv')
   
   # Compare mean QoE
   print(f"RL:    {rl['QoE'].mean():.6f}")
   print(f"Local: {local['QoE'].mean():.6f}")
   print(f"MEC:   {mec['QoE'].mean():.6f}")
   print(f"Cloud: {cloud['QoE'].mean():.6f}")
   ```

---

## 📧 Questions?

If you encounter issues:
1. Check the logs in `logs/rl_all_<JOBID>.out`
2. Verify environment setup with `python --version` and `pip list`
3. Test with fewer episodes first: `--episodes 10`
4. Run baselines separately to debug: `python run_baselines_scenario.py --scenario s1_class1_90 --all`

Happy experimenting! 🚀
