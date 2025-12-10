# Scenario 2: Communication Failure

This document explains **Scenario 2** from the paper, where the wireless communication link **completely disappears** between timesteps 500-1000, making offloading impossible.

---

## 📚 Overview

### **What is Scenario 2?**

**Scenario 2** is characterized by:
- ✅ **Stable MEC server**: Always has 80-100% of computational resources available
- ❌ **Communication failure**: Wireless link **completely disappears** between tasks 500-1000
- ⚠️ **Consequence**: UEs cannot reach MEC or Cloud (only accessible through BS)
- 📍 **Only option during failure**: Local execution

### **Key Difference from Scenario 1**

| Aspect | Scenario 1 | Scenario 2 |
|--------|-----------|------------|
| **MEC Availability** | Drops to 0% (500-750, 1250-1500) | Always 80-100% |
| **Communication** | Always available | **Fails completely** (500-1000) |
| **Problem** | MEC server overloaded | Wireless link lost |
| **Impact** | Can't offload to MEC only | Can't offload to MEC **OR** Cloud |

---

## 🔍 Scenario Configuration

### **Timeline**

```
Timestep:     0        500        1000       2000
              |---------|----------|----------|
MEC:          Available Available Available
Communication: OK     FAILURE      OK
```

### **Communication Failure Period: 500-1000**
- Channel quality multiplier = **0.0** (complete loss)
- MEC offloading: **FAILS** (can't reach MEC)
- Cloud offloading: **FAILS** (can't reach Cloud)
- Local execution: **ONLY OPTION** (always works)

### **Task Distributions**

Like Scenario 1, we test 4 distributions:

1. **`s2_class1_90`**: 90% delay-sensitive (Class 1) tasks
2. **`s2_class2_90`**: 90% energy-sensitive (Class 2) tasks
3. **`s2_class3_90`**: 90% insensitive (Class 3) tasks
4. **`s2_random`**: Equal distribution (33% each)

---

## 🎯 Expected Behavior

### **Class 1 Tasks (90% Delay-Sensitive)**

**Before failure (0-500):**
- Local execution: ✅ Good QoE (fast enough)
- MEC offloading: ✅ Good QoE (fast + energy efficient)
- Cloud offloading: ❌ Bad QoE (too much latency)

**During failure (500-1000):**
- Local execution: ✅ **Only option** (works fine for Class 1)
- MEC/Cloud: ❌ **Fails** (no communication)
- RL agent should: **Switch to local immediately**

**After failure (1000-2000):**
- Resume MEC offloading (best balance)

### **Class 2 Tasks (90% Energy-Sensitive)**

**Before failure (0-500):**
- Local execution: ❌ Bad QoE (burns battery, fails deadline)
- MEC offloading: ✅ Good QoE (fast + efficient)
- Cloud offloading: ⚠️ OK QoE (better than local)

**During failure (500-1000):**
- Local execution: ❌ **Only option but fails** (no CPU power)
- MEC/Cloud: ❌ **Fails** (no communication)
- RL agent: **No good choice!** (will fail regardless)
  - Keeps trying MEC/Cloud to save battery
  - Local fails too but burns less energy

**After failure (1000-2000):**
- Resume MEC/Cloud offloading
- May have depleted battery during failure period

### **Class 3 Tasks (90% Insensitive)**

**Before failure (0-500):**
- All methods work (lax requirements)
- MEC/Cloud preferred (energy efficient)

**During failure (500-1000):**
- Local execution: ✅ **Works but drains battery heavily**
- MEC/Cloud: ❌ **Fails** (no communication)
- RL agent: **Forced to local** (battery drops fast)

**After failure (1000-2000):**
- Resume offloading if battery remains
- Some UEs may be dead from battery drain

### **Random Distribution**

**Before failure:**
- MEC offloading: Best balance (handles all classes)
- Local: Fails Class 2
- Cloud: Fails Class 1

**During failure:**
- Local execution: Only option
- Class 2 tasks will fail
- Battery drains from Class 3 local execution

**After failure:**
- RL agent resumes offloading
- May adapt slowly due to battery state

---

## 🚀 Running Scenario 2

### **1. Quick Test (One Distribution)**

```bash
# Test Class 1 distribution
python run_baselines_scenario.py --scenario s2_class1_90 --all
python run_scenario.py --scenario s2_class1_90 --train --episodes 100 --eval
```

### **2. Run All Scenario 2 Variants**

```bash
# Local execution
python run_all_scenarios.py --scenario-set s2 --episodes 500 --all

# Or on cluster
sbatch run_rl_nibi_s2.slurm  # (you'll need to create this)
```

### **3. Run Specific Distributions**

```bash
# Just Class 2 (energy-sensitive) - most interesting for failure!
python run_all_scenarios.py --scenarios s2_class2_90 --episodes 500 --all
```

---

## 📊 Expected Results

### **QoE Trends**

```
Class 1 (90%):
  Local baseline:  ~~~~ (constant, OK)
  MEC baseline:    ~~~~___~~~ (drops during failure)
  Cloud baseline:  ____ (always poor for Class 1)
  RL agent:        ~~~~ (adapts to local during failure)

Class 2 (90%):
  Local baseline:  ____ (always fails)
  MEC baseline:    ~~~~___~~~ (drops during failure)
  Cloud baseline:  ~~~~___~~~ (drops during failure)
  RL agent:        ~~~~___~~~ (no good option during failure)

Class 3 (90%):
  Local baseline:  ~~~~ (OK but drains battery)
  MEC baseline:    ~~~~___~~~ (fails during failure)
  Cloud baseline:  ~~~~___~~~ (fails during failure)
  RL agent:        ~~~~ (adapts to local, manages battery)
```

### **Battery Depletion**

- **Local baseline**: Fastest depletion (especially Class 3)
- **MEC/Cloud baselines**: Moderate depletion overall, but failures during 500-1000
- **RL agent**: Smart adaptation
  - Class 1: Minimal extra drain (local works fine)
  - Class 2: May deplete during failure (no good option)
  - Class 3: Heavy drain during 500-1000, manages after

### **Decision Patterns**

RL agent decision heatmap should show:
1. **0-500**: Mostly MEC (orange) with some Cloud (green)
2. **500-1000**: **Sudden switch to Local (blue)** - adaptation!
3. **1000-2000**: **Gradual return to MEC/Cloud** as communication restored

---

## 💾 File Structure After Running

```
results/scenarios/
├── s2_class1_90/
│   ├── local_metrics.csv
│   ├── mec_metrics.csv
│   ├── cloud_metrics.csv
│   ├── random_metrics.csv
│   ├── rl_agent_metrics.csv
│   ├── actor.pt
│   └── critic.pt
├── s2_class2_90/ (same structure)
├── s2_class3_90/ (same structure)
└── s2_random/ (same structure)
```

---

## 🔍 Analysis Tips

### **1. Check Communication Failure Impact**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load RL agent results
df = pd.read_csv('results/scenarios/s2_class1_90/rl_agent_metrics.csv')

# Plot actions over time
plt.figure(figsize=(12, 4))
plt.scatter(range(len(df)), df['Action'], c=df['Action'], cmap='viridis', alpha=0.5)
plt.axvspan(500, 1000, alpha=0.2, color='red', label='Communication Failure')
plt.xlabel('Timestep')
plt.ylabel('Action (0=Local, 1=MEC, 2=Cloud)')
plt.title('RL Agent Decisions - Scenario 2 Class 1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# You should see actions shift to 0 (local) during 500-1000!
```

### **2. Compare Battery Drain During Failure**

```python
# Load all methods
rl = pd.read_csv('results/scenarios/s2_class1_90/rl_agent_metrics.csv')
local = pd.read_csv('results/scenarios/s2_class1_90/local_metrics.csv')
mec = pd.read_csv('results/scenarios/s2_class1_90/mec_metrics.csv')

# Battery during failure period
print("Battery at t=500 (before failure):")
print(f"  RL:    {rl.iloc[500]['Battery']:.2f} J")
print(f"  Local: {local.iloc[500]['Battery']:.2f} J")
print(f"  MEC:   {mec.iloc[500]['Battery']:.2f} J")

print("\nBattery at t=1000 (after failure):")
print(f"  RL:    {rl.iloc[1000]['Battery']:.2f} J")
print(f"  Local: {local.iloc[1000]['Battery']:.2f} J")
print(f"  MEC:   {mec.iloc[1000]['Battery']:.2f} J")

print("\nBattery drain during failure (500-1000):")
print(f"  RL:    {rl.iloc[500]['Battery'] - rl.iloc[1000]['Battery']:.2f} J")
print(f"  Local: {local.iloc[500]['Battery'] - local.iloc[1000]['Battery']:.2f} J")
print(f"  MEC:   {mec.iloc[500]['Battery'] - mec.iloc[1000]['Battery']:.2f} J")
```

### **3. Measure Adaptation Speed**

```python
# How quickly does RL agent adapt at t=500?
failure_start = 500
window = 50  # Look at first 50 timesteps of failure

actions_during_failure = df.iloc[failure_start:failure_start+window]['Action']
local_ratio = (actions_during_failure == 0).mean()

print(f"RL agent local execution ratio in first {window} steps of failure: {local_ratio:.1%}")
print("(Should be high for Class 1/3, may be lower for Class 2)")
```

---

## ⚠️ Common Issues

### **Issue 1: MEC/Cloud baselines show no failures during 500-1000**

**Cause**: Baseline simulator not checking `has_communication()`

**Fix**: Already implemented in `run_baselines_scenario.py`. The simulator checks:
```python
if not has_communication:
    # Offloading FAILS
    latency = task.latency_deadline * 10.0
    energy = 0.0
```

### **Issue 2: RL agent doesn't adapt during failure**

**Possible causes**:
1. State doesn't include communication indicator
2. Agent hasn't seen enough training examples
3. Network architecture too small

**Solutions**:
- State now includes `has_comm` feature (1.0 or 0.0)
- Train for more episodes (500+)
- Check state dimension matches (should be 13 now)

### **Issue 3: Battery depletes too quickly for Class 2**

**Expected behavior**: This is correct!
- Class 2 tasks **cannot** be executed locally (not enough CPU)
- During communication failure, **all options fail**
- Agent will keep trying offloading (uses less battery than local)
- Battery may deplete from idle drain + failed attempts

---

## 📝 Paper Alignment

From the paper (Section on Scenario 2):

> "When there is a majority of Class 1 tasks... the separated-network AC algorithm decides to execute locally from that moment on"

✅ **Implemented**: RL agent receives communication status in state

> "When the connectivity comes back, this figure also shows that the decisions gradually become MEC server offloading"

✅ **Implemented**: Agent can adapt back to offloading after t=1000

> "Class 2 tasks... there is no good choice for the algorithm, as they will fail with any of the 3 possible options"

✅ **Implemented**: All execution modes properly fail for Class 2 locally, and offloading fails without communication

> "Class 3... the proposed AC algorithm starts to decide to execute the tasks locally, as it is the only way they can be correctly executed"

✅ **Implemented**: Local execution always available, agent learns to use it

---

## 🎯 Next Steps

1. **Run Scenario 2**:
   ```bash
   python run_all_scenarios.py --scenario-set s2 --episodes 500 --all
   ```

2. **Analyze results**: Focus on timesteps 500-1000

3. **Compare with Scenario 1**: Different adaptation patterns

4. **Create plots**: Visualize decision switching during failure

5. **Report**: Include figures showing:
   - QoE drop during failure (all methods)
   - Battery impact during failure period
   - Decision heatmap showing local execution during 500-1000
   - Recovery after communication restored

---

## 📧 Questions?

Scenario 2 is now fully implemented! The key differences:

✅ Communication can **completely fail** (multiplier = 0.0)
✅ MEC stays available (unlike Scenario 1)
✅ RL environment checks `has_communication()` before offloading
✅ Baselines fail appropriately when communication is lost
✅ State includes communication indicator for RL agent

Run it and see how well the RL agent adapts to complete communication loss! 🚀
