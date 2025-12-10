#!/bin/bash
#SBATCH --job-name=rl_all_s1              # Job name
#SBATCH --account=def-naser2              # Your PI account
#SBATCH --gres=gpu:1                      # 1 GPU
#SBATCH --mem=32G                         # Increased memory for running all scenarios
#SBATCH --cpus-per-task=4                 # CPU cores
#SBATCH --time=12:00:00                   # 12 hours for all 4 scenarios
#SBATCH --output=logs/rl_all_%j.out       # STDOUT (%j = job ID)
#SBATCH --error=logs/rl_all_%j.err        # STDERR

echo "===================================="
echo " RL All Scenarios Job Started"
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo " Time: $(date)"
echo "===================================="
echo ""

#------------------------------
# CONFIGURATION - EDIT THESE
#------------------------------
# Scenario set to run (options: s1, s2, s1_base, all)
SCENARIO_SET="s1"

# Number of training episodes per scenario
EPISODES=500

# What to do
DO_TRAIN=true      # Train agents
DO_EVAL=true       # Evaluate agents  
DO_PLOT=true       # Generate plots

echo "[CONFIG] Scenario Set: $SCENARIO_SET"
echo "[CONFIG] Episodes per scenario: $EPISODES"
echo "[CONFIG] Train: $DO_TRAIN"
echo "[CONFIG] Evaluate: $DO_EVAL"
echo "[CONFIG] Plot: $DO_PLOT"
echo ""

#------------------------------
# 0) Create directories
#------------------------------
mkdir -p logs
mkdir -p results
mkdir -p results/scenarios

#------------------------------
# 1) Load modules
#------------------------------
echo "[SETUP] Loading modules..."
module purge
module load python/3.13.2

echo "[SETUP] Loaded modules:"
module list
echo ""

#------------------------------
# 2) Enter submission directory
#------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "[SETUP] Working directory: $(pwd)"
echo ""

#------------------------------
# 3) Activate virtual environment
#------------------------------
echo "[SETUP] Activating virtual environment..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "[SETUP] ✓ Virtual environment activated"
else
    echo "[ERROR] Virtual environment not found at .venv/"
    echo "[ERROR] Please create it first with: python -m venv .venv"
    exit 1
fi
echo ""

#------------------------------
# 4) Print Python and PyTorch info
#------------------------------
echo "====================================="
echo "PYTHON & PYTORCH DIAGNOSTICS"
echo "====================================="
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Pip version: $(pip --version)"
echo ""

echo "[PYTORCH] Testing PyTorch installation..."
python << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
PYEOF

if [ $? -ne 0 ]; then
    echo "[ERROR] PyTorch import failed!"
    exit 1
fi
echo ""

#------------------------------
# 5) Print environment variables
#------------------------------
echo "====================================="
echo "ENVIRONMENT VARIABLES"
echo "====================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
echo ""

#------------------------------
# 6) Set CPU threading limits
#------------------------------
echo "[SETUP] Setting CPU threading limits..."
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
echo "[SETUP] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[SETUP] MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo ""

#------------------------------
# 7) Check GPU with nvidia-smi
#------------------------------
echo "====================================="
echo "GPU DIAGNOSTICS (nvidia-smi)"
echo "====================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
    echo "[GPU] ✓ GPU detected"
else
    echo "[WARNING] nvidia-smi not found"
fi
echo ""

#------------------------------
# 8) Verify required packages
#------------------------------
echo "====================================="
echo "PYTHON PACKAGE VERIFICATION"
echo "====================================="
echo "[PACKAGES] Checking required packages..."
python << 'PYEOF'
import sys
try:
    import numpy
    print(f"NumPy: {numpy.__version__}")
    import pandas
    print(f"Pandas: {pandas.__version__}")
    import matplotlib
    print(f"Matplotlib: {matplotlib.__version__}")
    import torch
    print(f"PyTorch: {torch.__version__}")
    print("All packages verified")
except ImportError as e:
    print(f"ERROR: Missing package - {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "[ERROR] Package verification failed"
    exit 1
fi
echo ""

#------------------------------
# 9) List available scenarios
#------------------------------
echo "====================================="
echo "AVAILABLE SCENARIOS"
echo "====================================="
python run_all_scenarios.py --list
echo ""

#------------------------------
# 10) Run all scenarios
#------------------------------
echo "====================================="
echo "RUNNING ALL SCENARIOS: $SCENARIO_SET"
echo "====================================="
echo "Start time: $(date)"
echo ""

# Build command
CMD="python -u run_all_scenarios.py --scenario-set $SCENARIO_SET --episodes $EPISODES"

if [ "$DO_TRAIN" = true ]; then
    CMD="$CMD --train"
fi

if [ "$DO_EVAL" = true ]; then
    CMD="$CMD --eval"
fi

if [ "$DO_PLOT" = true ]; then
    CMD="$CMD --plot"
fi

echo "[CMD] $CMD"
echo ""

# Execute
$CMD

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo ""

#------------------------------
# 11) Report results
#------------------------------
if [ $EXIT_CODE -eq 0 ]; then
    echo "====================================="
    echo "✓ ALL SCENARIOS COMPLETED SUCCESSFULLY"
    echo "====================================="
else
    echo "====================================="
    echo "✗ JOB FAILED WITH EXIT CODE: $EXIT_CODE"
    echo "====================================="
fi

echo ""
echo "Results location: $(pwd)/results/scenarios/"
echo "Logs location: $(pwd)/logs/"
echo ""

#------------------------------
# 12) Show output files
#------------------------------
echo "====================================="
echo "OUTPUT FILES"
echo "====================================="

if [ -d "results/scenarios" ]; then
    echo "Scenario directories:"
    ls -d results/scenarios/*/ 2>/dev/null | head -10
    echo ""
    
    echo "CSV files:"
    find results/scenarios -name "*.csv" -type f | head -10
    echo ""
    
    echo "Model files:"
    find results/scenarios -name "*.pt" -type f | head -10
    echo ""
    
    echo "Plot files:"
    find results/scenarios -name "*.png" -type f | head -10
    echo ""
else
    echo "[WARNING] Results directory not found"
fi

echo "====================================="
echo " Job Finished"
echo "====================================="

exit $EXIT_CODE
