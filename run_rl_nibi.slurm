#!/bin/bash
#SBATCH --job-name=rl_offload_s1           # Job name (changed to indicate scenario)
#SBATCH --account=def-naser2                # Your PI account
#SBATCH --gres=gpu:1                        # 1 GPU
#SBATCH --mem=16G                           # CPU-side memory
#SBATCH --cpus-per-task=4                   # CPU cores
#SBATCH --time=04:00:00                     # Max runtime (increased to 4 hours for scenarios)
#SBATCH --output=logs/rl_%j.out             # STDOUT (%j = job ID)
#SBATCH --error=logs/rl_%j.err              # STDERR

echo "===================================="
echo " RL Scenario Simulation Started"
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo " Time: $(date)"
echo "===================================="

#------------------------------
# Configuration - EDIT THESE
#------------------------------
# Scenario to run (options: s1_base, s1_class1_90, s1_class2_90, s1_class3_90, s1_random, s2_base)
SCENARIO="s1_class1_90"

# Number of training episodes
EPISODES=500

# Whether to train (true) or just evaluate existing model (false)
DO_TRAIN=true

# Whether to evaluate after training
DO_EVAL=true

echo "[CONFIG] Scenario: $SCENARIO"
echo "[CONFIG] Training episodes: $EPISODES"
echo "[CONFIG] Train: $DO_TRAIN"
echo "[CONFIG] Evaluate: $DO_EVAL"
echo ""

#------------------------------
# 0) Create directories
#------------------------------
mkdir -p logs
mkdir -p results
mkdir -p results/scenarios

#------------------------------
# 1) Load required modules
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
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('cuDNN version:', torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A')"

if [ $? -ne 0 ]; then
    echo "[ERROR] PyTorch import failed!"
    echo "[ERROR] Installing/upgrading PyTorch..."
    pip install --upgrade torch torchvision torchaudio
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
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-Not set}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS:-Not set}"
echo ""

#------------------------------
# 6) Set CPU threading limits
#------------------------------
echo "[SETUP] Setting CPU threading limits..."
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[SETUP] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[SETUP] MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo ""

#------------------------------
# 7) Check GPU availability
#------------------------------
echo "====================================="
echo "GPU DIAGNOSTICS (nvidia-smi)"
echo "====================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
    echo "[GPU] ✓ GPU detected by nvidia-smi"
else
    echo "[WARNING] nvidia-smi not found - GPU may not be available"
fi
echo ""

#------------------------------
# 8) Verify required packages
#------------------------------
echo "====================================="
echo "PYTHON PACKAGE VERIFICATION"
echo "====================================="
echo "[PACKAGES] Checking required packages..."
python -c "import numpy; print('NumPy:', numpy.__version__)" || pip install numpy
python -c "import pandas; print('Pandas:', pandas.__version__)" || pip install pandas
python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)" || pip install matplotlib
python -c "import scipy; print('SciPy:', scipy.__version__)" || pip install scipy
python -c "import torch; print('PyTorch:', torch.__version__)" || pip install torch
echo "[PACKAGES] ✓ All packages verified"
echo ""

#------------------------------
# 9) List available scenarios
#------------------------------
echo "====================================="
echo "AVAILABLE SCENARIOS"
echo "====================================="
python run_scenario.py --list
echo ""

#------------------------------
# 10) Run scenario simulation
#------------------------------
echo "====================================="
echo "RUNNING SCENARIO: $SCENARIO"
echo "====================================="
echo "Start time: $(date)"
echo ""

# Set Python to unbuffered mode for real-time logging
export PYTHONUNBUFFERED=1

# Build command based on configuration
if [ "$DO_TRAIN" = true ] && [ "$DO_EVAL" = true ]; then
    # Full pipeline: train + evaluate
    echo "[MODE] Full pipeline: Training + Evaluation"
    python -u run_scenario.py \
        --scenario "$SCENARIO" \
        --full-pipeline \
        --episodes "$EPISODES"
    
elif [ "$DO_TRAIN" = true ]; then
    # Training only
    echo "[MODE] Training only"
    python -u run_scenario.py \
        --scenario "$SCENARIO" \
        --train \
        --episodes "$EPISODES"
    
elif [ "$DO_EVAL" = true ]; then
    # Evaluation only
    echo "[MODE] Evaluation only (using existing trained model)"
    python -u run_scenario.py \
        --scenario "$SCENARIO" \
        --eval
    
else
    echo "[ERROR] At least one of DO_TRAIN or DO_EVAL must be true"
    exit 1
fi

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo ""

#------------------------------
# 11) Report results
#------------------------------
if [ $EXIT_CODE -eq 0 ]; then
    echo "====================================="
    echo "✓ JOB COMPLETED SUCCESSFULLY"
    echo "====================================="
else
    echo "====================================="
    echo "✗ JOB FAILED WITH EXIT CODE: $EXIT_CODE"
    echo "====================================="
fi

echo ""
echo "Results location: $(pwd)/results/scenarios/$SCENARIO/"
echo "Logs location: $(pwd)/logs/"
echo ""

#------------------------------
# 12) Show output files
#------------------------------
echo "====================================="
echo "OUTPUT FILES"
echo "====================================="

if [ -d "results/scenarios/$SCENARIO" ]; then
    echo "Scenario results:"
    ls -lh "results/scenarios/$SCENARIO/"
    echo ""
fi

if [ -d "results" ]; then
    echo "All results:"
    find results -name "*.csv" -o -name "*.pt" -o -name "*.png" | head -20
    echo ""
else
    echo "[WARNING] Results directory not found"
fi

echo "====================================="
echo " Job Finished"
echo "====================================="

exit $EXIT_CODE
