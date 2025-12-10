#!/bin/bash
#SBATCH --job-name=rl_offloading            # Job name
#SBATCH --account=def-naser2                # Your PI account
#SBATCH --gres=gpu:1                        # 1 GPU
#SBATCH --mem=16G                           # CPU-side memory
#SBATCH --cpus-per-task=4                   # CPU cores
#SBATCH --time=02:00:00                     # Max runtime (increased to 2 hours)
#SBATCH --output=logs/rl_%j.out             # STDOUT (%j = job ID)
#SBATCH --error=logs/rl_%j.err              # STDERR

echo "===================================="
echo " RL Offloading Job Started"
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo " Time: $(date)"
echo "===================================="

#------------------------------
# 0) Create logs directory if it doesn't exist
#------------------------------
mkdir -p logs
mkdir -p results

#------------------------------
# 1) Load required modules
#------------------------------
echo "\n[SETUP] Loading modules..."
module purge
module load python/3.13.2   # or whatever python module you used to build .venv

echo "[SETUP] Loaded modules:"
module list

#------------------------------
# 2) Enter submission directory
#------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "\n[SETUP] Working directory: $(pwd)"

#------------------------------
# 3) Activate virtual environment
#------------------------------
echo "\n[SETUP] Activating virtual environment..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "[SETUP] Virtual environment activated"
else
    echo "[ERROR] Virtual environment not found at .venv/"
    echo "[ERROR] Please create it first with: python -m venv .venv"
    exit 1
fi

#------------------------------
# 4) Print Python and PyTorch info
#------------------------------
echo "\n====================================="
echo "PYTHON & PYTORCH DIAGNOSTICS"
echo "====================================="
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Pip version: $(pip --version)"

echo "\n[PYTORCH] Testing PyTorch installation..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('cuDNN version:', torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A')"

if [ $? -ne 0 ]; then
    echo "[ERROR] PyTorch import failed!"
    echo "[ERROR] Installing/upgrading PyTorch..."
    pip install --upgrade torch torchvision torchaudio
fi

#------------------------------
# 5) Print environment variables
#------------------------------
echo "\n====================================="
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

#------------------------------
# 6) Set CPU threading limits
#------------------------------
echo "\n[SETUP] Setting CPU threading limits..."
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[SETUP] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[SETUP] MKL_NUM_THREADS=$MKL_NUM_THREADS"

#------------------------------
# 7) Check GPU availability with nvidia-smi
#------------------------------
echo "\n====================================="
echo "GPU DIAGNOSTICS (nvidia-smi)"
echo "====================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "\n[GPU] GPU detected by nvidia-smi"
else
    echo "[WARNING] nvidia-smi not found - GPU may not be available"
fi

#------------------------------
# 8) Verify required Python packages
#------------------------------
echo "\n====================================="
echo "PYTHON PACKAGE VERIFICATION"
echo "====================================="
echo "[PACKAGES] Checking required packages..."
python -c "import numpy; print('NumPy:', numpy.__version__)" || pip install numpy
python -c "import pandas; print('Pandas:', pandas.__version__)" || pip install pandas
python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)" || pip install matplotlib
python -c "import scipy; print('SciPy:', scipy.__version__)" || pip install scipy
echo "[PACKAGES] All packages verified"

#------------------------------
# 9) Run RL Training and Evaluation
#------------------------------
echo "\n====================================="
echo "STARTING RL TRAINING & EVALUATION"
echo "====================================="
echo "Start time: $(date)"
echo "\n"

# Set Python to unbuffered mode for real-time logging
export PYTHONUNBUFFERED=1

# Run with --train flag to train before evaluation
# Use --eval-only if you already have trained weights
python -u rl_baseline_eval.py --train

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "\n====================================="
    echo "JOB COMPLETED SUCCESSFULLY"
    echo "====================================="
else
    echo "\n====================================="
    echo "JOB FAILED WITH EXIT CODE: $EXIT_CODE"
    echo "====================================="
fi

echo "End time: $(date)"
echo "Results saved in: $(pwd)/results/"
echo "Logs saved in: $(pwd)/logs/"

#------------------------------
# 10) Print output file locations
#------------------------------
echo "\n====================================="
echo "OUTPUT FILES"
echo "====================================="
if [ -d "results" ]; then
    echo "Results directory contents:"
    ls -lh results/
else
    echo "[WARNING] Results directory not found"
fi

echo "\n===================================="
echo " RL Offloading Job Finished"
echo "===================================="

exit $EXIT_CODE
