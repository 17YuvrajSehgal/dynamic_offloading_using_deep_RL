#!/bin/bash
#SBATCH --job-name=rl_debug           # Job name
#SBATCH --account=def-naser2         # Your PI account
#SBATCH --gres=gpu:1                 # 1 GPU
#SBATCH --mem=16G                    # Memory
#SBATCH --cpus-per-task=4            # CPU cores
#SBATCH --time=01:00:00              # 1 hour should be enough for debugging
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

# ----- BASIC SETUP -----
set -e

mkdir -p logs
mkdir -p debug_output

cd "$SLURM_SUBMIT_DIR"

# Activate virtualenv
source .venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $(pwd)"
echo ""

# Threading limits
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# ----- RUN DEBUGGING SCRIPT -----
CMD="python -u debug_ac_learning.py"
echo "[CMD] $CMD"
echo ""

$CMD
EXIT_CODE=$?

echo ""
echo "Exit code: $EXIT_CODE"
echo "Debug output saved to: $(pwd)/debug_output/"

exit $EXIT_CODE

