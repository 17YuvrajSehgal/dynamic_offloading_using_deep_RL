#!/bin/bash
#SBATCH --job-name=rl_debug           # Job name
#SBATCH --account=def-naser2         # Your PI account
#SBATCH --gres=gpu:1                 # 1 GPU
#SBATCH --mem=16G                    # Memory
#SBATCH --cpus-per-task=4            # CPU cores
#SBATCH --time=02:00:00              # Walltime (adjust as needed)
#SBATCH --output=logs/rl_debug_%j.out
#SBATCH --error=logs/rl_debug_%j.err

# ----- USER CONFIG -----
EPISODES=100
LOG_EVERY=5

# ----- BASIC SETUP -----
set -e

mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

# Activate virtualenv (assumes .venv exists in project root)
source .venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $(pwd)"
echo "Episodes: $EPISODES, log_every: $LOG_EVERY"
echo ""

# Threading limits (optional but good practice)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# ----- RUN DEBUG TRAINING -----
CMD="python -u train_rl.py --episodes $EPISODES --log-every $LOG_EVERY"
echo "[CMD] $CMD"
echo ""

$CMD
EXIT_CODE=$?

echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE