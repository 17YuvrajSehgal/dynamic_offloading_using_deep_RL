#!/bin/bash
#SBATCH --job-name=rl_improved       # Job name
#SBATCH --account=def-naser2         # Your PI account
#SBATCH --gres=gpu:1                 # 1 GPU
#SBATCH --mem=16G                    # Memory
#SBATCH --cpus-per-task=4            # CPU cores
#SBATCH --time=02:00:00              # Walltime
#SBATCH --output=logs/rl_improved_%j.out
#SBATCH --error=logs/rl_improved_%j.err

# ----- USER CONFIG -----
EPISODES=100
LOG_EVERY=5
REWARD_SCALE=10.0
ENTROPY_COEFF=0.05
BATTERY_PENALTY=0.1

# ----- BASIC SETUP -----
set -e

mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

# Activate virtualenv
source .venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $(pwd)"
echo "Episodes: $EPISODES"
echo "Reward scale: $REWARD_SCALE"
echo "Entropy coeff: $ENTROPY_COEFF"
echo "Battery penalty: $BATTERY_PENALTY"
echo ""

# Threading limits
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# ----- RUN IMPROVED TRAINING -----
CMD="python -u train_rl_improved.py --episodes $EPISODES --log-every $LOG_EVERY --reward-scale $REWARD_SCALE --entropy-coeff $ENTROPY_COEFF --battery-penalty $BATTERY_PENALTY"
echo "[CMD] $CMD"
echo ""

$CMD
EXIT_CODE=$?

echo ""
echo "Exit code: $EXIT_CODE"
echo "Results saved to: $(pwd)/results/"

exit $EXIT_CODE

