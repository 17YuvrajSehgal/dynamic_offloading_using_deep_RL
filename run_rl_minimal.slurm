#!/bin/bash
#SBATCH --job-name=rl_scenarios          # Job name
#SBATCH --account=def-naser2             # Your PI account
#SBATCH --gres=gpu:1                     # 1 GPU
#SBATCH --mem=32G                        # Memory
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --time=12:00:00                  # Walltime
#SBATCH --output=logs/rl_%j.out          # STDOUT
#SBATCH --error=logs/rl_%j.err           # STDERR

# ------------------------------
# USER CONFIG
# ------------------------------
SCENARIO_SET="s1"        # s1, s2, s1_base, s2_base, all
EPISODES=200             # training episodes per scenario
RESULTS_DIR="results/scenarios"

DO_TRAIN=true            # train RL agents
DO_EVAL=true             # evaluate RL agents
DO_BASELINES=true        # run baselines (includes greedy)
DO_PLOT=true             # generate plots

# ------------------------------
# BASIC SETUP
# ------------------------------
set -e

mkdir -p logs
mkdir -p "$RESULTS_DIR"

cd "$SLURM_SUBMIT_DIR"

# Activate virtualenv (assumes you already created .venv)
source .venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $(pwd)"
echo "Scenario set: $SCENARIO_SET"
echo "Episodes: $EPISODES"
echo ""

# Limit threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# ------------------------------
# BUILD COMMAND
# ------------------------------
CMD="python -u run_all_scenarios.py --scenario-set $SCENARIO_SET --episodes $EPISODES --results-dir $RESULTS_DIR"

if [ "$DO_TRAIN" = true ]; then
  CMD="$CMD --train"
fi

if [ "$DO_EVAL" = true ]; then
  CMD="$CMD --eval"
fi

if [ "$DO_BASELINES" = true ]; then
  CMD="$CMD --baselines"
fi

if [ "$DO_PLOT" = true ]; then
  CMD="$CMD --plot"
fi

# Shortcut equivalent to --all when we’re doing everything
if [ "$DO_TRAIN" = true ] && [ "$DO_EVAL" = true ] && [ "$DO_BASELINES" = true ] && [ "$DO_PLOT" = true ]; then
  CMD="python -u run_all_scenarios.py --scenario-set $SCENARIO_SET --episodes $EPISODES --all --results-dir $RESULTS_DIR"
fi

echo "[CMD] $CMD"
echo ""

# ------------------------------
# RUN
# ------------------------------
$CMD
EXIT_CODE=$?

echo "Exit code: $EXIT_CODE"
echo "Results dir: $(pwd)/$RESULTS_DIR"

exit $EXIT_CODE