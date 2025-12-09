#!/bin/bash
#SBATCH --job-name=rl_offloading            # Job name
#SBATCH --account=def-naser2                # Your PI account
#SBATCH --gres=gpu:1                        # 1 GPU
#SBATCH --mem=16G                           # CPU-side memory
#SBATCH --cpus-per-task=4                   # CPU cores
#SBATCH --time=02:00:00                     # Max runtime (hh:mm:ss)
#SBATCH --output=logs/rl_%j.out             # STDOUT (%j = job ID)
#SBATCH --error=logs/rl_%j.err              # STDERR

echo "===================================="
echo " RL Offloading Job Started"
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo " Time: $(date)"
echo "===================================="

#------------------------------
# 1) Load required modules
#------------------------------
module purge
module load python/3.13.2   # or whatever python module you used to build .venv

# NOTE:
# We are NOT loading cuda/cudnn modules here, because
# your PyTorch wheel in .venv should already include CUDA.
# If later you discover you need a system CUDA module, run
#   module spider cuda
# on the login node to see the correct name/version.

#------------------------------
# 2) Enter submission directory
#------------------------------
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

#------------------------------
# 3) Activate virtual environment
#------------------------------
source .venv/bin/activate

echo "Python version: $(python --version)"
python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "--------------------------------------"

#------------------------------
# 4) Set CPU threading limits
#------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

#------------------------------
# 5) Run RL Training
#------------------------------
echo "Starting RL training..."
python train_rl.py

echo "===================================="
echo " RL Offloading Job Finished"
echo " End Time: $(date)"
echo "===================================="
