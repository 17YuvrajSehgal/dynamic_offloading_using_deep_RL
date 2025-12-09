#!/bin/bash
#SBATCH --job-name=rl_offloading            # Job name
#SBATCH --account=def-naser2                # Your PI account
#SBATCH --gres=gpu:1                        # 1 GPU
#SBATCH --mem=16G                           # Memory for CPU side
#SBATCH --cpus-per-task=4                   # CPU cores
#SBATCH --time=12:00:00                     # Max runtime (hh:mm:ss)
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
module load python/3.13.2
module load cuda/12.0
module load cudnn/9.13.1.26

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
