#!/bin/bash
#SBATCH -p dev_gpu_4_a100      # Use the dev_gpu_4_a100 partition with A100 GPUs
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 00:30:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=50000             # Memory request (adjust as needed)
#SBATCH --gres=gpu:2           # Request 1 GPU (adjust if you need more)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

module load devel/miniconda

conda activate TP_linux

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Environment: $(conda info --envs)"

python ~/Interpretable-Deep-Fake-Detection/BWCluster/wandb_sync.py