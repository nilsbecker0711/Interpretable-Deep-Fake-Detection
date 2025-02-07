#!/bin/bash
#SBATCH -p gpu_4  # Use the dev_gpu_4_a100 partition with A100 GPUs
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 02:00:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=40000             # Memory request (adjust as needed)
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if you need more)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

module load devel/miniconda

conda activate TP_linux

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Environment: $(conda info --envs)"

module load devel/cuda/12.4

python ~/Interpretable-Deep-Fake-Detection/training/train.py --detector_path ~/Interpretable-Deep-Fake-Detection/training/config/detector/xception.yaml