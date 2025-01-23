#!/bin/bash
#SBATCH -p single      # Use the dev_gpu_4_a100 partition with A100 GPUs
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 10:00:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=60000             # Memory request (adjust as needed)
#SBATCH --cpus-per-task=32     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

module load devel/miniconda

conda activate TP_linux

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Environment: $(conda info --envs)"

cd /home/ma/ma_ma/ma_kreffert/Interpretable-Deep-Fake-Detection/preprocessing/

python /home/ma/ma_ma/ma_kreffert/Interpretable-Deep-Fake-Detection/preprocessing/preprocess.py