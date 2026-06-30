#!/bin/bash
#SBATCH -p dev_cpuonly      # Use the dev_gpu_4_a100 partition with A100 GPUs
#SBATCH -N 1                   # Number of tasks (1 for single node)
#SBATCH -t 00:10:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=100000             # Memory request (adjust as needed)
#SBATCH --ntasks-per-node=4    # Number of tasks per node (1 in this case)


echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"

source $(ws_find einspace_ws)/bcos_venv/bin/activate

python /home/hk-project-pai00137/ma_tischuet/Interpretable-Deep-Fake-Detection/preprocessing/rearrange.py