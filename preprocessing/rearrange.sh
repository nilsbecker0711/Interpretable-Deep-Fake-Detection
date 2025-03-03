#!/bin/bash
#SBATCH -p multiple_il      # Use the dev_gpu_4_a100 partition with A100 GPUs
#SBATCH -N 4                   # Number of tasks (1 for single node)
#SBATCH -t 02:00:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=100000             # Memory request (adjust as needed)
#SBATCH --ntasks-per-node=4    # Number of tasks per node (1 in this case)

module load devel/miniconda

conda activate /pfs/work7/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/envs/TP_main

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Environment: $(conda info --envs)"

python /home/ma/ma_ma/ma_tischuet/Interpretable-Deep-Fake-Detection/preprocessing/rearrange.py