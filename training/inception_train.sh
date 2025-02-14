#!/bin/bash
#SBATCH -p dev_gpu_4     # Use the dev_gpu_4_a100 partition with A100 GPUs
#SBATCH -t 00:10:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=80000             # Memory request (adjust as needed)
#SBATCH --gres=gpu:1            # number of GPUs
#SBATCH --ntasks=20    # Number of tasks per node (1 in this case)

module load devel/miniconda

conda activate /pfs/work7/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/envs/TP_main

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Environment: $(conda info --envs)"

python /home/ma/ma_ma/ma_tischuet/Interpretable-Deep-Fake-Detection/training/train.py \
--detector_path ./config/config/detector/inception_bcos.yaml