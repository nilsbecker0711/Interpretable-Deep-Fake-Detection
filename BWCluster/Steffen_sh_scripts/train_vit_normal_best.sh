#!/bin/bash
#SBATCH -p gpu20  # Use the dev_gpu_4_a100 partition with A100 GPUs dev_gpu_4
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 23:59:00           # Time limit (10 minutes for debugging purposes)
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if you need more)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

eval "$(conda shell.bash hook)"

conda activate /BS/robust-architectures/work/bcos-df-env
cd /BS/robust-architectures/work/bcos-df

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Environment: $(conda info --envs)"

python /BS/robust-architectures/work/bcos-df/training/train.py --detector_path /BS/robust-architectures/work/bcos-df/training/config/detector/vit_best_hpo.yaml
