#!/bin/bash
#SBATCH -p dev_single      # Use the dev_gpu_4_a100 partition with A100 GPUs
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 00:30:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=5000             # Memory request (adjust as needed)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

module load devel/miniconda

conda activate TP_linux

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Environment: $(conda info --envs)"

python ~/interpretable-deep-fake-detection/data/raw/faceforensics_download_v4.py ~/interpretable-deep-fake-detection/DeepfakeBench/datasets/rgb/FaceForensics++ -d all -c c40 -t videos --server EU2

# python /home/ma/ma_ma/ma_kreffert/interpretable-deep-fake-detection/data/raw/faceforensics_download_v4.py /home/ma/ma_ma/ma_kreffert/interpretable-deep-fake-detection/DeepFakeBench/datasets/rgb -d all -c c40 -t masks --server EU2