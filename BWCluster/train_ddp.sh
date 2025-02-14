#!/bin/bash
#SBATCH -p gpu_8  # Use the dev_gpu_4_a100 partition with A100 GPUs
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 05:00:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=45000             # Memory request (adjust as needed)
#SBATCH --gres=gpu:4          # Request 1 GPU (adjust if you need more)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

module load devel/miniconda

conda activate TP_main

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Environment: $(conda info --envs)"

module load devel/cuda/12.4

export RANK=0                  # Set the rank of the current process (0 for first process)
export WORLD_SIZE=4            # Set the total number of processes (2 for two GPUs)
# export LOCAL_RANK=0            # Local rank (used by DDP for each process)
export MASTER_ADDR="localhost"  # The master node's address (typically localhost for single-node)
export MASTER_PORT=29500    # The port for communication (can be any available port)

# Launch the training with two GPUs

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29500 ~/Interpretable-Deep-Fake-Detection/training/train.py --detector_path ~/Interpretable-Deep-Fake-Detection/training/config/detector/resnet34_bcos.yaml --ddp
