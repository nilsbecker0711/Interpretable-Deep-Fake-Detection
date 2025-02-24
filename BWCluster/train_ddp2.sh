#!/bin/bash
#SBATCH -p gpu_8  # Use the dev_gpu_4_a100 partition with A100 GPUs
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 06:00:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=60000             # Memory request (adjust as needed)
<<<<<<< HEAD
#SBATCH --gres=gpu:4          # Request 1 GPU (adjust if you need more)
=======
#SBATCH --gres=gpu:4           # Request 1 GPU (adjust if you need more)
>>>>>>> 9be4f264f00c6fc74996ae5e07c5823f990f3f52
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

module load devel/miniconda

conda activate TP_linux

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Environment: $(conda info --envs)"

module load devel/cuda/12.4

export RANK=0                  # Set the rank of the current process (0 for first process)
export WORLD_SIZE=4            # Set the total number of processes (2 for two GPUs)
# export LOCAL_RANK=0            # Local rank (used by DDP for each process)
export MASTER_ADDR="localhost"  # The master node's address (typically localhost for single-node)
export MASTER_PORT=29300    # The port for communication (can be any available port)

# Launch the training with two GPUs
<<<<<<< HEAD
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29300 ~/Interpretable-Deep-Fake-Detection/training/train.py --detector_path ~/Interpretable-Deep-Fake-Detection/training/config/detector/resnet34_bcos.yaml --ddp
=======
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29300 ~/Interpretable-Deep-Fake-Detection/training/train.py --detector_path ~/Interpretable-Deep-Fake-Detection/training/config/detector/vgg_bcos.yaml --ddp
>>>>>>> 9be4f264f00c6fc74996ae5e07c5823f990f3f52
