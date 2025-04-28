#!/bin/bash
#SBATCH -p gpu20  # Use the dev_gpu_4_a100 partition with A100 GPUs dev_gpu_4
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 23:59:00           # Time limit (10 minutes for debugging purposes)
#SBATCH --gres=gpu:2           # Request 1 GPU (adjust if you need more)
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

# python ~/Interpretable-Deep-Fake-Detection/training/hp_tuning.py --detector_path ~/Interpretable-Deep-Fake-Detection/training/config/detector/resnet34_bcos.yaml

export RANK=0                  # Set the rank of the current process (0 for first process)
export WORLD_SIZE=2            # Set the total number of processes (2 for two GPUs)
# export LOCAL_RANK=0            # Local rank (used by DDP for each process)
export MASTER_ADDR="localhost"  # The master node's address (typically localhost for single-node)
export MASTER_PORT=29100    # The port for communication (can be any available port)

# Launch the training with two G<PUs
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29100 /BS/robust-architectures/work/bcos-df/training/hp_tuning.py --detector_path /BS/robust-architectures/work/bcos-df/training/config/detector/resnet34.yaml --sweep_id xnle4kcz --ddp
#python ~/Interpretable-Deep-Fake-Detection/training/hp_tuning.py --detector_path ~/Interpretable-Deep-Fake-Detection/training/config/detector/resnet34.yaml