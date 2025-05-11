#!/bin/bash
#SBATCH -p dev_gpu_h100  # Use the dev_gpu_4_a100 partition with A100 GPUs dev_gpu_4
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 10           # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=60000             # Memory request (adjust as needed)
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if you need more)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

module load devel/cuda/12.8
module load devel/miniforge

conda activate GPG #/pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/envs/TP_main_2

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Environment: $(conda info --envs)"

# RESNET
# python ~/Interpretable-Deep-Fake-Detection/training/test.py --detector_path ~/Interpretable-Deep-Fake-Detection/BWCluster/test/resnet34_best_hpo.yaml --weights_path /pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/weights/best_weights/resnet34.pth

# python ~/Interpretable-Deep-Fake-Detection/training/test.py --detector_path ~/Interpretable-Deep-Fake-Detection/BWCluster/test/resnet34_bcos_v2_1_25_best_hpo.yaml --weights_path /pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/weights/best_weights/resnet34_bcos_1_25.pth
python ~/Interpretable-Deep-Fake-Detection/training/test.py --detector_path ~/Interpretable-Deep-Fake-Detection/BWCluster/test/resnet34_bcos_v2_2_5_best_hpo.yaml --weights_path /pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/weights/best_weights/resnet34_bcos_2_5.pth

# ViT
python ~/Interpretable-Deep-Fake-Detection/training/test.py --detector_path ~/Interpretable-Deep-Fake-Detection/BWCluster/test/vit_bcos_1_25_best_hpo.yaml --weights_path /pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/weights/best_weights/vit_bcos_1_25.pth
python ~/Interpretable-Deep-Fake-Detection/training/test.py --detector_path ~/Interpretable-Deep-Fake-Detection/BWCluster/test/vit_bcos_1_75_best_hpo.yaml --weights_path /pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/weights/best_weights/vit_bcos_1_75.pth
python ~/Interpretable-Deep-Fake-Detection/training/test.py --detector_path ~/Interpretable-Deep-Fake-Detection/BWCluster/test/vit_bcos_2_best_hpo.yaml --weights_path /pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/weights/best_weights/vit_bcos_2.pth
python ~/Interpretable-Deep-Fake-Detection/training/test.py --detector_path ~/Interpretable-Deep-Fake-Detection/BWCluster/test/vit_bcos_2_5_best_hpo.yaml --weights_path /pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/weights/best_weights/vit_bcos_2_5.pth