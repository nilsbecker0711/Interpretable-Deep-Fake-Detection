#!/bin/bash
#SBATCH -p multiple      
#SBATCH -N 4                   
#SBATCH -t 03:00:00            
#SBATCH --mem=50000             
#SBATCH --ntasks-per-node=4

module load devel/miniconda

conda activate /pfs/work7/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/envs/TP_main

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Environment: $(conda info --envs)"

python /home/ma/ma_ma/ma_tischuet/Interpretable-Deep-Fake-Detection/datasets/faceforensics_download_v4.py /pfs/work7/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/datasets/rgb/FaceForensics++ -d all -c c40 -t videos --server EU2

python /home/ma/ma_ma/ma_tischuet/Interpretable-Deep-Fake-Detection/datasets/faceforensics_download_v4.py /pfs/work7/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/datasets/rgb/FaceForensics++ -d all -c c40 -t masks --server EU2

python /home/ma/ma_ma/ma_tischuet/Interpretable-Deep-Fake-Detection/datasets/faceforensics_download_v4.py /pfs/work7/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/datasets/rgb/FaceForensics++ -d FaceSwap -c c40 -t masks --server EU2

python /home/ma/ma_ma/ma_tischuet/Interpretable-Deep-Fake-Detection/datasets/faceforensics_download_v4.py /pfs/work7/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/datasets/rgb/FaceForensics++ -d NeuralTextures -c c40 -t masks --server EU2