# Spot the Fake B-cos You Can?

This repository provides an interpretable deepfake detection pipeline. The models are designed to be flexible and modular, consisting of a backbone and a detector. Model training is done via shell scripts located in the `BWCluster` folder, and the configuration parameters for each model are specified in YAML files.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Model Training](#model-training)
  - [YAML Configuration Files](#yaml-configuration-files)
  - [Shell Scripts in `BWCluster`](#shell-scripts-in-bwcluster)
  - [Example of Running Training](#example-of-running-training)
- [Getting Started](#getting-started)
- [License](#license)

## Pipeline Overview

The pipeline for training deepfake detection models is structured around two key components for each network:

1. **Backbone**: The backbone is the core feature extractor of the model, typically based on deep learning architectures such as CNNs (e.g., ResNet) or Vision Transformers (ViT). It processes the input data (images or videos) to extract meaningful features that will be used for detection.

2. **Detector**: The detector takes the extracted features from the backbone and makes predictions, such as classifying an image or video as real or fake. It is customizable to meet the specific task requirements.

## Model Training

Training the models is done through shell scripts in the `BWCluster` folder. Each shell script is designed to handle the training process, including setting up the environment, loading the YAML configuration files, and starting the training job on the cluster. 

### YAML Configuration Files

Each network (model architecture) is linked to a YAML configuration file that specifies various parameters. These parameters include:

- **Backbone architecture** (e.g., ResNet, ViT, etc.)
- **Detector configuration** (e.g., number of layers, activation functions)
- **Training hyperparameters** (e.g., learning rate, batch size, number of epochs)

The YAML files allow for flexible and easy customization of the training parameters. They can be adjusted without modifying the core code, making it simpler to experiment with different settings.

### Shell Scripts in `BWCluster`

The shell scripts in the `BWCluster` folder automate the training process. They handle the following tasks:

1. **Environment Setup**: Ensure that all dependencies are installed and the environment is properly configured.
2. **Model Configuration**: Load the appropriate YAML configuration file for the specific model and its parameters.
3. **Training Execution**: Start the training process on the specified hardware (e.g., GPUs or CPUs) by running the training loop.

### Example of Running Training

To train a model, follow these steps:

1. **Choose the Model**: Select the model architecture (e.g., ResNet backbone with a custom detector).
2. **Modify the YAML Configuration**: Edit the relevant YAML file to adjust the parameters such as batch size, learning rate, etc.
3. **Run the Training Script**: Navigate to the `BWCluster` directory and execute the corresponding shell script:
   ```bash
   cd BWCluster
   bash train_model.sh --config /path/to/config.yaml


### Example 2.0

#### Training

python training/train.py \
  --detector_path training/config/detector/resnet34.yaml

#### Testing

python training/test.py \
  --detector_path training/config/detector/resnet34.yaml \
  --weights_path logs/training/resnet34_2026-07-28-22-22-48/val/avg/ckpt_best.pth \
  --test_dataset FaceForensics++ Celeb-DF-v1 Celeb-DF-v2 DFDCP DFDC UADFV \
  simswap_ff inswap_ff fsgan_ff blendface_ff e4s_cdf danet_cdf


python training/test.py \
  --detector_path training/config/detector/xception_bcos_b1_75.yaml \
  --weights_path  logs/training/xception_bcos_detector_b1_75_2026-08-15-13-58-12/val/avg/ckpt_best.pth \
  --test_dataset FaceForensics++ Celeb-DF-v1 Celeb-DF-v2 DFDCP UADFV 

#### GPG

###### shared grids

python notebooks/Linus/GridPointingGame/GPG_eval.py \
  --model-config training/config/detector/resnet34_bcos_v2.yaml \
  --test-config results/test_bcos_res_2_config.yaml \
  --weights logs/training/resnet34_bcos_v2_2026-07-29-20-50-31_b2/val/avg/ckpt_best.pth \
  --xai-method bcos --split test \
  --grid-dir results/GPG_assets/shared_per_dataset/FaceForensics++_test_256/3x3 \
  --output-dir results/eval/gpg_perdataset_bcos_b2 \
  --set 'backbone_config={"b": 2.0}' \
  --set dataset_json_folder=preprocessing/dataset_json_v3 \
  --set 'test_dataset=[FaceForensics++]'


##### highest confidence per dataset

python notebooks/Linus/GridPointingGame/GPG_eval.py \
  --model-config training/config/detector/resnet34_bcos_v2.yaml \
  --test-config results/test_bcos_res_2_config.yaml \
  --weights logs/training/resnet34_bcos_v2_2026-07-29-20-50-31_b2/val/avg/ckpt_best.pth \
  --xai-method bcos --split test \
  --selection confidence --real-selection confident --dataset-mixing single \
  --output-dir results/eval/gpg_conf_perdataset_bcos_b2 \
  --set 'backbone_config={"b": 2.0}' \
  --set dataset_json_folder=preprocessing/dataset_json_v3 \
  --set 'test_dataset=[FaceForensics++, Celeb-DF-v1, Celeb-DF-v2, DFDCP, DFDC, UADFV]' \
  --set max_grids=500



#### MPG on shared assets

python notebooks/Linus/GridPointingGame/MPG_eval.py \
  --model-config training/config/detector/resnet34_bcos_v2.yaml \
  --test-config results/test_MPG_bcos_2_5.yaml \
  --weights logs/training/resnet34_bcos_v2_2026-07-29-20-50-31_b2/val/avg/ckpt_best.pth \
  --xai-method bcos \
  --image-list results/MPG_assets/shared_random/FaceForensics++_test/images.json \
  --output-dir results/eval/mpg_bcos_b2_bcos \
  --batch-size 8 \
  --set 'backbone_config={"b": 2.0}' \
  --set mask_resolution=256 \
  --set with_mask=true \
  --set dataset_json_folder=preprocessing/dataset_json_v3 \
  --set 'test_dataset=[FaceForensics++]' # specifiy datasets here


#### GPG asset creation

cd $REPO   # only training/config/test_config yaml's rgb_dir needs to point here

# test split, 500 grids
python notebooks/Linus/GridPointingGame/GPG_eval.py --grids-only --selection random \
  --split test --set resolution=256 --set max_grids=500
python notebooks/Linus/GridPointingGame/GPG_eval.py --grids-only --selection random \
  --split test --set resolution=224 --set max_grids=500

# val split (monitoring grids), 100 each
python notebooks/Linus/GridPointingGame/GPG_eval.py --grids-only --selection random \
  --split val --set resolution=256 --set max_grids=100
python notebooks/Linus/GridPointingGame/GPG_eval.py --grids-only --selection random \
  --split val --set resolution=224 --set max_grids=100

