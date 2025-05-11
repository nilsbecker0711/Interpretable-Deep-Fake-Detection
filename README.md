# Interpretable Deep-Fake Detection

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
