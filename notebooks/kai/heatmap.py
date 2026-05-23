import yaml
import os
os.chdir('/home/ma/ma_ma/ma_kreffert/Interpretable-Deep-Fake-Detection/training/')
import torch
import random
# init seed
# init_seed(config)

torch.manual_seed(34)
random.seed(34)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Disable for strict reproducibility
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
# torch.cuda.reset_peak_memory_stats()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.set_device(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
sys.argv = ["train.py"]
from train import init_seed, prepare_training_data, prepare_testing_data, choose_optimizer, choose_scheduler, choose_metric
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datetime import timedelta
from detectors import DETECTOR
from trainer.trainer import Trainer
# from test import test_epoch, test_one_dataset, test_epoch, inference
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import numpy as np

import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from tqdm import tqdm
from bcos.interpretability import grad_to_img, to_numpy
import ipywidgets as widgets


def load_config(path, additional_args = {}):
    # Parse options and load config
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:  # KAI: Added this to ensure it finds the config file
        with open('./training/config/train_config.yaml', 'r') as f:
            config2 = yaml.safe_load(f)
    except FileNotFoundError:
        with open(os.path.expanduser('~/Interpretable-Deep-Fake-Detection/training/config/train_config.yaml'), 'r') as f:
            config2 = yaml.safe_load(f)
    
    # Update label_dict if it exists in the config
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    
    # Update config with the new values from config2
    config.update(config2)
    
    # If dry_run, set specific values to avoid long training runs
    if config.get('dry_run', False):
        config['nEpochs'] = 0
        config['save_feat'] = False
    
    # Update nested dictionary (like backbone_config) without overwriting entire dictionary
    for key, value in additional_args.items():
        if key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                # If both the current config value and the additional_args value are dictionaries, merge them
                deep_update(config[key], value)
            else:
                # Otherwise, just overwrite the key in the config
                config[key] = value
        else:
            config[key] = value
    
    return config


def create_heatmap_visualization(test_data_loaders, configs, threshold=0.1, num_samples=6, random_seed=0):
    """
    Creates a visualization grid with original images and heatmaps from different models.
    
    Args:
        test_data_loaders: Dictionary of test data loaders
        configs: List of model configurations with different 'b' values
        threshold: Threshold value for binarizing heatmaps (default: 0.1)
        num_samples: Number of image samples to display (default: 6)
    
    Returns:
        fig: The matplotlib figure object
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Number of models to display
    num_models = len(configs)
    
    # Create figure with 2 rows per sample (original and thresholded)
    # and columns for original image + each model's heatmap
    fig, axes = plt.subplots(num_samples * 2, num_models + 1, 
                            figsize=(3.5 * (num_models + 1), 3 * num_samples * 2))
    
    # Choose a specific dataset (FaceForensics++ in this case)
    dataset_key = 'FaceForensics++'

    # Add column labels at the top of the figure
    for model_idx, config in enumerate(configs):
        col_idx = model_idx + 1

    for model_idx, config in enumerate(configs):
        sample_count = 0
        col_idx = model_idx + 1
        # Prepare the model (detector)
        model_class = DETECTOR[config['model_name']]
        model = model_class(config)
        state_dict = torch.load(config['pretrained'])
        
        # Remove "module." prefix if present in the state_dict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        model.to(device)
        if model_idx == 0:
            # Create a list to store sampled batch indices and sample indices
            sampled_indices = []
        # Loop through data loaders and process the first batch
        for i, data_dict in tqdm(enumerate(test_data_loaders[dataset_key]), total=min(num_samples, len(test_data_loaders[dataset_key]))):
            if i >= num_samples:
                break
            # Process data and ensure labels are binary (0 or 1)
            img_batch, label_batch, landmark, mask = (data_dict[k] for k in ['image', 'label', 'landmark', 'mask'])
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')
            data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)

            for key in data_dict.keys():
                if data_dict[key] is not None:
                    data_dict[key] = data_dict[key].to(device)


            # For the first model, randomly select an image from the batch and store the index
            # For subsequent models, use the same indices for consistency
            batch_size = img_batch.size(0)
            
            if model_idx == 0:
                if batch_size > 1:
                    # Randomly sample an image from the batch
                    sample_idx = np.random.randint(0, batch_size)
                else:
                    # Use the first image if not sampling or batch size is 1
                    sample_idx = 0
                # Store this index for consistent use across models
                sampled_indices.append(sample_idx)
            else:
                # Use the same sample index as the first model
                sample_idx = sampled_indices[i]

            # Take the first image and process it
            img = img_batch[sample_idx].unsqueeze(0).to(device)
            label = label_batch[sample_idx].item()  # Get the scalar value

            # Generate explanation
            model.backbone.eval()
            explanation = model.backbone.explain(img)

            # Get explanation map
            explanation_map = explanation['explanation'].copy()
            
            # Create thresholded version
            thresholded_map = explanation_map.copy()
            thresholded_map[:, :, -1] = (thresholded_map[:, :, -1] > threshold).astype(np.uint8)
            img_np = np.array(to_numpy(img[0, [0, 1, 2]].permute(1, 2, 0)) * 255, dtype=np.uint8)

            # Row indices for this sample
            original_row = i * 2
            threshold_row = i * 2 + 1
            
            # Display original image in first column of both rows
            for row in [original_row, threshold_row]:
                axes[row, 0].imshow(img_np)
                # Remove titles and add text annotations instead
                # Add label as text annotation instead of title
                axes[row, 0].text(0.5, 0.02, f"True label: {label}", 
                                transform=axes[row, 0].transAxes,
                                color='white', fontsize=10, ha='center', va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7))
                axes[row, 0].set_xticks([])
                axes[row, 0].set_yticks([])
                
                # Add row label on the left
                if model_idx == 0:
                    row_label = "Original Heatmap" if row == original_row else f"Thresholded to >{threshold}"
                    axes[row, 0].text(-0.15, 0.5, row_label, 
                                    transform=axes[row, 0].transAxes, 
                                    fontsize=12, rotation=90, va='center', ha='center')

            # Plot original heatmap (top row)
            im = axes[original_row, col_idx].imshow(explanation_map, 
                                               cmap='jet', 
                                               alpha=0.5, 
                                               extent=(0, config['resolution'], 0, config['resolution']))
            
            # Add prediction as text annotation instead of title
            pred_value = explanation['prediction']
            pred_color = 'green' if (pred_value > 0.5 and label == 1) or (pred_value <= 0.5 and label == 0) else 'red'
            
            axes[original_row, col_idx].text(0.5, 0.02, f"Pred: {pred_value:.2f}", 
                                         transform=axes[original_row, col_idx].transAxes,
                                         color='white', fontsize=10, ha='center', va='bottom',
                                         bbox=dict(boxstyle="round,pad=0.3", fc=pred_color, alpha=0.7))
            if i==0:
                title = f"b={config['backbone_config']['b']}"
            else:
                title = f""
            axes[original_row, col_idx].set_title(title)
            axes[original_row, col_idx].set_xticks([])
            axes[original_row, col_idx].set_yticks([])
            
            # Plot thresholded heatmap (bottom row)
            im = axes[threshold_row, col_idx].imshow(thresholded_map, 
                                                cmap='jet', 
                                                alpha=0.5, 
                                                extent=(0, config['resolution'], 0, config['resolution']))
            
            axes[threshold_row, col_idx].set_xticks([])
            axes[threshold_row, col_idx].set_yticks([])
    
    # Improve spacing
    plt.subplots_adjust(wspace=0.1, hspace=0.15, top=0.95, bottom=0.05, left=0.07, right=0.95)
    
    return fig


def init():
    base_path = '/pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/weights/best_weights/'
    pretrained_paths = [f'{base_path}resnet34.pth',
                        f'{base_path}resnet34_bcos_1_25.pth',
                    f'{base_path}resnet34_bcos_2_5.pth',
                    f'{base_path}vit_bcos_1_25.pth',
                    f'{base_path}vit_bcos_1_75.pth',
                    f'{base_path}vit_bcos_2_5.pth',
                    ]
    # ------------------   Resnets
    # Basis Resnet -> not used
    resnet34_args = {'test_batchSize': 8,
                    'pretrained': pretrained_paths[0],
                    }
    path = "/home/ma/ma_ma/ma_kreffert/Interpretable-Deep-Fake-Detection/BWCluster/test/resnet34_best_hpo.yaml"
    config_resnet34 = load_config(path, additional_args=resnet34_args)

    # Resnet34 with b 1_25 -> also adjusts the compression, since it is used for the test data loaders
    resnet34_1_25_args = {'test_batchSize': 8,
                    'pretrained': pretrained_paths[1],
                        'compression': 'c23',
                    # 'backbone_config':{'b': 1.25}
                    }
    path = "/home/ma/ma_ma/ma_kreffert/Interpretable-Deep-Fake-Detection/BWCluster/test/resnet34_bcos_v2_1_25_best_hpo.yaml"
    config_resnet34_1_25 = load_config(path, additional_args=resnet34_1_25_args)

    # Resnet34 with b 2_5
    resnet34_2_5_args = {'test_batchSize': 8,
                    'pretrained': pretrained_paths[2],
                    # 'backbone_config':{'b': 1.25}
                    }
    path = "/home/ma/ma_ma/ma_kreffert/Interpretable-Deep-Fake-Detection/BWCluster/test/resnet34_bcos_v2_2_5_best_hpo.yaml"
    config_resnet34_2_5 = load_config(path, additional_args=resnet34_2_5_args)

    configs = [config_resnet34, config_resnet34_1_25, config_resnet34_2_5]

    # ---------------------  vit
    # ViT with b 1_25
    vit_1_25_args = {'test_batchSize': 8,
                    'pretrained': pretrained_paths[3],
                    # 'backbone_config':{'b': 1.25}
                    }
    path = "/home/ma/ma_ma/ma_kreffert/Interpretable-Deep-Fake-Detection/BWCluster/test/vit_bcos_1_25_best_hpo.yaml"
    config_vit_1_25 = load_config(path, additional_args=vit_1_25_args)

    # ViT with b 1_75
    vit_1_75_args = {'test_batchSize': 8,
                    'pretrained': pretrained_paths[4],
                    # 'backbone_config':{'b': 1.25}
                    }
    path = "/home/ma/ma_ma/ma_kreffert/Interpretable-Deep-Fake-Detection/BWCluster/test/vit_bcos_1_75_best_hpo.yaml"
    config_vit_1_75 = load_config(path, additional_args=vit_1_75_args)

    # ViT with b 2_5
    vit_2_5_args = {'test_batchSize': 8,
                    'pretrained': pretrained_paths[5],
                    # 'backbone_config':{'b': 1.25}
                    }
    path = "/home/ma/ma_ma/ma_kreffert/Interpretable-Deep-Fake-Detection/BWCluster/test/vit_bcos_2_5_best_hpo.yaml"
    config_vit_2_5 = load_config(path, additional_args=vit_2_5_args)

    vit_configs = [config_vit_1_25, config_vit_1_75, config_vit_2_5]

    # loading the test data
    test_data_loaders = prepare_testing_data(configs[1])
    return configs, vit_configs, test_data_loaders


def main():
    configs, vit_configs, test_data_loaders = init()
    fig = create_heatmap_visualization(test_data_loaders, configs[1:], threshold=0.5, num_samples=6, random_seed=1)
    plt.savefig('/home/ma/ma_ma/ma_kreffert/Interpretable-Deep-Fake-Detection/notebooks/kai/resnet_heatmap_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    # test_data_loaders = prepare_testing_data(configs[1])

    # Example usage:
    fig = create_heatmap_visualization(test_data_loaders, vit_configs, threshold=0.5, num_samples=6, random_seed=1)
    plt.savefig('/home/ma/ma_ma/ma_kreffert/Interpretable-Deep-Fake-Detection/notebooks/kai/vit_heatmap_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    main()