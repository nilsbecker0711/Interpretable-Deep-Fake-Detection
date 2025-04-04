import os
# from dataset import *
# from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
import yaml
import torch

def deep_update(d1, d2):
    """Recursively update dictionary d1 with values from d2."""
    for key, value in d2.items():
        if isinstance(value, dict) and key in d1 and isinstance(d1[key], dict):
            # Recursively update nested dictionaries
            deep_update(d1[key], value)
        else:
            # Directly update the key
            d1[key] = value

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
    
# def load_config(path, additional_args = {}):
#     # parse options and load config
#     with open(path, 'r') as f:
#         config = yaml.safe_load(f)
#     try:# KAI: added this, to ensure it finds the config file
#         with open('./training/config/train_config.yaml', 'r') as f:
#             config2 = yaml.safe_load(f)
#     except FileNotFoundError:
#         with open(os.path.expanduser('~/Interpretable-Deep-Fake-Detection/training/config/train_config.yaml'), 'r') as f:
#             config2 = yaml.safe_load(f)
#     if 'label_dict' in config:
#         config2['label_dict']=config['label_dict']
#     config.update(config2)
#     # config['local_rank']=args.local_rank
#     if config['dry_run']:
#         config['nEpochs'] = 0
#         config['save_feat']=False
#     for key, value in additional_args.items():
#         config[key] = value
#     return config


def prepare_testing_data(config):
    '''
    for this function we need a config with the following keys:
        test_dataset
        dataset_type
        test_batchSize
        workers

    for the DeepfakeBcosDataset we need further config keys:
        compression
        frame_num: 
            mode
        video_mode
        clip_size
        lmdb
        lmdb_dir
        (train_dataset)
        resolution
        dataset_json_folder
        label_dict
            [video_info['label']]
        rgb_dir
        with_mask
        with_landmark
        use_data_augmentation
    '''
    
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        if config.get('dataset_type', None) == 'lrl':
            test_set = LRLDataset(
                config=config,
                mode='test',
            )
        elif config.get('dataset_type', None) == 'bcos':
            test_set = DeepfakeBcosDataset(
                    config=config,
                    mode='test',
            )
        else:
            test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='test',
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last = (test_name=='DeepFakeDetection'),
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders