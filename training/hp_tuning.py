import os
import argparse
import datetime
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from datetime import timedelta
from logger import create_logger, RankFilter
from trainer.trainer import Trainer
from detectors import DETECTOR
from train import (
    init_seed,
    prepare_training_data,
    prepare_testing_data,
    choose_metric,
    choose_optimizer,
    choose_scheduler
)

# Argument Parsing
parser = argparse.ArgumentParser(description="Process some paths.")
parser.add_argument("--detector_path", type=str, default="/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml", help="path to detector YAML file")
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument("--no-save_ckpt", dest="save_ckpt", action="store_false", default=True)
parser.add_argument("--no-save_feat", dest="save_feat", action="store_false", default=True)
parser.add_argument("--ddp", action="store_true", default=False)
parser.add_argument("--task_target", type=str, default="", help="specify the target of current training task")
args = parser.parse_args()

# Distributed Training Setup
local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Default to 0 if not set
args.local_rank = local_rank
if torch.cuda.is_available():
    torch.cuda.set_device(args.local_rank)

os.environ["WANDB_API_KEY"] = "bcd0e878ee944f48096df279bea051e62defbb36"

def load_config():
    """Loads and merges training configurations."""
    with open(args.detector_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        with open("./training/config/train_config.yaml", "r") as f:
            config2 = yaml.safe_load(f)
    except FileNotFoundError:
        with open(os.path.expanduser("~/Interpretable-Deep-Fake-Detection/training/config/train_config.yaml"), "r") as f:
            config2 = yaml.safe_load(f)

    if "label_dict" in config:
        config2["label_dict"] = config["label_dict"]
    config.update(config2)
    
    # Argument Overrides
    config["local_rank"] = args.local_rank
    if args.train_dataset:
        config["train_dataset"] = args.train_dataset
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset
    config["save_ckpt"] = args.save_ckpt
    config["save_feat"] = args.save_feat
    
    return config

def set_wandb_config(config):
    """Overrides config values with hyperparameters from W&B Sweeps."""
    wandb_config = wandb.config  # Fetch hyperparameters from sweep
    print(wandb_config)

    # general parameters
    #config['freeze'] = wandb_config.freeze
    config['train_batchSize'] = wandb_config.batchSize
    config['test_batchSize'] = wandb_config.batchSize
    config['nEpochs'] = wandb_config.nEpochs
    config['manualSeed'] = wandb_config.manualSeed

    # optimizer
    optimizer_type = wandb_config.optimizer
    config['optimizer']['type'] = wandb_config.optimizer
    config['optimizer'][optimizer_type]['lr'] = wandb_config.lr
    config['optimizer'][optimizer_type]['weight_decay'] = wandb_config.weight_decay
    if optimizer_type == 'adam':
        config['optimizer'][optimizer_type]['beta1'] = wandb_config.beta1
        config['optimizer'][optimizer_type]['beta2'] = wandb_config.beta2
        config['optimizer'][optimizer_type]['eps'] = wandb_config.eps
        config['optimizer'][optimizer_type]['amsgrad'] = wandb_config.amsgrad
    elif optimizer_type == 'sgd':
        config['optimizer'][optimizer_type]['momentum'] = wandb_config.momentum
    
    # lr scheduler
    lr_scheduler = wandb_config.lr_scheduler
    config['lr_scheduler'] = wandb_config.lr_scheduler
    if lr_scheduler == 'step':
        config['lr_step'] = wandb_config.lr_step
        config['lr_gamma'] = wandb_config.lr_gamma
    elif  lr_scheduler in ['warmup', 'cosine']:
        config['lr_T_max'] = wandb_config.lr_T_max
        config['lr_eta_min'] = wandb_config.lr_eta_min

    # model parameters
    #config['backbone_config']['log_temperature'] = wandb_config.log_temperature
    #config['backbone_config']['bias'] = wandb_config.bias
    config['backbone_config']['b'] = wandb_config.b
    config['backbone_config']['stochastic_depth_prob'] = wandb_config.stochastic_depth_prob
     
    config["use_data_augmentation"] = wandb_config.use_data_augmentation
    return config


def broadcast_config(config):
    """Broadcast config from rank 0 to all other processes."""
    if dist.is_initialized():
        for key, value in config.items():
            if isinstance(value, (int, float)):  
                value_tensor = torch.tensor(value, dtype=torch.float32).cuda()
                dist.broadcast(value_tensor, src=0)
                config[key] = value_tensor.item()  
    return config   

def train():
    """Main training function, called per W&B Sweep run."""
    config = load_config()
    
    if config["ddp"]:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        logger.addFilter(RankFilter(0))

    timestamp = datetime.datetime.now().strftime("%b_%d_%H_%M")  # Format: Month_Day_Hour_Minute
    is_main_process = (not config["ddp"]) or (dist.is_initialized() and dist.get_rank() == 0)
    IS_MAIN_PROCESS = not dist.is_initialized() or dist.get_rank() == 0
    if IS_MAIN_PROCESS:  
        print(dist.get_rank())
        print(f"This is the main process.3")
        wandb.init(
            project="deepfake_training",  
            group="HP_tuning",  
            name=f"{config['model_name']}_{timestamp}" if not config["ddp"] else f"{config['model_name']}_{timestamp}_rank_{dist.get_rank()}",
            dir=None
        )
        # Fetch WandB config only in the main process
        config = set_wandb_config(config)

    if config["ddp"]:
        dist.barrier()  # Ensure all processes wait for W&B to initialize
    
    # Now broadcast config from rank 0 to all other processes
    if config["ddp"]:
        config = broadcast_config(config)
    
    # Update config with WandB Sweep parameters
    config = set_wandb_config(config)
    
    # Setup logger
    timenow = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    task_str = f"_{config['task_target']}" if config.get("task_target") else ""
    logger_path = os.path.join(config["log_dir"], config["model_name"] + task_str + "_" + timenow)
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, "training.log"))
    logger.info(f"Save log to {logger_path}")

    # Print Configuration (Only rank 0)
    if is_main_process:
        logger.info("--------------- Configuration ---------------")
        for key, value in config.items():
            logger.info(f"{key}: {value}")

    # Initialize Seed
    init_seed(config)

    # CUDNN Benchmarking
    if config["cudnn"]:
        cudnn.benchmark = True

    # Prepare Data Loaders
    train_data_loader = prepare_training_data(config)
    val_data_loaders = prepare_testing_data(config, mode='val')
    test_data_loaders = prepare_testing_data(config, mode='test')

    # Model, Optimizer, Scheduler, Metric
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config)
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)

    # Initialize Trainer (Handles WandB init)
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow)

    # Start Training
    for epoch in range(config["start_epoch"], config["nEpochs"] + 1):
        trainer.model.epoch = epoch
        val_best_metric, test_best_metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            test_data_loaders=test_data_loaders,
            val_data_loaders=val_data_loaders
        )
        if val_best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with val {metric_scoring}: {val_best_metric}!")
        if test_best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {test_best_metric}!")

    logger.info(f"Stop Training on best Validation metric {val_best_metric}")
    logger.info(f"Stop Training on best Testing metric {test_best_metric}")

    # Update SVDD Model
    if "svdd" in config["model_name"]:
        model.update_R(epoch)
    
    if scheduler is not None:
        scheduler.step()

    # Close Writers
    for writer in trainer.writers.values():
        writer.close()

    if is_main_process:
        wandb.finish()  # Mark WandB run as complete

sweep_config = {
    'method': 'random',  # You can also use 'bayes' here
    'metric': {'name': 'val_metrics/auc', 'goal': 'maximize'},  # Set the metric for optimization
    'parameters': {
        # general parameters
        #'freeze':{'values': [True, False]},
        'batchSize': {'values': [16, 32, 64, 128]},
        'nEpochs': {'values': [10, 15, 20, 35]},
        'manualSeed': {'values': [1, 10, 1024],},
        #TODO gradient clipping

        # ------  Optimizer parameters ------
        'optimizer': {
            'values': ['adam', 'sgd'] #TODO sam
        },
        'lr': {
            'min': 2.5e-5,
            'max': 2.5e-3,
            'distribution': 'uniform'  # Uniform distribution for random search
        },
        'weight_decay': {
            'min': 1e-7,
            'max': 1e-3,
            'distribution': 'uniform'  # For weight decay
        },
        # adam parameters
        'beta1': {
            'min': 0.85,
            'max': 0.95,
            'distribution': 'uniform',
            # 'conditions': {'optimizer': 'adam'}  # This will only apply if 'scheduler' is 'step'
        },
        'beta2': {
            'min': 0.8,
            'max': 0.999,
            'distribution': 'uniform',
            # 'conditions': {'optimizer': 'adam'}  # This will only apply if 'scheduler' is 'step'
        },
        'eps': {
            'min': 1e-8,
            'max': 1e-5,
            'distribution': 'uniform',
            # 'conditions': {'optimizer': 'adam'}  # This will only apply if 'scheduler' is 'step'
        },
        'amsgrad': {'values': [True, False],
            # 'conditions': {'optimizer': 'adam'}  # This will only apply if 'scheduler' is 'step'
        },
        # sgd
        'momentum': {
            'min': 0.8,
            'max': 0.99,
            'distribution': 'uniform',
            # 'conditions': {'optimizer': 'adam'}  # This will only apply if 'scheduler' is 'step'
        },


        # ------  Scheduler parameters ------
        'lr_scheduler': {
            'values': [None, 'cosine', 'step', 'linear', 'warmup_cosine']  # Discrete values, choose scheduler type
        },
        # step
        'lr_step': {
            'min': 4,
            'max': 10,
            # 'conditions': {'lr_scheduler': 'step'}  # This will only apply if 'scheduler' is 'step'
        },
        'lr_gamma': {
            'min': 0.1,
            'max': 0.5,
            'distribution': 'uniform',
            # 'conditions': {'lr_scheduler': 'step'}  # This will only apply if 'scheduler' is 'step'
        },
        # cosine
        'lr_T_max': {
            'min': 50,
            'max': 200,
            'distribution': 'uniform',
            # 'conditions': {'lr_scheduler': ['cosine']}  # This will only apply if 'scheduler' is 'cosine'
        },
        'lr_eta_min': {
            'min': 0,
            'max': 0.000000001,
            'distribution': 'uniform',
            # 'conditions': {'lr_scheduler': ['cosine', 'warmup_cosine']}  # This will only apply if 'scheduler' is 'cosine'
        },


        # ------   Model parameters --------
        # 'log_temperature': {
        #     'min': 0,
        #     'max': 0.000000001,
        #     'distribution': 'uniform',
        # },
        # 'bias': {
        #     'values': [[0.5, 0.5], [0.25, 0.75], [0.75, 0.25], [0.1, 0.9]]
        #     },
        'b':{
            'min': 1,
            'max': 2.5,
            'distribution': 'uniform',
        },
        'stochastic_depth_prob':{
            'min': 0.0,
            'max': 0.2,
            'distribution': 'uniform',
        },
        #TODO other resnet sizes


        # --------  Data ---------
        'use_data_augmentation':{
            'values':[True, False],
        },

    }
}


if __name__ == "__main__":
    # dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    # logger.addFilter(RankFilter(0))
    # sweep_id = 'interpretable_deefake_detection/deepfake_training/mlmd0ips' #
    IS_MAIN_PROCESS = not dist.is_initialized() or dist.get_rank() == 0
    if IS_MAIN_PROCESS:
        sweep_id = wandb.sweep(sweep_config, project="deepfake_training")
        wandb.agent(sweep_id, function=train, ) # count=1) -> you can also specify count to only run N combinations
    else:
        train()