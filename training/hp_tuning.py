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
parser.add_argument("--sweep_id", type=str, default="", help="w&b sweep ID")
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

    # General parameters
    general_params = ['batchSize', 'nEpochs', 'manualSeed']
    for param in general_params:
        value = getattr(wandb_config, param, None)
        if value is not None:
            if param == 'batchSize':
                config['train_batchSize'] = value
                config['val_batchSize'] = value
                config['test_batchSize'] = value
            else:
                config[param] = value

    # Optimizer settings
    optimizer_type = wandb_config["optimizer"]
    config["optimizer"]["type"] = optimizer_type
    opt_config = config["optimizer"][optimizer_type]
    for key in ["lr", "weight_decay", "beta1", "beta2", "eps", "amsgrad", "momentum"]:
        if key in wandb_config:
            opt_config[key] = wandb_config[key]

    # Scheduler
    lr_scheduler = getattr(wandb_config, 'lr_scheduler', None)
    if lr_scheduler != "None":
        config['lr_scheduler'] = lr_scheduler
        if lr_scheduler == 'step':
            config['lr_step'] = getattr(wandb_config, 'lr_step', config.get('lr_step'))
            config['lr_gamma'] = getattr(wandb_config, 'lr_gamma', config.get('lr_gamma'))
        elif lr_scheduler in ['warmup', 'cosine', 'warmup_cosine']:
            config['lr_T_max'] = getattr(wandb_config, 'lr_T_max', config.get('lr_T_max'))
            config['lr_eta_min'] = getattr(wandb_config, 'lr_eta_min', config.get('lr_eta_min'))
    else:
        config['lr_scheduler'] = None
    # Data
    config['use_data_augmentation'] = getattr(wandb_config, 'use_data_augmentation', False)
    
    # Backbone/model parameters
    # backbone_params = ['b', 'stochastic_depth_prob']
    # for param in backbone_params:
    #     config['backbone_config'][param] = getattr(wandb_config, param, config['backbone_config'].get(param))
    if "backbone_config" in config:
        for key in config["backbone_config"].keys():
            if key in wandb_config:
                config["backbone_config"][key] = wandb_config[key]
                print(f"Setting parameter {key}")
    
    return config


# def broadcast_config(config):
#     """Broadcast config from rank 0 to all other processes."""
#     if dist.is_initialized():
#         for key, value in config.items():
#             if isinstance(value, (int, float)):  
#                 value_tensor = torch.tensor(value, dtype=torch.float32).cuda()
#                 dist.broadcast(value_tensor, src=0)
#                 config[key] = value_tensor.item()  
#     return config   

def train():
    """Main training function, called per W&B Sweep run."""
    config = load_config()
    
    # if config["ddp"]:
    #     dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    #     logger.addFilter(RankFilter(0))

    timestamp = datetime.datetime.now().strftime("%b_%d_%H_%M")  
    wandb.init(
        project="deepfake_training",  
        group="HP_tuning",  
        name=f"{config['model_name']}_{timestamp}" if not config["ddp"] else f"{config['model_name']}_{timestamp}_rank_{dist.get_rank()}",
        config=config,
        dir=None, 
    )
    config = set_wandb_config(config)
    # Fetch WandB config only in the main process

    # if config["ddp"]:
    #     dist.barrier()  # Ensure all processes wait for W&B to initialize
    
    # Now broadcast config from rank 0 to all other processes
    # if config["ddp"]:
    #     config = broadcast_config(config)
    
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
    # if is_main_process:
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

    wandb.finish()  # Mark WandB run as complete

# run with python ~/Interpretable-Deep-Fake-Detection/training/hp_tuning.py --detector_path ~/Interpretable-Deep-Fake-Detection/training/config/detector/resnet34_bcos_v2.yaml
if __name__ == "__main__":
    if False:

        with open("/home/ma/ma_ma/ma_kreffert/Interpretable-Deep-Fake-Detection/training/hp_tuning/vit.yaml", "r") as f:

            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep_config, project="deepfake_training")
    else: # as soon as you have a sweep in which you want to try out more runs, replace the last sweep_id below
        #sweep_id = wandb.sweep(sweep=sweep_config, project="deepfake_training")
        sweep_id = f'interpretable_deefake_detection/deepfake_training/{args.sweep_id}'

    wandb.agent(sweep_id, function=train) # count=1) -> you can also specify count to only run N combinations