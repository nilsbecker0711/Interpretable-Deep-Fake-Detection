# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from metrics.utils import get_test_metrics

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_MAIN_PROCESS = not dist.is_initialized() or dist.get_rank() == 0

import wandb
os.environ["WANDB_API_KEY"] = "bcd0e878ee944f48096df279bea051e62defbb36"

class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time_val = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.best_metrics_all_time_test = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()  # move model to GPU
        self.logger.info(f'Running model on {self.model.device}, since device={device} available.')
        
        timestamp = datetime.datetime.now().strftime("%b_%d_%H_%M")  # Format: Month_Day_Hour_Minute
        # if self.config['ddp'] == False:
        #     wandb.init(project="deepfake_training", name=f"{config['model_name']}_{timestamp}", config=self.config)
        # elif IS_MAIN_PROCESS:
        #     wandb.init(project="deepfake_training",  
        #     group=f"DDP_{config['model_name']}_{timestamp}",
        #     name=f"{config['model_name']}_{timestamp}_rank_{dist.get_rank()}" if dist.is_initialized() else "single_process",
        #     config=self.config)

        
        
        # get current time
        self.timenow = time_now
        # create directory path
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

    def print_gradient_stats(self, model):
        """
        Prints the mean and standard deviation of gradients for all parameters
        that require gradients. tilo: for debugging the gradient flow
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                print(f"{name}: grad mean = {grad_mean:.6f}, grad std = {grad_std:.6f}")

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]


    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            self.logger.info(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']], find_unused_parameters=False, output_device=self.config['local_rank'])
            #self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key,ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")


    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self,data_dict):
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()

                # For Distributed Data Parallel, print only on one process (e.g., rank 0)
                """ if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        print("Gradient stats for backbone:")
                        if isinstance(self.model, DDP):
                            self.print_gradient_stats(self.model.module.backbone)
                        else:
                            self.print_gradient_stats(self.model.backbone)
                else:
                    print("Gradient stats for backbone:")
                    if isinstance(self.model, DDP):
                        self.print_gradient_stats(self.model.module.backbone)
                    else:
                        self.print_gradient_stats(self.model.backbone) """

                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:
            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            self.optimizer.zero_grad()

            losses['overall'].backward()

            # tilo: apply gradient clipping to migitate exploding gradient problem
            #clip_grad_norm_(self.model.parameters(), max_norm=8.0)
            #print("Gradients have been clipped successfully!")

            # For Distributed Data Parallel, print only on one process (e.g., rank 0)
            """  if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    print("Gradient stats for backbone:")
                    if isinstance(self.model, DDP):
                        self.print_gradient_stats(self.model.module.backbone)
                    else:
                        self.print_gradient_stats(self.model.backbone)
            else:
                print("Gradient stats for backbone:")
                if isinstance(self.model, DDP):
                    self.print_gradient_stats(self.model.module.backbone)
                else:
                    self.print_gradient_stats(self.model.backbone) """

            self.optimizer.step()
            return losses, predictions


    def train_epoch(
        self,
        epoch,
        train_data_loader,
        test_data_loaders=None,
        val_data_loaders=None
        ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))
        if epoch>=1:
            times_per_epoch = 2
        else:
            times_per_epoch = 1


        #times_per_epoch=4

        test_step = len(train_data_loader) // times_per_epoch    # test 10 times per epoch
        step_cnt = epoch * len(train_data_loader)

        # save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=len(train_data_loader)):
            self.setTrain()
            if torch.cuda.is_available():
                # more elegant and more scalable way of moving data to GPU
                for key in data_dict.keys():
                    if data_dict[key]!=None and key!='name':
                        data_dict[key]=data_dict[key].cuda()

            losses, predictions = self.train_step(data_dict)

            # compute the gradient norms
            if (iteration in [0, 1, 2, 3, 4, 6]) or (iteration % 300 == 0):  # Log gradient norms every _ iterations (or adjust as needed)
                grads = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grads.append(grad_norm)

                # Compute statistics
                grad_norms = torch.tensor(grads)
                grad_norm_min = grad_norms.min().item()
                grad_norm_max = grad_norms.max().item()
                grad_norm_mean = grad_norms.mean().item()
                grad_norm_std = grad_norms.std().item()

                logits = predictions['cls']
                class_shares = (predictions['prob'] >= 0.5).float().mean()

                # Collect all parameters into a single list
                all_params = []
                for param in self.model.parameters():
                    if param.requires_grad:
                        all_params.append(param.view(-1))  # Flatten each parameter

                # Concatenate all parameters into a single tensor
                all_params = torch.cat(all_params)

                # Calculate aggregate statistics
                param_min = all_params.min().item()
                param_max = all_params.max().item()
                param_mean = all_params.mean().item()
                param_std = all_params.std().item()

                lr = self.optimizer.param_groups[0]['lr']  # assuming you have a single optimizer

                weight_decay = self.optimizer.param_groups[0].get('weight_decay', 0)

                # Log to WandB
                wandb.log({
                    'other/grad_norm_min': grad_norm_min,
                    'other/grad_norm_max': grad_norm_max,
                    'other/grad_norm_mean': grad_norm_mean,
                    'other/grad_norm_std': grad_norm_std,
                    'other/class_shares':class_shares,
                    # 'other/class_counts': class_counts,
                    'other/param_min': param_min,
                    'other/param_max': param_max,
                    'other/param_mean': param_mean,
                    'other/param_std': param_std,
                    'other/learning_rate': lr,
                    'other/weight_decay':weight_decay,
                    'step': step_cnt
                })

            # update learning rate
            if 'SWA' in self.config and self.config['SWA'] and epoch>self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if iteration % 300 == 0 and self.config['local_rank']==0:
                if self.config['SWA'] and (epoch>self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                    # also log to wandb
                    wandb.log({f'train_loss/{k}': v_avg, 'step': step_cnt})
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                    # also log to wandb
                    wandb.log({f'train_metric/{k}': v_avg, 'step': step_cnt})
                self.logger.info(metric_str)



                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()
            
            # run test
            if (step_cnt+1) % test_step == 0:
                if val_data_loaders is not None and (not self.config['ddp'] ):
                    self.logger.info("===> Val start!")
                    val_best_metric = self.val_epoch(
                        epoch,
                        iteration,
                        val_data_loaders,
                        step_cnt,
                    )
                elif val_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                    self.logger.info("===> Val start!")
                    val_best_metric = self.val_epoch(
                        epoch,
                        iteration,
                        val_data_loaders,
                        step_cnt,
                    )
                else:
                    val_best_metric = None

            # run test
            if (step_cnt+1) % test_step == 0:
                if test_data_loaders is not None and (not self.config['ddp'] ):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                elif test_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                else:
                    test_best_metric = None

                    # total_end_time = time.time()
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
            step_cnt += 1
        if val_data_loaders is not None:
            return val_best_metric, test_best_metric
        return test_best_metric

    def get_respect_acc(self,prob,label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        # Precision, Recall, F1-score
        precision = precision_score(label, pred, zero_division=0)
        recall = recall_score(label, pred, zero_division=0)
        f1 = f1_score(label, pred, zero_division=0)
        return acc_real, acc_fake, precision, recall, f1

    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists=[]
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader),total=len(data_loader)):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key]!=None:
                    data_dict[key]=data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

    def save_best(self,epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset, mode='test'):
        if mode == 'val':
            best_metric = self.best_metrics_all_time_val[key].get(self.metric_scoring,
                                                            float('-inf') if self.metric_scoring != 'eer' else float(
                                                                'inf'))
        else:
            best_metric = self.best_metrics_all_time_test[key].get(self.metric_scoring,
                                                            float('-inf') if self.metric_scoring != 'eer' else float(
                                                                'inf'))
            # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            if mode == 'val':
                # Update the best metric
                self.best_metrics_all_time_val[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
                if key == 'avg':
                    self.best_metrics_all_time_val[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            else:
                # Update the best metric
                self.best_metrics_all_time_test[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
                if key == 'avg':
                    self.best_metrics_all_time_test[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt(f'{mode}', key, f"{epoch}+{iteration}")
            self.save_metrics(f'{mode}', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer(f'{mode}', key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'{mode}_losses/{k}', v_avg, global_step=step)
                # also log to wandb
                wandb.log({f'{mode}_losses/{k}': v_avg, 'step': step})
                loss_str += f"{mode}-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k=='dataset_dict':
                continue
            metric_str += f"{mode}-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer(f'{mode}', key, k)
            writer.add_scalar(f'{mode}_metrics/{k}', v, global_step=step)
            # also log to wandb
            wandb.log({f'{mode}_metrics/{k}': v, 'step': step})
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake, precision, recall, f1 = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'{mode}ing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'{mode}_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'{mode}_metrics/acc_fake', acc_fake, global_step=step)
            writer.add_scalar(f'{mode}_metrics/precision', precision, global_step=step)
            writer.add_scalar(f'{mode}_metrics/recall', recall, global_step=step)
            writer.add_scalar(f'{mode}_metrics/f1', f1, global_step=step)
            
            # also log to wandb
            wandb.log({f'{mode}_metrics/acc_real': acc_real, 'step': step})
            wandb.log({f'{mode}_metrics/acc_fake': acc_fake, 'step': step})
            wandb.log({f'{mode}_metrics/precision': precision, 'step': step})
            wandb.log({f'{mode}_metrics/recall': recall, 'step': step})
            wandb.log({f'{mode}_metrics/f1': f1, 'step': step})
        self.logger.info(metric_str)


    def val_epoch(self, epoch, iteration, val_data_loaders, step):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        # testing for all test data
        keys = val_data_loaders.keys()
        for key in keys:
            # save the testing data_dict
            data_dict = val_data_loaders[key].dataset.data_dict
            self.save_data_dict('val', data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(val_data_loaders[key])
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=data_dict['image'])
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"validation-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset, mode='val')

        if len(keys)>0 and self.config.get('save_avg',False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric, mode='val')

        self.logger.info('===> Val Done!')
        return self.best_metrics_all_time_val  # return all types of mean metrics for determining the best ckpt


    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(test_data_loaders[key])
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=data_dict['image'])
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset, mode='test')

        if len(keys)>0 and self.config.get('save_avg',False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric, mode='test')

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time_test  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions
