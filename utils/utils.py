import torch
from torch import distributed as dist, nn as nn

import libs.utils as other_utils
from libs.nn.customize import SegmentationBCELossNoReduce, SegmentationBCELoss, SegmentationBCENullSecondStage
from loggers.file_loggers import RecentModelTracker, BestModelTracker
from loggers.local_loggers import SimpleLoggerExample
from loggers.wandb_loggers import WandbSimplePrinter, WandbSummaryPrinter
from setup import setup_wandb
from torch.distributed.optim import ZeroRedundancyOptimizer


def create_schedulers(optimizers, mode, num_epochs, iters_per_epoch, min_lr=0):
    schedulers = {}
    for name, optimizer in optimizers.items():
        schedulers[name] = other_utils.LR_Scheduler(optimizer, mode, num_epochs, iters_per_epoch, min_lr=min_lr)
    return schedulers


def create_val_loggers(configs, export_root, key='val/') -> list:
    metric_printer = WandbSimplePrinter(key) if configs.use_wandb else SimpleLoggerExample(key, export_root)
    loggers = [metric_printer, RecentModelTracker(export_root, log_start=configs.log_start)]
    metric_keys = []
    for metric_key in list(configs.best_model_keys):
        for d_name in configs.eval_dataset_names:
            filename = d_name + "_" + metric_key + "_best.pth"
            log_metric_key = d_name + "_" + metric_key
            metric_keys.append(log_metric_key)
            loggers.append(BestModelTracker(export_root, metric_key=log_metric_key, ckpt_filename=filename))
    loggers.append(WandbSummaryPrinter(prefix="best_", summary_keys=metric_keys)) if configs.use_wandb else None
    return loggers


def create_optimizers(model, configs) -> dict:
    """ Create optimizers for each part of the segmentation network"""

    optimizers = {}
    optimizer_type = configs.get('optimizer', 'SGD')
    if optimizer_type == 'AdamW':
        optimizers['pretrained'] = torch.optim.AdamW(model.parameters(), lr=configs.lr,
                                                     weight_decay=configs.weight_decay)
        # optimizers['pretrained'] = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW,
        #                                                    lr=configs.lr, weight_decay=configs.weight_decay)
        # print(f"Using AdamW with lr (ZeroRedundancyOptimizer): {configs.lr}")
        print(f"Using AdamW with lr: {configs.lr}")
    else:
        optimizers['pretrained'] = torch.optim.SGD(model.parameters(), lr=configs.lr, momentum=configs.momentum,
                                                   weight_decay=configs.weight_decay)
        print(f"Using SGD with lr: {configs.lr}")
    return optimizers


def resume_training_setup(configs, model, optimizers: dict, load_optimizer=True) -> int:
    print("Resume training")
    ckpt = torch.load(configs.ckpt_path, map_location='cpu')

    curr_epoch = int(ckpt['step']) + 1

    # Load model
    model_sd = ckpt['model_state_dict']['segmentation']
    new_sd = {}
    for key, val in model_sd.items():
        if key.startswith("module"):
            new_sd[key[7:]] = val
        else:
            new_sd[key] = val
    model.load_state_dict(new_sd)
    print("Model state dict loaded")

    if load_optimizer:
        for optimizer_name, opt in optimizers.items():
            opt.load_state_dict(ckpt['optimizer_state_dict'][optimizer_name])
            print("{} optimizer state dict loaded".format(optimizer_name))
            print("{} LR resuming at: {}".format(optimizer_name, opt.param_groups[0]['lr']))

    print("Resuming from step {}".format(ckpt['step']))

    return curr_epoch

def resume_training_setup_v2(configs, model, schedulers, start_step):
    print("Resume training V2")
    ckpt = torch.load(configs.ckpt_path, map_location='cpu')

    curr_epoch = start_step

    # Load model
    model_sd = ckpt['model_state_dict']['segmentation']
    new_sd = {}
    for key, val in model_sd.items():
        if key.startswith("module"):
            new_sd[key[7:]] = val
        else:
            new_sd[key] = val
    model.load_state_dict(new_sd)
    print("Model state dict loaded")

    start_lr = 0.
    for _ in range(curr_epoch):
        for scheduler in schedulers.values():
            scheduler(0, curr_epoch)
            start_lr = scheduler.get_current_lr()
    print(f"Start LR: {start_lr}")

    print("Resuming from step {}".format(start_step))

    return curr_epoch

def setup_distributed_experiment(configs, gpu, is_distributed_training):
    if is_distributed_training:
        rank = configs.nr * configs.num_gpus + gpu
        is_main_process = False
        dist.init_process_group(backend=configs.backend, init_method="env://", world_size=configs.world_size, rank=rank)
        if dist.get_rank() == 0:
            is_main_process = True
            if configs.use_wandb:
                setup_wandb(configs)

        torch.cuda.set_device(gpu)
    else:
        if configs.use_wandb:
            setup_wandb(configs)
        is_main_process = True
    return is_main_process


def get_loss_fn(configs):
    if configs.method == "unified" or configs.method == "single":
        return nn.CrossEntropyLoss(ignore_index=-1)
    else:
        if configs.bcenull_v2_loss:
            print("Loading V2 BCE Null loss (for second stage)")
            return SegmentationBCENullSecondStage()
        if configs.bce_no_reduce:
            print("loading with no reduction loss")
            return SegmentationBCELossNoReduce()
        else:
            return SegmentationBCELoss()
