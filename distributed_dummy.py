import os

import hydra
import torch
import torch.multiprocessing as mp
from torch import distributed as dist, nn as nn

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from datasets import get_num_classes, distributed_dataloader_factory
from loggers.local_loggers import SimpleLoggerExample
from loggers.wandb_loggers import WandbSimplePrinter
from models import get_segmentation_model
from setup import setup_experiment_without_wandb
from trainers.bcenull_trainer import BCENullTrainer
from trainers.single_trainer import SingleTrainer
from trainers.unified_classifier_trainer import UnifiedClassifierTrainer
from utils.distributed_utils import find_free_port
from utils.utils import create_val_loggers, create_optimizers, create_schedulers, resume_training_setup, \
    setup_distributed_experiment, get_loss_fn


@hydra.main(config_path='configs/config.yaml', strict=False)
def main(configs):
    num_available_gpus = torch.cuda.device_count()
    configs.world_size = configs.num_gpu_per_node * num_available_gpus
    configs.num_gpus = num_available_gpus
    configs.batch_size_per_gpu = configs.batch_size // num_available_gpus
    configs.num_workers_per_gpu = configs.num_workers // num_available_gpus
    os.environ['MASTER_ADDR'] = str(configs.master_address)
    os.environ['MASTER_PORT'] = str(find_free_port())

    if num_available_gpus < 2:
        print("Single GPU - no distributed")
        main_process(0, configs, False)
    else:
        mp.spawn(main_process, nprocs=configs.world_size, args=(configs, True))


def reduce_dict(d, gpu, cast_cpu=False):
    out = {}
    for key, value in d.items():
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value).cuda(gpu)
        dist.all_reduce(value)
        if cast_cpu:
            value = value.cpu()
        out[key] = value
    return out


def main_process(gpu, configs, is_distributed_training=True):
    """ Main function loads all necessary modules"""
    """----- Setup Experiment -----"""

    rank = configs.nr * configs.num_gpus + gpu
    is_main_process = False
    dist.init_process_group(backend=configs.backend, init_method="env://", world_size=configs.world_size, rank=rank)
    if dist.get_rank() == 0:
        is_main_process = True
    a = torch.tensor([1.0]).cuda(gpu)
    b = torch.ones(10).cuda(gpu)
    c = torch.ones(10).cuda(gpu) + 10
    hi = {'a': a, 'b': b, 'c': c}
    out = reduce_dict(hi, gpu, True)

    if is_main_process:
        print(out)


if __name__ == "__main__":
    print("Distributed Training!")
    main()
