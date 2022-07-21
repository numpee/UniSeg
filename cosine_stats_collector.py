import os

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from datasets import get_num_classes, distributed_dataloader_factory
from models import get_segmentation_model
from setup import setup_experiment_without_wandb
from trainers.stats_collector_trainer import StatsCollectorTrainer
from utils.distributed_utils import find_free_port


@hydra.main(config_path='configs/stats_collector_config.yaml', strict=False)
def main(configs):
    """
    IMPORTANT NOTE! ONLY WORKS WITH 1 GPU!!!!!!!!!
    """
    num_available_gpus = torch.cuda.device_count()
    configs.world_size = configs.num_gpu_per_node * num_available_gpus
    configs.num_gpus = num_available_gpus
    configs.batch_size_per_gpu = configs.batch_size // num_available_gpus
    configs.num_workers_per_gpu = configs.num_workers // num_available_gpus
    os.environ['MASTER_ADDR'] = str(configs.master_address)
    os.environ['MASTER_PORT'] = str(find_free_port())

    export_root, configs = setup_experiment_without_wandb(configs)
    configs.export_root = export_root
    print(configs.pretty())
    print("Distributed Training")

    if num_available_gpus < 2:
        print("Single GPU - no distributed")
        main_process(0, configs, False)
    else:
        mp.spawn(main_process, nprocs=configs.world_size, args=(configs, True))


def main_process(gpu, configs, is_distributed_training=True):
    """ Main function loads all necessary modules"""
    """----- Setup Experiment -----"""
    rank = configs.nr * configs.num_gpus + gpu
    is_main_process = False
    dist.init_process_group(backend=configs.backend, init_method="env://", world_size=configs.world_size, rank=rank)
    if dist.get_rank() == 0:
        is_main_process = True

    """----- Load Dataset -----"""
    dataloaders, train_sampler = distributed_dataloader_factory(configs)

    """----- Load Model, Optimizer, & Loss fn -----"""
    num_classes = configs.num_classes if configs.num_classes is not None else get_num_classes(configs)
    configs.num_classes = num_classes
    model = get_segmentation_model(configs.model, num_classes=num_classes, configs=configs)
    if configs.resume_training:
        print("resume training")
        start_epoch = load_model(configs, model)
    if is_distributed_training:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(gpu)

    start_epoch = 0

    model = DistributedDataParallel(model, device_ids=[gpu]) if is_distributed_training else model

    models = {'segmentation': model}

    """----- Load Trainer -----"""
    trainer_cls = StatsCollectorTrainer

    trainer = trainer_cls(models, dataloaders, {}, {}, {}, configs.epochs,
                          [], [], num_classes=num_classes, start_epoch=start_epoch,
                          dataset_names=configs.dataset_names, mapping_scheme=configs.train_mapping_scheme,
                          scale_factor=configs.scale_factor, save_file_name=configs.save_file_name)
    if is_distributed_training:
        trainer.setup_distributed_trainer(train_sampler=train_sampler, is_main_process=is_main_process)
    trainer.collect_stats()


def load_model(configs, model):
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
    print("Loaded from {}".format(configs.ckpt_path))

    return curr_epoch


if __name__ == "__main__":
    """
    IMPORTANT NOTE! ONLY WORKS WITH 1 GPU!!!!!!!!!
    """
    print("Distributed Training!")
    main()
