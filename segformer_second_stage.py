import os
import pickle
import sys

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel

from datasets import get_num_classes, distributed_dataloader_factory
from loggers.local_loggers import SimpleLoggerExample
from loggers.wandb_loggers import WandbSimplePrinter
from models import get_segmentation_model
from setup import setup_experiment_without_wandb
from trainers.bcenull_trainer_second_stage import BCENullTrainerSecondStage
from utils.distributed_utils import find_free_port, cleanup_distributed
from utils.utils import create_val_loggers, create_optimizers, create_schedulers, setup_distributed_experiment, \
    get_loss_fn


@hydra.main(config_path='configs/segformer_second_stage_config.yaml', strict=False)
def main(configs):
    num_available_gpus = torch.cuda.device_count()
    configs.world_size = configs.num_gpu_per_node * num_available_gpus
    configs.num_gpus = num_available_gpus
    configs.batch_size_per_gpu = configs.batch_size // num_available_gpus
    configs.num_workers_per_gpu = configs.num_workers // num_available_gpus
    os.environ['MASTER_ADDR'] = str(configs.master_address)
    os.environ['MASTER_PORT'] = str(find_free_port())

    export_root, configs = setup_experiment_without_wandb(configs)
    configs.export_root = export_root
    configs.use_amp = configs.get('use_amp', False)
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
    is_main_process = setup_distributed_experiment(configs, gpu, is_distributed_training)

    """----- Load Dataset -----"""
    dataloaders, train_sampler = distributed_dataloader_factory(configs)

    """----- Load Model, Optimizer, & Loss fn -----"""
    num_classes = configs.num_classes if configs.num_classes is not None else get_num_classes(configs)
    configs.num_classes = num_classes
    model = get_segmentation_model(configs.model, num_classes=num_classes, configs=configs)
    if configs.resume_training:
        print("resume training")
        ckpt_path = configs.ckpt_path
        print("Loading model in path: {}".format(ckpt_path))
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model_state_dict = state_dict['model_state_dict']['segmentation']
        new_sd = {}
        for key, val in model_state_dict.items():
            if key.startswith("module."):
                new_sd[key[7:]] = val
            else:
                new_sd[key] = val
        model.load_state_dict(new_sd)
        print("Successfully loaded model")
    if is_distributed_training:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(gpu)
    optimizers = create_optimizers(model, configs)
    schedulers = create_schedulers(optimizers, mode=configs.lr_scheduler,
                                   num_epochs=configs.epochs - configs.min_lr_epochs,
                                   iters_per_epoch=len(dataloaders['train']), min_lr=configs.min_lr)

    model = DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True) if is_distributed_training else model

    models = {'segmentation': model}
    criterions = get_loss_fn(configs).cuda(gpu)

    """----- Get Loggers -----"""
    train_loggers = []
    if configs.use_wandb:
        train_logger = WandbSimplePrinter('train/')
    else:
        train_logger = SimpleLoggerExample('train/', configs.experiment_dir)
    if is_main_process:
        train_loggers.append(train_logger)
    val_loggers = create_val_loggers(configs, configs.export_root) if is_main_process else []

    """----- Load Trainer -----"""
    with open(configs.distribution_path, 'rb') as f:
        print("Using distribution from path: {}".format(configs.distribution_path))
        class_distributions = pickle.load(f)
    start_epoch = 0
    trainer = BCENullTrainerSecondStage(models, dataloaders, criterions, optimizers, schedulers, configs.epochs,
                                        train_loggers, val_loggers, num_classes=num_classes, start_epoch=start_epoch,
                                        dataset_names=configs.dataset_names,
                                        distribution_threshold=configs.distribution_threshold,
                                        class_distributions=class_distributions,
                                        train_mapping_scheme=configs.train_mapping_scheme,
                                        val_mapping_scheme=configs.val_mapping_scheme,
                                        train_dataset_names=configs.dataset_names,
                                        mapillary_special=configs.mapillary_special_mapping,
                                        eval_start_epoch=configs.eval_start_epoch, use_amp=configs.use_amp)
    if is_distributed_training:
        trainer.setup_distributed_trainer(train_sampler=train_sampler, is_main_process=is_main_process)
    trainer.run()
    if is_distributed_training:
        dist.barrier()
        cleanup_distributed()
        if is_main_process:
            wandb.finish()
        sys.exit()


if __name__ == "__main__":
    print("Distributed Training!")
    main()
