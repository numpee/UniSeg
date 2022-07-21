import hydra
import torch
import torch.nn as nn
import os
from datasets import test_dataloader_factory, get_num_classes
from evaluators.paper_evaluator import PaperEvaluator
from loggers.wandb_loggers import WandbSimplePrinter
from models import get_segmentation_model
from setup import setup_wandb_evaluation
from transformations import transform_factory_test


@hydra.main(config_path='configs/paper_evaluation_config.yaml')
def main(configs):
    """ Main function loads all necessary modules"""
    """----- Setup Experiment -----"""
    setup_wandb_evaluation(configs)
    print(configs.pretty())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """----- Load Dataset -----"""
    joint_val_transform, input_val_transform = transform_factory_test(configs)
    joint_transforms = {'train': None, 'val': joint_val_transform}
    input_transforms = {'train': None, 'val': input_val_transform}

    if configs.batch_size > 1:
        if configs.use_sliding_window:
            raise NotImplementedError("Cannot use batch size > 1 for sliding window!")
    dataloaders = test_dataloader_factory(configs, joint_transforms=joint_transforms, input_transforms=input_transforms)

    """----- Load Model -----"""
    num_hierarchy_classes = configs.get("num_hierarchy_classes", 0)
    num_classes = configs.num_classes if configs.num_classes is not None else get_num_classes(configs)
    configs.num_classes = num_classes
    num_total_classes = num_classes + num_hierarchy_classes
    num_datasets = len(configs.dataset_names)

    ckpt_names = ["{}_mIoU_best.pth".format(d_name) for d_name in
                  ['cityscapes', 'idd', 'bdd', 'mapillary', 'kitti', 'wilddash']]
    ckpt_names.append("recent.pth")

    if configs.model_folder_path.endswith('.pth'):
        files_in_dir = [configs.model_folder_path]
    else:
        files_in_dir = os.listdir(configs.model_folder_path)
    step = 0
    for ckpt_name in ckpt_names:
        if not ckpt_name in files_in_dir:
            step += 1
            print("-----------------\nSkipping:{}\n-----------------".format(ckpt_name))
            continue
        print("-----------------\nModel:{}\n-----------------".format(ckpt_name))
        ckpt_path = os.path.join(configs.model_folder_path, ckpt_name)
        model = get_segmentation_model(configs.model, backbone=configs.backbone, aux=configs.aux,
                                       se_loss=configs.se_loss, num_classes=num_total_classes, num_heads=num_datasets,
                                       configs=configs)
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

        if device == 'cuda':
            model = nn.DataParallel(model.to(device))
        # model = model.to(device)
        models = {"segmentation": model}

        """----- Loggers -----"""
        loggers = [WandbSimplePrinter("results/")]

        """----- Load Evaluator -----"""
        samples_datasets = configs.image_log_datasets
        num_samples = configs.num_images_to_log

        if configs.use_sliding_window:
            print("---------------------------\nUse Sliding Window\n---------------------------")
        else:
            print("---------------------------\nOrdinary Eval\n---------------------------")

        evaluator_cls = PaperEvaluator
        evaluator = evaluator_cls(models, dataloaders, loggers, samples_datasets, num_samples, num_classes,
                                  crop_size=configs.crop_size, use_sliding_window=configs.use_sliding_window,
                                  mapping_scheme=configs.train_mapping_scheme,
                                  dataset_key=configs.dataset_key)
        evaluator.run(step=step)
        step += 1
    print("Done evaluating: Printed to wandb")


if __name__ == "__main__":
    main()
