import numpy as np
import torch
import wandb
from tqdm import tqdm

from datasets.dataset_utils import get_label_2_train, get_dataset_categories
from evaluators.abc import AbstractBaseEvaluator


class SingleEvaluator(AbstractBaseEvaluator):
    def __init__(self, models, dataloaders, loggers, visualize_samples_datasets, num_image_samples, **kwargs):
        print("Single Evaluator")
        super().__init__(models, dataloaders, loggers, visualize_samples_datasets, num_image_samples, **kwargs)
        self.model = self.models['segmentation']
        self.dataset_name = list(self.dataloaders.keys())[0]
        self.dataloader = self.dataloaders[self.dataset_name]
        self.dataset_categories = get_dataset_categories(self.dataset_name)
        self.num_classes = len(self.dataset_categories)

    @torch.no_grad()
    def evaluate(self):
        total_correct, total_label, total_inter, total_union = 0, 0, 0, 0
        pix_acc, miou = 0, 0
        tbar = tqdm(self.dataloader, desc='\r')
        for i, (image, target, caption) in enumerate(tbar):
            image, target = image.to(self.device), target.to(self.device)
            preds = self.model(image)[0]
            correct, labeled, inter, union = self.eval_batch(preds, target, self.num_classes)
            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pix_acc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            miou = IoU.mean()
            tbar.set_description('pixAcc: %.3f, mIoU: %.3f' % (pix_acc, miou))

        results = {'{}_pixAcc'.format(self.dataset_name): pix_acc,
                   '{}_mIoU'.format(self.dataset_name): miou}

        iou_table = wandb.Table(columns=["Class name", "IoU"])
        for i, (category_name, iou) in enumerate(zip(self.dataset_categories, IoU)):
            iou_table.add_data(category_name, "{:.3f}".format(iou))
        iou_table.add_data("All", "{:.3f}".format(miou))
        results['{}_IoU_table'.format(self.dataset_name)] = iou_table

        return results
