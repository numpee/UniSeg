import numpy as np
import torch
from tqdm import tqdm
import torch.distributed as dist

import libs.utils as utils
from trainers.abc import AbstractBaseTrainer
from utils.metrics import AverageMeterSet
from libs.utils.metrics import intersectionAndUnionGPU
from datasets.dataset_utils import get_label_2_train


class SingleTrainer(AbstractBaseTrainer):
    def __init__(self, models, dataloaders, criterions, optimizers, lr_schedulers, num_epochs, train_loggers,
                 val_loggers, **kwargs):
        print("Single Trainer")
        super().__init__(models, dataloaders, criterions, optimizers, lr_schedulers, num_epochs,
                         train_loggers, val_loggers, **kwargs)
        self.model = models['segmentation']
        self.criterion = criterions
        self.val_dataloaders = dataloaders['val']
        self.dataset_name = list(self.val_dataloaders.keys())[0]
        self.val_dataloader = self.val_dataloaders[self.dataset_name]
        self.num_classes = kwargs['num_classes']

    def train_one_epoch(self, epoch):
        average_meter_set = AverageMeterSet()
        dataloader_tqdm = tqdm(self.train_dataloader, desc="Epoch {}".format(epoch))
        for batch_idx, (image, target, _) in enumerate(dataloader_tqdm):
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)

            self._reset_grad()

            outputs = self.model(image)[0]
            loss = self.criterion(outputs, target)
            loss.backward()
            self._update_grad()
            self._step_schedulers(batch_idx, epoch)
            average_meter_set.update('train_loss', loss.item())
            dataloader_tqdm.set_description('Train loss: %.3f' % average_meter_set['train_loss'].avg)

        train_results = average_meter_set.averages()
        return train_results

    @torch.no_grad()
    def validate(self, epoch):
        total_correct, total_label, total_inter, total_union = 0, 0, 0, 0
        pix_acc, miou = 0, 0
        tbar = tqdm(self.val_dataloader, desc='\r')
        for i, (image, target, _) in enumerate(tbar):
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            preds = self.model(image)[0]
            inter, union, _ = intersectionAndUnionGPU(preds.max(1)[1], target, K=self.num_classes)
            if self.distributed_training:
                dist.all_reduce(inter), dist.all_reduce(union)
            inter, union = inter.cpu().numpy(), union.cpu().numpy()
            total_inter += inter
            total_union += union
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            miou = IoU.mean()
            tbar.set_description('pixAcc: %.3f, mIoU: %.3f' % (pix_acc, miou))

        results = {'{}_pixAcc'.format(self.dataset_name): 0,
                   '{}_mIoU'.format(self.dataset_name): miou}
        return results
