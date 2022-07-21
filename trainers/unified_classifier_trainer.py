import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from tqdm import tqdm

from datasets.dataset_utils import get_label_2_train
from libs.utils.metrics import intersectionAndUnionGPU
from trainers.abc import AbstractBaseTrainer
from utils.metrics import AverageMeterSet


class UnifiedClassifierTrainer(AbstractBaseTrainer):
    def __init__(self, models, dataloaders, criterions, optimizers, lr_schedulers, num_epochs, train_loggers,
                 val_loggers, **kwargs):
        print("Unified Trainer")
        super().__init__(models, dataloaders, criterions, optimizers, lr_schedulers, num_epochs,
                         train_loggers, val_loggers, **kwargs)
        self.model = models['segmentation']
        self.criterion = criterions
        self.val_dataloaders = dataloaders['val']
        self.dataset_names = kwargs['train_dataset_names']
        self.eval_dataset_names = list(self.val_dataloaders.keys())
        all_dataset_names = set(self.eval_dataset_names).union(set(self.dataset_names))
        self.class_idxs = {}
        self.mapping_scheme = kwargs['mapping_scheme'] if 'mapping_scheme' in kwargs else "_to_combined"
        for name in all_dataset_names:
            if name not in self.dataset_names:
                mapping_scheme = "_to_combined"
            else:
                mapping_scheme = self.mapping_scheme
            combined_mapping_key = "{}{}".format(name, mapping_scheme)
            combined_label2train = torch.tensor(get_label_2_train(combined_mapping_key))
            cls_idxs = combined_label2train[:, 1].unique()
            cls_idxs = cls_idxs[cls_idxs != 255].tolist()
            self.class_idxs[name] = cls_idxs

    def train_one_epoch(self, epoch):
        average_meter_set = AverageMeterSet()
        dataloader_tqdm = tqdm(self.train_dataloader, desc="Epoch {}".format(epoch))
        for batch_idx, ((image, target, _), _) in enumerate(dataloader_tqdm):
            image, target = image.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            self._reset_grad()

            with autocast(enabled=self.use_amp):
                outputs = self.model(image)[0]
                loss = self.criterion(outputs, target)

            self.scaler.scale(loss).backward()
            self._update_grad()
            self._step_schedulers(batch_idx, epoch)
            self.scaler.update()
            average_meter_set.update('train_loss', loss.item())
            dataloader_tqdm.set_description('Train loss: %.3f' % average_meter_set['train_loss'].avg)

        train_results = average_meter_set.averages()
        return train_results

    @torch.no_grad()
    def validate(self, epoch):
        all_miou = {}

        for d_name, val_loader in self.val_dataloaders.items():
            cls_idxs = self.class_idxs[d_name]
            num_classes = len(cls_idxs)

            total_inter, total_union = 0, 0
            tbar = tqdm(val_loader, desc='\r')
            for batch_idx, (image, target, caption) in enumerate(tbar):
                image, target = image.to(self.device), target.to(self.device)
                with autocast(enabled=self.use_amp):
                    preds = self.model(image)[0]
                preds = preds[:, cls_idxs, :, :]

                inter, union, _ = intersectionAndUnionGPU(preds.max(1)[1], target, K=num_classes)
                if self.distributed_training:
                    dist.all_reduce(inter), dist.all_reduce(union)
                inter, union = inter.cpu().numpy(), union.cpu().numpy()
                total_inter += inter
                total_union += union
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                miou = IoU.mean()
                tbar.set_description('{} mIoU: {:.3f}'.format(d_name, miou))
            all_miou[d_name] = miou

        results = {}
        for d_name in self.eval_dataset_names:
            results[d_name + "_mIoU"] = all_miou[d_name]
        del image
        del target
        del inter
        del union

        return results
