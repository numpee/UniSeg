import torch
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from datasets.dataset_utils import get_label_2_train
from libs.utils.metrics import intersectionAndUnionGPU
from trainers.abc import AbstractBaseTrainer
from utils.metrics import AverageMeterSet


class CutMixTrainer(AbstractBaseTrainer):
    def __init__(self, models, dataloaders, criterions, optimizers, lr_schedulers, num_epochs, train_loggers,
                 val_loggers, num_classes, **kwargs):
        print("CutMix Trainer")
        super().__init__(models, dataloaders, criterions, optimizers, lr_schedulers, num_epochs,
                         train_loggers, val_loggers, **kwargs)
        self.model = models['segmentation']
        self.val_dataloaders = dataloaders['val']
        self.dataset_names = kwargs['dataset_names']
        self.mapping_scheme = kwargs['mapping_scheme'] if 'mapping_scheme' in kwargs else "_to_combined"
        self.criterion = criterions
        self.eval_dataset_names = list(self.val_dataloaders.keys())
        all_dataset_names = set(self.eval_dataset_names).union(set(self.dataset_names))
        self.class_idxs = {}
        self.class_idxs_mask = {}
        for name in all_dataset_names:
            combined_mapping_key = "{}{}".format(name, self.mapping_scheme)
            combined_label2train = torch.tensor(get_label_2_train(combined_mapping_key))
            cls_idxs = combined_label2train[:, 1].unique()
            cls_idxs = cls_idxs[cls_idxs != 255].tolist()
            self.class_idxs[name] = cls_idxs
            mask = torch.zeros(num_classes).cuda()
            mask[cls_idxs] = 1
            self.class_idxs_mask[name] = mask.unsqueeze(-1).unsqueeze(-1)
        self.num_classes = num_classes
        self.cls_arange = torch.arange(self.num_classes).cuda()
        self.use_hierarchy = False

    def generate_target_one_hot(self, target):
        target_oh = (self.cls_arange == target[..., None]).float().permute(0, 3, 1, 2)
        return target_oh

    def train_one_epoch(self, epoch):
        average_meter_set = AverageMeterSet()
        dataloader_tqdm = tqdm(self.train_dataloader, desc="Epoch {}".format(epoch))
        for batch_idx, ((image, target, mixed_cls_idx), domains) in enumerate(dataloader_tqdm):
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            mixed_cls_idx = mixed_cls_idx.cuda(non_blocking=True)
            all_target_one_hot = self.generate_target_one_hot(target)
            self._reset_grad()

            outputs = self.model(image)[0]
            loss_return = self.criterion(outputs, all_target_one_hot)
            if isinstance(loss_return, tuple):
                loss, num_loss = loss_return
                masking = []
                cls_masking = []
                for i, d in enumerate(domains):
                    curr_cls_idx = mixed_cls_idx[i]
                    domain_mask = self.class_idxs_mask[d]
                    masking.append(self.class_idxs_mask[d])
                    cls_mask = torch.zeros_like(loss[i])
                    if not curr_cls_idx.float() in domain_mask:
                        cls_mask[curr_cls_idx, target[i] == curr_cls_idx] = 1
                    cls_masking.append(cls_mask)
                mask = torch.stack(tuple(masking), dim=0)
                cls_mask = torch.stack(tuple(cls_masking), dim=0)
                loss = (loss * mask + loss * cls_mask).sum() / num_loss
            else:
                loss = loss_return
            loss.backward()
            self._update_grad()
            self._step_schedulers(batch_idx, epoch)
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

            h_miou, miou = 0, 0
            tbar = tqdm(val_loader, desc='\r')
            for i, (image, target, _) in enumerate(tbar):
                image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
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
                tbar.set_description('{} mIoU: {:.3f}, h_mIoU: {:.3f}'.format(d_name, miou, h_miou))
            all_miou[d_name] = miou

        results = {}
        for d_name in self.eval_dataset_names:
            results[d_name + "_mIoU"] = all_miou[d_name]
        return results

    @staticmethod
    def _map_gt_values_with_dict(input_tensor, mapping_dict):
        out = input_tensor.clone()
        for source_val, target_val in mapping_dict.items():
            out[input_tensor == source_val] = target_val
        return out
