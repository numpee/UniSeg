import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from tqdm import tqdm

from datasets.dataset_utils import get_label_2_train, get_dataset_category_union, get_dataset_categories
from libs.utils.metrics import intersectionAndUnionGPU
from trainers.abc import AbstractBaseTrainer
from utils.metrics import AverageMeterSet


class BCENullTrainerSecondStage(AbstractBaseTrainer):
    def __init__(self, models, dataloaders, criterions, optimizers, lr_schedulers, num_epochs, train_loggers,
                 val_loggers, num_classes, **kwargs):
        print("BCE Second Stage Trainer")
        super().__init__(models, dataloaders, criterions, optimizers, lr_schedulers, num_epochs,
                         train_loggers, val_loggers, **kwargs)
        self.model = models['segmentation']
        self.criterion = criterions
        self.val_dataloaders = dataloaders['val']
        self.dataset_names = kwargs['dataset_names']
        self.eval_dataset_names = list(self.val_dataloaders.keys())
        all_dataset_names = set(self.eval_dataset_names).union(set(self.dataset_names))
        self.train_dataset_names = kwargs.get('train_dataset_names', [])
        self.class_idxs = {}
        self.train_mapping_scheme = kwargs['train_mapping_scheme']
        self.val_mapping_scheme = kwargs['val_mapping_scheme']
        self.mapillary_special = kwargs['mapillary_special']
        for name in self.eval_dataset_names:
            if name in self.train_dataset_names:
                mapping_scheme = self.train_mapping_scheme
                combined_mapping_key = "{}{}".format(name, mapping_scheme)
                combined_label2train = torch.tensor(get_label_2_train(combined_mapping_key))
                cls_idxs = combined_label2train[:, 1].unique()
                cls_idxs = cls_idxs[cls_idxs != 255].tolist()
                cls_idxs = sorted(cls_idxs)
            else:
                union = sorted(get_dataset_category_union(self.train_dataset_names))
                curr_d_cls_names = sorted(get_dataset_categories(name))
                if name == "camvid" and self.train_mapping_scheme == "_to_cib":
                    tmp_name = "camvid_cib"
                    print("USE CAMVID CIB")
                    curr_d_cls_names = sorted(get_dataset_categories(tmp_name))
                cls_idxs = []
                for cls_name in curr_d_cls_names:
                    if cls_name in union:
                        cls_idxs.append(union.index(cls_name))
            self.class_idxs[name] = cls_idxs
        self.num_classes = num_classes
        self.cls_arange = torch.arange(self.num_classes).cuda()
        self.class_distributions = kwargs['class_distributions']
        self.distribution_threshold = kwargs['distribution_threshold']
        self.identity = torch.eye(self.num_classes).cuda()
        self.class_distributions = self.process_class_distributions(self.distribution_threshold)

    def train_one_epoch(self, epoch):
        average_meter_set = AverageMeterSet()
        dataloader_tqdm = tqdm(self.train_dataloader, desc="Epoch {}".format(epoch))
        for batch_idx, ((image, target, _), domains) in enumerate(dataloader_tqdm):
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            target_labels = self.generate_soft_bce_labels(target, domains)
            self._reset_grad()

            with autocast(enabled=self.use_amp):
                outputs = self.model(image)[0]
                loss = self.criterion(outputs, target_labels)

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

        return results

    def generate_soft_bce_labels(self, gt, domains):
        oh = (self.cls_arange == gt[..., None]).float()
        B, H, W, C = oh.size()
        tensor_list = []
        for d in domains:
            tensor_list.append(self.class_distributions[d])
        distribution_matrix = torch.stack(tensor_list)
        soft_labels = torch.matmul(oh.view(B, H * W, C), distribution_matrix)
        soft_labels = soft_labels.view(B, H, W, C).permute(0, 3, 1, 2)
        return soft_labels

    @torch.no_grad()
    def process_class_distributions(self, threshold=0.5):
        processed_dict = {}
        for dname, dicts in self.class_distributions.items():
            processed_dict[dname] = {}
            tensor_list = []
            for cls_idx, dist in dicts.items():
                tensor_list.append(dist)
            out_tensor = torch.stack(tensor_list).cuda()
            out_tensor = out_tensor + self.identity
            out_tensor[out_tensor > 1] = 1
            if not threshold == -1:
                out_tensor[out_tensor < threshold] = 0
            processed_dict[dname] = out_tensor
        return processed_dict
