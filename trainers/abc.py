from abc import ABC

import torch
from torch import nn as nn
from torch.cuda.amp import GradScaler

from loggers.abc import LoggingService


class AbstractBaseTrainer(ABC):
    def __init__(self, models, dataloaders, criterions, optimizers, lr_schedulers, num_epochs, train_loggers,
                 val_loggers, **kwargs):
        self.models = models
        self.model = models['segmentation']
        self.train_dataloader = dataloaders['train']
        self.val_dataloaders = dataloaders['val']
        self.eval_dataset_names = list(self.val_dataloaders.keys())
        self.criterions = criterions
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.num_epochs = num_epochs
        self.train_logging_service = LoggingService(train_loggers)
        self.val_logging_service = LoggingService(val_loggers)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.start_epoch = kwargs['start_epoch'] if 'start_epoch' in kwargs else 0
        self.train_sampler = None
        self.is_main_process = True
        self.distributed_training = False
        self.class_idxs = dict()
        self.cls_arange = torch.empty(0)
        self.eval_start_epoch = kwargs['eval_start_epoch'] if 'eval_start_epoch' in kwargs else 0
        self.use_amp = kwargs.get('use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.eval_start_epoch != 0:
            print("Starting evaluation from epoch {}".format(self.eval_start_epoch))

    def setup_distributed_trainer(self, train_sampler, is_main_process):
        self.is_main_process = is_main_process
        self.train_sampler = train_sampler
        self.distributed_training = True
        if self.is_main_process:
            print("Distributed trainer setup")

    def train_one_epoch(self, epoch) -> dict:
        raise NotImplementedError

    def run(self) -> dict:
        for epoch in range(self.start_epoch, self.num_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self._to_train_mode()
                    train_results = self.train_one_epoch(epoch)
                    if self.is_main_process:
                        self.train_logging_service.log(train_results, step=epoch)
                        print(train_results)
                else:
                    if epoch >= self.eval_start_epoch:
                        self._to_eval_mode()
                        val_results = self.validate(epoch)
                        model_state_dicts = self._get_state_dicts(self.models)
                        # optimizer_state_dicts = self._get_state_dicts(self.optimizers)
                        val_results['model_state_dict'] = model_state_dicts
                        # val_results['optimizer_state_dict'] = optimizer_state_dicts
                        if self.is_main_process:
                            self.val_logging_service.log(val_results, step=epoch, commit=True)
                    torch.cuda.empty_cache()

        return self.models

    @torch.no_grad()
    def validate(self, epoch):
        all_miou = {}
        for d_name in self.eval_dataset_names:
            all_miou[d_name + "_mIoU"] = 0
        return all_miou
        # for d_name, val_loader in self.val_dataloaders.items():
        #     cls_idxs = self.class_idxs[d_name]
        #     num_classes = len(cls_idxs)
        #     total_h_inter, total_h_union, total_inter, total_union = 0, 0, 0, 0
        #
        #     miou = 0
        #     tbar = tqdm(val_loader, desc='\r')
        #     for i, (image, target, _) in enumerate(tbar):
        #         image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
        #         preds = self.model(image)[0]
        #         preds = preds[:, cls_idxs, :, :]
        #         inter, union, _ = intersectionAndUnionGPU(preds.max(1)[1], target, K=num_classes)
        #         if self.distributed_training:
        #             dist.all_reduce(inter), dist.all_reduce(union)
        #         inter, union = inter.cpu().numpy(), union.cpu().numpy()
        #         total_inter += inter
        #         total_union += union
        #         IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        #         miou = IoU.mean()
        #         tbar.set_description('{} mIoU: {:.3f}'.format(d_name, miou))
        #     all_miou[d_name] = miou
        #
        # results = {}
        # for d_name in self.eval_dataset_names:
        #     results[d_name + "_mIoU"] = all_miou[d_name]
        # return results

    def generate_target_one_hot(self, target):
        target_oh = (self.cls_arange == target[..., None]).float().permute(0, 3, 1, 2)
        return target_oh

    def _load_models_to_device(self):
        for model in self.models.values():
            model.to(self.device)

    def _to_train_mode(self, keys=None):
        keys = keys if keys else self.models.keys()
        for key in keys:
            self.models[key].train()

    def _to_eval_mode(self, keys=None):
        keys = keys if keys else self.models.keys()
        for key in keys:
            self.models[key].eval()

    def _reset_grad(self, keys=None):
        keys = keys if keys else self.optimizers.keys()
        for key in keys:
            self.optimizers[key].zero_grad()

    def _update_grad(self, keys=None, exclude_keys=None):
        keys = keys if keys else list(self.optimizers.keys())
        if exclude_keys:
            keys = [key for key in keys if key not in exclude_keys]
        for key in keys:
            self.scaler.step(self.optimizers[key])

    def _step_schedulers(self, iteration, epoch):
        if self.lr_schedulers is None:
            return
        for scheduler in self.lr_schedulers.values():
            scheduler(iteration, epoch)

    @staticmethod
    def _get_state_dicts(dict_of_models):
        state_dicts = {}
        for model_name, model in dict_of_models.items():
            if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
                state_dicts[model_name] = model.module.state_dict()
            else:
                state_dicts[model_name] = model.state_dict()
        return state_dicts
