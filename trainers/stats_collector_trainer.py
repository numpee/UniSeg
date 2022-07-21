import torch
import torch.distributed as dist
from tqdm import tqdm

from datasets.dataset_utils import get_label_2_train
from trainers.abc import AbstractBaseTrainer
from utils.metrics import AverageMeterSet


class StatsCollectorTrainer(AbstractBaseTrainer):
    def __init__(self, models, dataloaders, criterions, optimizers, lr_schedulers, num_epochs, train_loggers,
                 val_loggers, num_classes, **kwargs):
        print("Stats Collector Trainer")
        super().__init__(models, dataloaders, criterions, optimizers, lr_schedulers, num_epochs,
                         train_loggers, val_loggers, **kwargs)
        self.model = models['segmentation']
        self.criterion = criterions
        self.val_dataloaders = dataloaders['val']
        self.dataset_names = kwargs['dataset_names']
        self.eval_dataset_names = list(self.val_dataloaders.keys())
        self.num_classes = num_classes
        print(self.num_classes)
        self.cls_arange = torch.arange(self.num_classes).cuda()
        self.scale_factor = kwargs['scale_factor']
        self.save_file_name = kwargs['save_file_name']

    @torch.no_grad()
    def train_one_epoch(self, epoch):
        dataloader_tqdm = tqdm(self.train_dataloader, desc="Epoch {}".format(epoch))
        self._to_eval_mode()
        stats = {d_name: {cls_id: {'data': torch.zeros(self.num_classes).cuda(), 'num': 0} for cls_id in
                          range(self.num_classes)} for d_name in self.dataset_names}
        for batch_idx, ((image, target, _), domains) in enumerate(dataloader_tqdm):
            # if batch_idx >= 100:
            #     break
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            outputs = self.model(image)[0]
            for i, d in enumerate(domains):
                for cls_idx in range(self.num_classes):
                    mask = target[i]
                    cls_mask = mask == cls_idx
                    num_samples = cls_mask.sum().item()
                    masked_out = outputs[i] * cls_mask.unsqueeze(0).type(outputs[i].type())
                    summed_out = masked_out.sum(-1).sum(-1) / self.scale_factor
                    stats[d][cls_idx]['data'] += summed_out
                    stats[d][cls_idx]['num'] += num_samples
        # print(stats)
        out_stats = reduce_stats(stats, cast_cpu=True)
        return out_stats

    def collect_stats(self):
        stats = self.train_one_epoch(0)
        # print(stats)
        import pickle
        with open('/home/dongwan/uniseg/{}'.format(self.save_file_name), 'wb') as f:
            pickle.dump(stats, f)
        # torch.save(stats, 'stats_cosine.pth')
        # if self.is_main_process:


def reduce_stats(stats, cast_cpu=True):
    out = {}
    for d_name, d in stats.items():
        new_cls_dict = {}
        for cls_id, sum_vals in d.items():
            values = sum_vals['data'].cpu() if isinstance(sum_vals, torch.Tensor) else sum_vals
            num = sum_vals['num']
            new_sum_vals = {'data': values, 'num': num}
            new_cls_dict[cls_id] = new_sum_vals
        out[d_name] = new_cls_dict
    # print(out)
    return out


def reduce_dict(d, gpu, cast_cpu=True):
    out = {}
    for key, value in d.items():
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value).cuda(gpu)
        dist.all_reduce(value)
        if cast_cpu:
            value = value.cpu()
        out[key] = value
    return out
