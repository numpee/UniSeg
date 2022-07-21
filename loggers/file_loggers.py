import os

import torch

from loggers import AbstractBaseLogger


def _checkpoint_file_path(export_path, filename):
    return os.path.join(export_path, filename)


def _set_up_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def _save_state_dict_with_step(log_data, step, path, filename):
    log_data = {k: v for k, v in log_data.items() if isinstance(v, dict)}
    log_data['step'] = step
    torch.save(log_data, _checkpoint_file_path(path, filename))


class RecentModelTracker(AbstractBaseLogger):
    def __init__(self, export_path, ckpt_filename='recent.pth', log_start=0):
        self.export_path = export_path
        _set_up_path(self.export_path)
        self.ckpt_filename = ckpt_filename
        self.log_start = log_start

    def log(self, log_data, step, commit=False):
        if step >= self.log_start:
            _save_state_dict_with_step(log_data, step, self.export_path, "recent.pth")

    def complete(self, log_data, step):
        pass


class BestModelTracker(AbstractBaseLogger):
    def __init__(self, export_path, ckpt_filename='best.pth', metric_key='total_mIoU'):
        self.export_path = export_path
        _set_up_path(self.export_path)

        self.metric_key = metric_key
        self.ckpt_filename = ckpt_filename

        self.best_value = -9e9

    def log(self, log_data, step, commit=False):
        if self.metric_key not in log_data:
            print("WARNING: The key: {} is not in logged data. Not saving best model".format(self.metric_key))
            return
        recent_value = log_data[self.metric_key]
        if self.best_value < recent_value:
            self.best_value = recent_value
            _save_state_dict_with_step(log_data, step, self.export_path, self.ckpt_filename)
            print("Update Best {} Model at Step {} with value {}".format(self.metric_key, step, self.best_value))

    def complete(self, *args, **kwargs):
        pass
