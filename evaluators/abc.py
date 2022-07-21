import torch
import torchvision.transforms as transforms
from abc import ABC
import libs.utils as utils

from loggers.abc import LoggingService


class AbstractBaseEvaluator(ABC):
    def __init__(self, models, dataloaders, loggers, visualize_samples_datasets, num_image_samples, **kwargs):
        self.models = models
        self.dataloaders = dataloaders
        self.logging_service = LoggingService(loggers)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.visualize_samples_datasets = visualize_samples_datasets
        self.num_image_samples = num_image_samples
        self.unnormalize_to_pil = transforms.Compose(
            [transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
             transforms.ToPILImage()])

    @torch.no_grad()
    def evaluate(self) -> dict:
        raise NotImplementedError

    def run(self, step=0):
        self._load_models_to_device()  # Redundant
        self._to_eval_mode(self.models)
        results = self.evaluate()
        self.logging_service.log(results, step=step, commit=True)
        # self.logging_service.log(results, step=1, commit=True)

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

    @staticmethod
    def eval_batch(pred, target, num_classes=28):
        correct, labeled = utils.batch_pix_accuracy(pred.data, target)
        inter, union = utils.batch_intersection_union(pred.data, target, num_classes)
        return correct, labeled, inter, union
