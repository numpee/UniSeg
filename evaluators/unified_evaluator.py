import numpy as np
import torch
import wandb
from tqdm import tqdm

from datasets.dataset_utils import get_label_2_train, get_dataset_categories
from evaluators.abc import AbstractBaseEvaluator


class UnifiedEvaluator(AbstractBaseEvaluator):
    def __init__(self, models, dataloaders, loggers, visualize_samples_datasets, num_image_samples, **kwargs):
        print("Unified Evaluator")
        super().__init__(models, dataloaders, loggers, visualize_samples_datasets, num_image_samples, **kwargs)
        self.model = self.models['segmentation']
        self.dataset_names = list(self.dataloaders.keys())
        self.class_idxs = {}
        for name in self.dataset_names:
            combined_mapping_key = "{}_to_combined".format(name)
            combined_label2train = torch.tensor(get_label_2_train(combined_mapping_key))
            cls_idxs = combined_label2train[:, 1].unique()
            cls_idxs = cls_idxs[cls_idxs != 255].tolist()
            self.class_idxs[name] = cls_idxs
        self.samples_dump = {d_name: {} for d_name in self.visualize_samples_datasets}

    @torch.no_grad()
    def evaluate(self):
        all_pix_acc = {}
        all_miou = {}
        iou_tables = {}
        dataset_key = "city_idd_bdd_mapillary" if len(self.dataset_names) == 4 else "city_idd_bdd"
        dataset_categories = get_dataset_categories(dataset_key)   # TODO find a way to not hardcode this
        for d_name, val_loader in self.dataloaders.items():
            cls_idxs = self.class_idxs[d_name]
            num_classes = len(cls_idxs)
            num_batches = len(val_loader)
            sample_every = num_batches // self.num_image_samples
            sample_batch_idxs = list(range(0, num_batches, sample_every))

            visualization_samples = []
            pred_samples = []
            gt_samples = []
            total_correct, total_label, total_inter, total_union = 0, 0, 0, 0
            pix_acc, miou = 0, 0
            tbar = tqdm(val_loader, desc='\r')
            for batch_idx, (image, target, caption) in enumerate(tbar):
                image, target = image.to(self.device), target.to(self.device)
                preds = self.model(image)[0]
                preds = preds[:, cls_idxs, :, :]
                correct, labeled, inter, union = self.eval_batch(preds, target, num_classes)
                total_correct += correct
                total_label += labeled
                total_inter += inter
                total_union += union
                pix_acc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                miou = IoU.mean()
                tbar.set_description('{}: pixAcc: {:.3f}, mIoU: {:.3f}'.format(d_name, pix_acc, miou))
                if batch_idx in sample_batch_idxs and d_name in self.visualize_samples_datasets:
                    pred_classes = preds[0].argmax(dim=0)
                    visualization_samples.append(image[0].cpu())
                    pred_samples.append(pred_classes.cpu())
                    gt_samples.append(target[0].cpu())
            all_pix_acc[d_name] = pix_acc
            all_miou[d_name] = miou

            if d_name in self.visualize_samples_datasets:
                pil_samples = []
                for img in visualization_samples:
                    img = self.unnormalize_to_pil(img)
                    pil_samples.append(img)
                self.samples_dump[d_name] = {'images': pil_samples, 'predictions': pred_samples,
                                             'ground_truth': gt_samples}

            iou_table = wandb.Table(columns=["Class name", "IoU"])
            for idx, iou in zip(cls_idxs, IoU):
                iou_table.add_data(dataset_categories[idx], "{:.3f}".format(iou))
            iou_table.add_data("All", "{:.3f}".format(miou))
            iou_tables[d_name] = iou_table

        results = {}
        for d_name in self.dataset_names:
            results[d_name + "_pixAcc"] = all_pix_acc[d_name]
            results[d_name + "_mIoU"] = all_miou[d_name]
            results[d_name + "_IoU_table"] = iou_tables[d_name]

        for d_name, dump in self.samples_dump.items():
            class_dict = {}
            for i, l in enumerate(self.class_idxs[d_name]):
                class_dict[i] = dataset_categories[l]
            images = dump['images']
            preds = dump['predictions']
            gts = dump['ground_truth']
            wandb_images = []
            for img, pred, gt in zip(images, preds, gts):
                pred = pred.numpy().astype(np.uint8)
                gt = gt.numpy().astype(np.uint8)
                wandb_img = wandb.Image(img, masks={
                    "prediction": {"mask_data": pred, "class_labels": class_dict},
                    "ground_truth": {"mask_data": gt, "class_labels": class_dict}
                })
                wandb_images.append(wandb_img)
            results['samples/{}'.format(d_name)] = wandb_images
        return results
