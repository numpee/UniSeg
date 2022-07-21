import numpy as np
import torch
import wandb
from tqdm import tqdm

from datasets.dataset_utils import get_label_2_train, get_dataset_categories
from evaluators.abc import AbstractBaseEvaluator


class WildDashEvaluator(AbstractBaseEvaluator):
    def __init__(self, models, dataloaders, loggers, visualize_samples_datasets, num_image_samples, **kwargs):
        print("WildDash Evaluator")
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
        results = {}
        for d_name, val_loader in self.dataloaders.items():
            cls_idxs = self.class_idxs[d_name]
            num_classes = len(cls_idxs)
            assert val_loader.batch_size == 1
            min_miou = 1
            max_miou = 0
            visualizations = {key: {"img": None, "pred": None, "gt": None} for key in ["min", "max"]}
            tbar = tqdm(val_loader, desc='\r')
            for batch_idx, (image, target, caption) in enumerate(tbar):
                # if batch_idx != 24 and batch_idx != 6:
                #     continue
                image, target = image.to(self.device), target.to(self.device)
                preds = self.model(image)[0]
                preds = preds[:, cls_idxs, :, :]
                correct, labeled, inter, union = self.eval_batch(preds, target, num_classes)

                curr_iou = 1.0 * inter / (np.spacing(1) + union)
                curr_miou = curr_iou.mean()
                adjusted_union = union[union != 0]
                adjusted_inter = inter[union != 0]
                adjusted_iou = 1.0 * adjusted_inter / (np.spacing(1) + adjusted_union)
                adjusted_miou = adjusted_iou.mean()
                tbar.set_description('{}: mIoU: {:.3f}, adjusted: {:.3f}'.format(d_name, curr_miou, adjusted_miou))
                # if curr_miou < min_miou:
                if adjusted_miou < min_miou:
                    print("Update min sample")
                    # min_miou = curr_miou
                    min_miou = adjusted_miou
                    pred_classes = preds[0].argmax(dim=0)
                    visualizations['min']['img'] = image[0].cpu()
                    visualizations['min']['pred'] = pred_classes.cpu()
                    visualizations['min']['gt'] = target[0].cpu()
                    visualizations['min']['miou'] = curr_miou
                    visualizations['min']['iou'] = curr_iou
                # if curr_miou > max_miou:
                if adjusted_miou > max_miou:
                    print("Update max sample")
                    max_miou = adjusted_miou
                    pred_classes = preds[0].argmax(dim=0)
                    visualizations['max']['img'] = image[0].cpu()
                    visualizations['max']['pred'] = pred_classes.cpu()
                    visualizations['max']['gt'] = target[0].cpu()
                    visualizations['max']['miou'] = curr_miou
                    visualizations['max']['iou'] = curr_iou

            class_dict = {}
            dataset_categories = get_dataset_categories('city_idd_bdd')
            for i, l in enumerate(self.class_idxs[d_name]):
                class_dict[i] = dataset_categories[l]

            for minmax, values in visualizations.items():
                img = values['img']
                pil_img = self.unnormalize_to_pil(img)
                pred = values['pred'].numpy().astype(np.uint8)
                gt = values['gt'].numpy().astype(np.uint8)
                wandb_img = wandb.Image(pil_img, masks={
                    "prediction": {"mask_data": pred, "class_labels": class_dict},
                    "ground_truth": {"mask_data": gt, "class_labels": class_dict}
                })
                results['{}_miou_sample'.format(minmax)] = wandb_img

            iou_table = wandb.Table(columns=["Class name", "min_IoU", "max_IoU"])
            min_iou = visualizations['min']['iou']
            min_miou = visualizations['min']['miou']
            max_iou = visualizations['max']['iou']
            max_miou = visualizations['max']['miou']
            for idx, min_iou, max_iou in zip(cls_idxs, min_iou, max_iou):
                iou_table.add_data(dataset_categories[idx], "{:.3f}".format(min_iou), "{:.3f}".format(max_iou))
            iou_table.add_data("All", "{:.3f}".format(min_miou), "{:.3f}".format(max_miou))
            results['{}_IoU_table'.format(d_name)] = iou_table

        return results
