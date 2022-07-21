import numpy as np
import torch
import wandb
from tqdm import tqdm

from datasets.dataset_utils import get_label_2_train, get_dataset_categories, get_dataset_category_union
from evaluators.abc import AbstractBaseEvaluator
from libs.utils.metrics import intersectionAndUnionGPU


class PostEvaluator(AbstractBaseEvaluator):
    def __init__(self, models, dataloaders, loggers, visualize_samples_datasets, num_image_samples, num_classes,
                 crop_size, use_sliding_window, **kwargs):
        print("Paper Evaluator")
        super().__init__(models, dataloaders, loggers, visualize_samples_datasets, num_image_samples, **kwargs)
        self.model = self.models['segmentation']
        self.dataset_names = list(self.dataloaders.keys())
        self.train_dataset_names = kwargs.get('train_dataset_names', [])
        self.class_idxs = {}
        self.train_mapping_scheme = kwargs['train_mapping_scheme']
        self.val_mapping_scheme = kwargs['val_mapping_scheme']
        self.mapillary_special = kwargs['mapillary_special']
        for name in self.dataset_names:
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
        self.dataset_key = kwargs.get("dataset_key", None)
        self.crop_size = crop_size
        self.use_sliding_window = use_sliding_window
        # self.nuscenes_pred_mapping = None
        # if "nuscenes" in self.dataset_names:
        #     from datasets.dataset_utils import get_nuscenes_prediction_mapping
        #     self.nuscenes_pred_mapping = get_nuscenes_prediction_mapping()

    @torch.no_grad()
    def evaluate(self):
        all_miou = {}
        iou_tables = {}
        dataset_key = self.dataset_key
        dataset_categories = get_dataset_categories(dataset_key)  # TODO find a way to not hardcode this

        for d_name, val_loader in self.dataloaders.items():
            cls_idxs = self.class_idxs[d_name]
            num_classes = len(cls_idxs)
            # num_batches = len(val_loader)

            total_inter, total_union = 0, 0
            crop_size = self.crop_size
            tbar = tqdm(val_loader, desc='\r')
            for batch_idx, (image, target, caption) in enumerate(tbar):
                image, target = image.to(self.device), target.to(self.device)
                b, c, h, w = image.size()
                if self.use_sliding_window:
                    preds = self.sliding_crop_evaluation(crop_size, image)
                else:
                    preds = self.model(image)[0]

                # if d_name == 'nuscenes':
                #     for target_idx, source_idxs in self.nuscenes_pred_mapping.items():
                #         preds[:, target_idx, :, :] = preds[:, source_idxs, :, :].max(1)[0]

                preds = preds[:, cls_idxs, :, :]

                inter, union, _ = intersectionAndUnionGPU(preds.max(1)[1], target, K=num_classes)
                inter, union = inter.cpu().numpy(), union.cpu().numpy()
                total_inter += inter
                total_union += union
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                miou = IoU.mean()
                tbar.set_description('{}: mIoU: {:.3f}. ImgSize: [{}, {}]'.format(d_name, miou, w, h))
            all_miou[d_name] = miou

            iou_table = wandb.Table(columns=["Class name", "IoU"])
            for idx, iou in zip(cls_idxs, IoU):
                iou_table.add_data(dataset_categories[idx], "{:.3f}".format(iou))
            iou_table.add_data("All", "{:.3f}".format(miou))
            iou_tables[d_name] = iou_table
            curr_dataset_cats = get_dataset_categories(d_name)
            # names = ['building', 'bus','car', 'bicycle','fence','motorcycle','person','pole','rider','road','sidewalk',
            #         'sky', 'terrain','traffic light','traffic sign','train','truck','vegetation','wall']
            # vals = []
            # for n in names:
            #     i = sorted(curr_dataset_cats).index(n)
            #     val = IoU[i]
            #     vals.append(val)
            #
            # for val in vals:
            #     print('{:.1f} &'.format(val*100))

        results = {}
        for d_name in self.dataset_names:
            results[d_name + "_mIoU"] = all_miou[d_name]
            results[d_name + "_IoU_table"] = iou_tables[d_name]
        return results

    def sliding_crop_evaluation(self, crop_size, image):
        b, c, h, w = image.size()
        all_pred = torch.zeros(b, self.num_classes, h, w).to(self.device)
        num_horizontals = (w // crop_size) + 1
        num_verticals = (h // crop_size) + 1
        if w % crop_size == 0:
            num_horizontals -= 1
        if h % crop_size == 0:
            num_verticals -= 1
        overlap_w = num_horizontals * crop_size - w
        overlap_h = num_verticals * crop_size - h
        overlap_w_per_crop = overlap_w // (num_horizontals - 1)
        overlap_h_per_crop = overlap_h // (num_verticals - 1)
        for j in range(num_verticals):
            start_x = 0
            start_y = crop_size * j - overlap_h_per_crop * j
            for i in range(num_horizontals):
                curr_x = start_x + crop_size
                curr_y = start_y + crop_size
                if curr_x > w:
                    start_x = w - crop_size
                    curr_x = w
                if curr_y > h:
                    start_y = h - crop_size
                    curr_y = w
                curr_img = image[:, :, start_y:curr_y, start_x:curr_x]
                curr_pred = self.model(curr_img)[0]
                all_pred[:, :, start_y:curr_y, start_x:curr_x] += curr_pred
                start_x = curr_x - overlap_w_per_crop
        return all_pred

    @staticmethod
    def _map_gt_values_with_dict(input_tensor, mapping_dict):
        out = input_tensor.clone()
        for source_val, target_val in mapping_dict.items():
            out[input_tensor == source_val] = target_val
        return out
