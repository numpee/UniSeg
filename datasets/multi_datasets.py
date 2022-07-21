import pickle
import random
from collections import OrderedDict
from itertools import accumulate

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage

from datasets.dataset_utils import get_dataset_categories, get_mixable_categories
from transformations.joint_transforms import CenterCropPad


class ConcatMultiDataset(Dataset):
    """
    Concat multiple datasets into a single dataset_configs.
    Sample from the concatenated dataset_configs, as well as the membership of the sample (which dataset_configs it came from)
    """

    def __init__(self, datasets):
        if type(datasets) == OrderedDict:
            self.names = list(datasets.keys())
            self.datasets = list(datasets.values())
        elif type(datasets) == list:
            self.names = [i for i in range(0, len(datasets))]
            self.datasets = datasets
        self.num_datasets = len(self.datasets)
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]
        self._accumulated_len = list(accumulate(self.dataset_lengths))

    def __getitem__(self, index):
        membership = None
        for i, l in enumerate(self._accumulated_len):
            if index < l:
                membership = i
                break

        offset_index = index - self._accumulated_len[membership]
        sample = self.datasets[membership][offset_index]
        return sample, self.names[membership]

    def __len__(self):
        return self._accumulated_len[-1]

    def get_dataset_names(self) -> list:
        return self.names


class CutPasteMixupDataset(Dataset):
    def __init__(self, datasets, dataset_key='city_idd_bdd_mapillary', weighted=False, num_mix_classes=3):
        self.dataset_key = dataset_key
        if type(datasets) == OrderedDict:
            self.names = list(datasets.keys())
            self.datasets = list(datasets.values())
        elif type(datasets) == list:
            self.names = [d.NAME for d in datasets]
            self.datasets = datasets
        # Cache transformations to list and set no joint transformation
        self.joint_transformations = []
        self.input_transformations = []
        for d in self.datasets:
            self.joint_transformations.append(d.joint_transform)
            self.input_transformations.append(d.input_transform)
            d.joint_transform = None
            d.input_transform = None
        self.num_datasets = len(self.datasets)
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]
        self._accumulated_len = list(accumulate(self.dataset_lengths))
        self.tensorize = ToTensor()
        self.to_pil = ToPILImage()
        self.center_crop_pad = CenterCropPad(size=1000, ignore_index=-1)
        self.all_categories = get_dataset_categories(dataset_key)
        self.category_to_idx_mapping = {name: i for i, name in enumerate(self.all_categories)}
        self.mixable_categories = get_mixable_categories(dataset_key)
        self.mixable_category_weights = {}
        self.num_mix_classes = num_mix_classes
        if weighted:
            print("Using class weights for mixing")
        self.dataset_to_category_mapping = self.get_pickle_data(
            "/home/dongwan/universal-semantic-seg/CIBM_dataset_to_category_mapping_v2.pkl")
        self.classes_each_sample = self.get_pickle_data(
            "/home/dongwan/universal-semantic-seg/CIBM_img_category_info.pkl")
        self.mixable_categories_per_dataset = {d_name: [] for d_name in self.names}
        for d_name in self.names:
            present_class_names = []

            for cls_idx, samples_list in self.dataset_to_category_mapping[d_name].items():
                if len(samples_list) > 0:
                    present_class_names.append(self.all_categories[cls_idx])
            # print("{}: {}".format(d_name, present_class_names))
            mixable_cats = list(set(self.mixable_categories).intersection(set(present_class_names)))
            # print("{}: {}".format(d_name, mixable_cats))
            self.mixable_categories_per_dataset[d_name] = mixable_cats
            if weighted:
                weight_data = self.get_pickle_data(
                    "/home/dongwan/universal-semantic-seg/new_{}_class_weights.pkl".format(d_name))
                print(weight_data.keys())
                weights = []
                for cat in mixable_cats:
                    if cat not in weight_data:
                        print("{}: {} not in dict".format(d_name, cat))
                        weights.append(0)
                    else:
                        weight = weight_data[cat]
                        weights.append(weight)
                sum_weights = sum(weights)
                weights = [w / sum_weights for w in weights]
                # assert sum(weights) == 1
                self.mixable_category_weights[d_name] = weights
            else:
                self.mixable_category_weights[d_name] = None

    def __getitem__(self, index, use_transform=True):
        membership = None
        mem_list = list(range(self.num_datasets))
        for i, l in enumerate(self._accumulated_len):
            if index < l:
                membership = i
                mem_list.pop(i)
                break

        offset_index = index - self._accumulated_len[membership]
        img, mask, name = self.datasets[membership][offset_index]
        # Sample from random dataset other than current dataset
        rand_dataset_idx = random.choice(mem_list)
        rand_dataset = self.names[rand_dataset_idx]
        mixable_classes = self.mixable_categories_per_dataset[rand_dataset]
        rand_cls = np.random.choice(mixable_classes,
                                    p=self.mixable_category_weights[rand_dataset])
        rand_cls_idx = self.category_to_idx_mapping[rand_cls]
        source_sample_list = self.dataset_to_category_mapping[rand_dataset][rand_cls_idx]
        source_sample_idx = random.choice(source_sample_list) if len(source_sample_list) > 0 else None

        if source_sample_idx is not None:
            source_img, source_mask, source_name = self.datasets[rand_dataset_idx].get_raw_samples(source_sample_idx)
            self.center_crop_pad.set_size(img.size)

            cropped_source_img, cropped_source_mask = self.center_crop_pad(source_img, source_mask)
            img, cropped_source_img = self.tensorize(img), self.tensorize(cropped_source_img)
            cropped_source_mask = self.datasets[rand_dataset_idx]._mask_transform(cropped_source_mask)
            mixup_mask = torch.zeros_like(cropped_source_mask).float()
            mixup_mask[cropped_source_mask == rand_cls_idx] = 1
            mixed_img = img * (1 - mixup_mask) + cropped_source_img * mixup_mask
            mask = mask.float()
            mask = mask * (1 - mixup_mask) + rand_cls_idx * mixup_mask
            # mask[mixup_mask == 1] = rand_cls_idx
            mixed_img = self.to_pil(mixed_img)
            mixed_img_tensor, mixed_mask_tensor = self._sync_transform(mixed_img, self._mask_to_pil(mask), membership)
            return (mixed_img_tensor, mixed_mask_tensor, rand_cls_idx), self.names[membership]
        else:
            img, mask = self._sync_transform(img, self._mask_to_pil(mask), membership)
            return (img, mask, ""), self.names[membership]

    def __len__(self):
        return self._accumulated_len[-1]

    def get_dataset_names(self):
        return self.names

    def _sync_transform(self, img: Image, mask: Image, domain_idx: int):
        joint_transform = self.joint_transformations[domain_idx]
        input_transform = self.input_transformations[domain_idx]
        if joint_transform is not None:
            img, mask = joint_transform(img, mask)
        if input_transform is not None:
            img = input_transform(img)
        return img, self._mask_to_tensor(mask)

    def _mask_to_pil(self, mask):
        return Image.fromarray(np.array(mask.type(torch.int)))

    def _mask_to_tensor(self, mask):
        return torch.from_numpy(np.array(mask).astype('int32')).long()

    @staticmethod
    def get_pickle_data(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
