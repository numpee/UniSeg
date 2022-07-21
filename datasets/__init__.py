from collections import OrderedDict

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.dataset_utils import get_label_2_train
from datasets.multi_datasets import ConcatMultiDataset, CutPasteMixupDataset
from transformations import transform_factory
from .bdd import BDDSegmentation
from .camvid import CamvidSegmentation, CamvidSegmentationV2
from .cityscapes import CityscapesSegmentation
from .idd import IDDSegmentation
from .kitti import KittiSegmentation
from .mapillary import MAPILLARYSegmentation
from .wilddash import WildDashSegmentation

datasets = {
    'cityscapes': CityscapesSegmentation,
    'idd': IDDSegmentation,
    'bdd': BDDSegmentation,
    'kitti': KittiSegmentation,
    'wilddash': WildDashSegmentation,
    'mapillary': MAPILLARYSegmentation,
    'camvid': CamvidSegmentation,
    'camvid_v2': CamvidSegmentationV2
}


def get_num_classes(configs) -> int:
    """
    Returns the correct num classes of the dataset for single case.
    For non-single dataset, you must specify num classes in config file
    """
    dataset_names = configs.dataset_names
    if len(dataset_names) == 1:
        dataset_name = dataset_names[0]
        label2train = np.array(get_label_2_train(dataset_name))
        cls_idxs = np.unique(label2train[:, 1])
        cls_idxs = cls_idxs[cls_idxs != 255].tolist()
        print("{} classes in dataset".format(len(cls_idxs)))
        return len(cls_idxs)
    else:
        raise NotImplementedError("Manually enter num_classes in the config file!")


def dataset_factory(name, split='train', mode='train', joint_transform=None, input_transform=None, mapping_key=None,
                    sampler=None, root_paths=None, **kwargs):
    if root_paths:
        root = root_paths[name.lower()]
        return datasets[name.lower()](root=root, split=split, mode=mode, joint_transform=joint_transform,
                                      input_transform=input_transform, mapping_key=mapping_key, **kwargs)
    return datasets[name.lower()](split=split, mode=mode, joint_transform=joint_transform,
                                  input_transform=input_transform, mapping_key=mapping_key, **kwargs)


def get_combined_dataset(configs, to_combine_datasets):
    combine_method = configs.combine_method
    if combine_method is None:
        return list(to_combine_datasets.values())[0]
    elif combine_method == "concat":
        return ConcatMultiDataset(to_combine_datasets)
    elif combine_method == "cut_paste_mixup":
        print("CutPaste Mixup")
        return CutPasteMixupDataset(to_combine_datasets, configs.dataset_key, configs.cutmix_weighted)
    else:
        raise ValueError("No such combine method: {}".format(combine_method))


def distributed_dataloader_factory(configs):
    """
    Load train and validation dataloaders for the distributed training setup.
    """
    joint_train_transform, input_train_transform, joint_val_transform, input_val_transform = transform_factory(configs)
    joint_transforms = {'train': joint_train_transform, 'val': joint_val_transform}
    input_transforms = {'train': input_train_transform, 'val': input_val_transform}

    # Train Dataset
    train_datasets = OrderedDict()
    dataset_names = configs.dataset_names
    eval_dataset_names = configs.eval_dataset_names

    for d_name in dataset_names:
        if configs.train_mapping_scheme is None:
            train_mapping_key = d_name
        else:
            train_mapping_key = "{}{}".format(d_name, configs.train_mapping_scheme)

        train_dataset = dataset_factory(d_name, split='train', mode='train', joint_transform=joint_transforms['train'],
                                        input_transform=input_transforms['train'], mapping_key=train_mapping_key,
                                        root_paths=configs.dataset_root_paths)
        train_datasets[d_name] = train_dataset

    combined_train_dataset = get_combined_dataset(configs, train_datasets)
    if configs.num_gpus >= 2:
        train_sampler = DistributedSampler(combined_train_dataset, shuffle=True)
    else:
        train_sampler = None
    train_dataloader = DataLoader(combined_train_dataset, batch_size=configs.batch_size_per_gpu,
                                  num_workers=configs.num_workers_per_gpu, pin_memory=True, drop_last=True,
                                  sampler=train_sampler)
    dataloaders = {'train': train_dataloader, 'val': OrderedDict()}

    # Eval dataset
    for d_name in eval_dataset_names:
        if configs.val_mapping_scheme is None:
            val_mapping_key = d_name
        else:
            if d_name in dataset_names:
                mapping_scheme = "_to_comb_ind"
            else:
                mapping_scheme = configs.val_mapping_scheme
            val_mapping_key = "{}{}".format(d_name, mapping_scheme)
        val_dataset = dataset_factory(d_name, split='val', mode='val', joint_transform=joint_transforms['val'],
                                      input_transform=input_transforms['val'], mapping_key=val_mapping_key,
                                      root_paths=configs.dataset_root_paths)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if configs.num_gpus >= 2 else None
        dataloaders["val"][d_name] = DataLoader(val_dataset, batch_size=configs.batch_size_per_gpu, shuffle=False,
                                                num_workers=configs.num_workers_per_gpu, pin_memory=True,
                                                drop_last=False, sampler=val_sampler)

    return dataloaders, train_sampler


def test_dataloader_factory(configs, joint_transforms=None, input_transforms=None, sampler=None, **kwargs):
    val_datasets = OrderedDict()
    dataset_names = configs.eval_dataset_names
    train_dataset_names = configs.dataset_names
    print("Eval dataset: {}".format(dataset_names))
    for d_name in dataset_names:
        if configs.val_mapping_scheme is None:
            val_mapping_key = d_name
        else:
            if d_name in train_dataset_names:
                mapping_scheme = "_to_comb_ind"
            else:
                mapping_scheme = configs.val_mapping_scheme
            val_mapping_key = "{}{}".format(d_name, mapping_scheme)
            if d_name == "mapillary" and configs.mapillary_special_mapping:
                print("Use Mapillary special mapping (merge classes)")
                val_mapping_key = "mapillary_to_cib_ind_special"
        val_dataset = dataset_factory(d_name, split='val', mode='val', joint_transform=joint_transforms['val'],
                                      input_transform=input_transforms['val'], mapping_key=val_mapping_key,
                                      root_paths=configs.dataset_root_paths, **kwargs)
        val_datasets[d_name] = val_dataset

    dataloaders = OrderedDict()
    for d_name, v_dataset in val_datasets.items():
        dataloaders[d_name] = DataLoader(v_dataset, batch_size=configs.batch_size, shuffle=False,
                                         num_workers=configs.num_workers, pin_memory=True, drop_last=False)
    return dataloaders


def post_test_dataloader_factory(configs, joint_transforms=None, input_transforms=None, sampler=None, **kwargs):
    val_datasets = OrderedDict()
    dataset_names = configs.eval_dataset_names
    train_dataset_names = configs.dataset_names
    print("Eval dataset: {}".format(dataset_names))
    for d_name in dataset_names:
        if configs.val_mapping_scheme is None:
            val_mapping_key = d_name
        else:
            if d_name in train_dataset_names:
                mapping_scheme = "_to_comb_ind"
            else:
                mapping_scheme = configs.val_mapping_scheme
            val_mapping_key = "{}{}".format(d_name, mapping_scheme)
            if d_name == "mapillary" and configs.mapillary_special_mapping:
                print("Use Mapillary special mapping (merge classes)")
                val_mapping_key = "mapillary_to_cib_ind_special"
        val_dataset = dataset_factory(d_name, split='val', mode='val', joint_transform=joint_transforms['val'],
                                      input_transform=input_transforms['val'], mapping_key=val_mapping_key,
                                      root_paths=configs.dataset_root_paths, **kwargs)
        val_datasets[d_name] = val_dataset

    dataloaders = OrderedDict()
    for d_name, v_dataset in val_datasets.items():
        dataloaders[d_name] = DataLoader(v_dataset, batch_size=configs.batch_size, shuffle=False,
                                         num_workers=configs.num_workers, pin_memory=True, drop_last=False)
    return dataloaders