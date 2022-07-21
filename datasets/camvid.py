import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets.base import BaseDataset
from datasets.dataset_utils import get_label_2_train

DEFAULT_CAMVID_MASK_DIR = "labels"
DEFAULT_CAMVID_DATA_ROOT_PATH = "/net/acadia14a/data/dokim/mseg_dataset/Camvid"


class CamvidSegmentation(BaseDataset):
    NUM_CLASS = 19
    BASE_DIR = 'data'
    NAME = 'camvid'

    def __init__(self, root=DEFAULT_CAMVID_DATA_ROOT_PATH, split='val', mode=None, joint_transform=None,
                 input_transform=None, mapping_key='camvid', **kwargs):
        super(CamvidSegmentation, self).__init__(root, split, mode, joint_transform, input_transform)
        _camvid_root = os.path.join(self.root, self.BASE_DIR)

        img_list_path = os.path.join(DEFAULT_CAMVID_DATA_ROOT_PATH, 'camvid_{}_lists.pkl'.format(mode))
        self._load_img_lists(img_list_path)
        mapping_key = "camvid" if mapping_key is None else mapping_key
        self.mapping_list = get_label_2_train(mapping_key)

        # LMDB
        if self.mode == "train":
            lmdb_out_path = os.path.join(DEFAULT_CAMVID_DATA_ROOT_PATH,
                                         'camvid_lmdb_{}_resized.mdb'.format(self.mode))
        else:
            lmdb_out_path = os.path.join(DEFAULT_CAMVID_DATA_ROOT_PATH, 'camvid_lmdb_{}.mdb'.format(self.mode))
        self.lmdb_out_path = self._create_lmdb_database(lmdb_out_path)

    def __len__(self):
        return len(self.images)

class CamvidSegmentationV2(BaseDataset):
    NUM_CLASS = 19
    BASE_DIR = 'data'
    NAME = 'camvid'

    def __init__(self, root=DEFAULT_CAMVID_DATA_ROOT_PATH, split='val', mode=None, joint_transform=None,
                 input_transform=None, mapping_key='camvid', **kwargs):
        super(CamvidSegmentationV2, self).__init__(root, split, mode, joint_transform, input_transform)
        _camvid_root = os.path.join(self.root, self.BASE_DIR)

        img_list_path = os.path.join(DEFAULT_CAMVID_DATA_ROOT_PATH, 'camvid_v2_{}_lists.pkl'.format(mode))
        self._load_img_lists(img_list_path)
        mapping_key = "camvid" if mapping_key is None else mapping_key
        self.mapping_list = get_label_2_train(mapping_key)

        # LMDB
        if self.mode == "train":
            lmdb_out_path = os.path.join(DEFAULT_CAMVID_DATA_ROOT_PATH,
                                         'camvid_lmdb_{}_resized.mdb'.format(self.mode))
        else:
            lmdb_out_path = os.path.join(DEFAULT_CAMVID_DATA_ROOT_PATH, 'camvid_v2_lmdb_{}.mdb'.format(self.mode))
        self.lmdb_out_path = self._create_lmdb_database(lmdb_out_path)

    def __len__(self):
        return len(self.images)


def remap_camvid_labels(root=DEFAULT_CAMVID_DATA_ROOT_PATH, labels_path=DEFAULT_CAMVID_MASK_DIR):
    color_mapping_dict = {
        "sky": [128, 128, 128], "building": [128, 0, 0], "pole": [192, 192, 128], "road": [128, 64, 128],
        "sidewalk": [0, 0, 192], "vegetation": [128, 128, 0], "traffic sign": [192, 128, 128], "fence": [64, 64, 128],
        "car": [64, 0, 128], "person": [64, 64, 0], "bicyclist": [0, 128, 192]}

    labels_dir = os.path.join(root, labels_path)
    new_labels_dir = os.path.join(root, "new_labels")
    if not os.path.exists(new_labels_dir):
        os.mkdir(new_labels_dir)
    label_paths = os.listdir(labels_dir)
    for path in tqdm(label_paths):
        full_img_path = os.path.join(labels_dir, path)
        img = Image.open(full_img_path).convert("RGB")
        img_np = np.array(img)
        w, h, _ = img_np.shape
        mask = np.zeros((w, h)).astype("int32") - 1
        for i, cls_name in enumerate(sorted(list(color_mapping_dict.keys()))):
            pixel_val = color_mapping_dict[cls_name]
            mask1 = img_np[:, :, 0] == pixel_val[0]
            mask2 = img_np[:, :, 1] == pixel_val[1]
            mask3 = img_np[:, :, 2] == pixel_val[2]
            mask_all = mask1 * mask2 * mask3
            mask[mask_all] = i
        new_label = Image.fromarray(mask)
        new_ext_path = path.replace(".png", ".tif")
        new_path = os.path.join(new_labels_dir, new_ext_path)
        new_label.save(new_path)


def remap_camvid_labels_v2(root=DEFAULT_CAMVID_DATA_ROOT_PATH, labels_path=DEFAULT_CAMVID_MASK_DIR):
    color_mapping_dict = {
        "sky": [128, 128, 128], "building": [128, 0, 0], "pole": [192, 192, 128], "road": [128, 64, 128],
        "sidewalk": [0, 0, 192], "vegetation": [128, 128, 0], "traffic sign": [192, 128, 128], "fence": [64, 64, 128],
        "car": [64, 0, 128], "person": [64, 64, 0], "bicyclist": [0, 128, 192],
        "lane marking - general": [128, 0, 192]}

    labels_dir = os.path.join(root, labels_path)
    new_labels_dir = os.path.join(root, "new_labels_v2")
    if not os.path.exists(new_labels_dir):
        os.mkdir(new_labels_dir)
    label_paths = os.listdir(labels_dir)
    for path in tqdm(label_paths):
        full_img_path = os.path.join(labels_dir, path)
        img = Image.open(full_img_path).convert("RGB")
        img_np = np.array(img)
        w, h, _ = img_np.shape
        mask = np.zeros((w, h)).astype("int32") - 1
        for i, cls_name in enumerate(sorted(list(color_mapping_dict.keys()))):
            pixel_val = color_mapping_dict[cls_name]
            mask1 = img_np[:, :, 0] == pixel_val[0]
            mask2 = img_np[:, :, 1] == pixel_val[1]
            mask3 = img_np[:, :, 2] == pixel_val[2]
            mask_all = mask1 * mask2 * mask3
            mask[mask_all] = i
        new_label = Image.fromarray(mask)
        new_ext_path = path.replace(".png", ".tif")
        new_path = os.path.join(new_labels_dir, new_ext_path)
        new_label.save(new_path)


if __name__ == "__main__":
    # remap_camvid_labels()
    remap_camvid_labels_v2()
