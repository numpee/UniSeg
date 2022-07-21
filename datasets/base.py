import pickle

import lmdb
import numpy as np
import os
import six
import torch
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm

__all__ = ['BaseDataset']


class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, joint_transform=None, input_transform=None):
        self.root = root
        self.joint_transform = joint_transform
        self.input_transform = input_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.images = []
        self.masks = []
        self.names = []
        self.lmdb = None
        self.lmdb_out_path = ""
        self.mapping_list = []

    def __getitem__(self, index):
        img_path, mask_path = self.images[index], self.masks[index]
        img, target = self._get_img_mask_from_lmdb(img_path, mask_path)
        img, target = self._sync_transform(img, target)
        return img, target, self.names[index]

    def get_raw_samples(self, index):
        img_path, mask_path = self.images[index], self.masks[index]
        img, target = self._get_img_mask_from_lmdb(img_path, mask_path)
        return img, target, self.names[index]

    def get_sample_paths(self, index):
        img_path, mask_path = self.images[index], self.masks[index]
        return img_path, mask_path

    def _create_lmdb_database(self, lmdb_out_path):
        img_path_exists = os.path.exists(lmdb_out_path)

        if img_path_exists:
            print("File already exists. Not generating new LMDB database")
            return lmdb_out_path

        with lmdb.open(lmdb_out_path, map_size=510 * 1e9).begin(write=True) as txn:
            print("Creating LMDB. \nPath: {}".format(lmdb_out_path))
            for img_path, mask_path in tqdm(zip(self.images, self.masks)):
                raw_img_bytes = open(img_path, 'rb').read()
                raw_mask_bytes = open(mask_path, 'rb').read()
                txn.put(img_path.encode(), raw_img_bytes)
                txn.put(mask_path.encode(), raw_mask_bytes)
        return lmdb_out_path

    def _init_lmdb(self):
        self.lmdb = lmdb.open(self.lmdb_out_path, readonly=True, lock=False, readahead=False, meminit=False)

    def _get_img_mask_from_lmdb(self, img_path, mask_path):
        if self.lmdb is None:
            self._init_lmdb()
        img_buf = self.lmdb.begin(write=False).get(img_path.encode())
        mask_buf = self.lmdb.begin(write=False).get(mask_path.encode())
        buf = six.BytesIO()
        buf.write(img_buf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        buf = six.BytesIO()
        buf.write(mask_buf)
        buf.seek(0)
        target = Image.open(buf)
        return img, target

    def _sync_transform(self, img, mask):
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.input_transform is not None:
            img = self.input_transform(img)
        return img, self._mask_transform(mask)

    def label_mapping(self, input, mapping):
        output = np.copy(input)
        for ind in range(len(mapping)):
            output[input == mapping[ind][0]] = mapping[ind][1]
        return np.array(output, dtype=np.int32)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        mapping = np.array(self.mapping_list, dtype=np.int)
        target = self.label_mapping(target, mapping)

        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def _load_img_lists(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.images = data['images']
        self.masks = data['masks']
        self.names = data['names']
