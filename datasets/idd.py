import os

from .base import BaseDataset
from .dataset_utils import get_label_2_train

DEFAULT_IDD_DATA_ROOT_PATH = "/net/acadia14a/data/dokim/mseg_dataset/IDD/IDD_Segmentation"


class IDDSegmentation(BaseDataset):
    NUM_CLASS = 26
    NAME = 'idd'

    def __init__(self, root=DEFAULT_IDD_DATA_ROOT_PATH, split='train', mode=None, joint_transform=None,
                 input_transform=None, mapping_key='idd', **kwargs):
        super(IDDSegmentation, self).__init__(root, split, mode, joint_transform, input_transform)

        img_list_path = os.path.join(DEFAULT_IDD_DATA_ROOT_PATH, 'idd_{}_lists.pkl'.format(mode))
        self._load_img_lists(img_list_path)
        mapping_key = "idd" if mapping_key is None else mapping_key
        self.mapping_list = get_label_2_train(mapping_key)

        # LMDB
        lmdb_out_path = os.path.join(DEFAULT_IDD_DATA_ROOT_PATH, 'idd_lmdb_{}_resized.mdb'.format(self.mode))
        self.lmdb_out_path = self._create_lmdb_database(lmdb_out_path)

    def __len__(self):
        return len(self.images)
