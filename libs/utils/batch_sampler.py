import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
from torchvision import transforms
from PIL import Image


def shuffle_list(data_len=1,batch_size=2):
    # import pdb; pdb.set_trace()
    return_list = []
    lists = []
    batches = []
    en = 0
    for ll in range(len(data_len)):
        st = en
        en = st + data_len[ll]
        id = list(range(st,en))
        random.shuffle(id)
        batch = data_len[ll]//batch_size

        lists.append(id)
        batches.append(batch)

    # import pdb; pdb.set_trace()
    random_list = list(range(sum(batches)))
    random.shuffle(random_list)

    for random_batch_index in random_list:
        num = 0
        for ll in range(len(data_len)):
            if random_batch_index < num + batches[ll]:
                id = lists[ll]
                index = random_batch_index - num
                return_list += id[index*batch_size : (index+1)*batch_size]
                break
            else:
                num += batches[ll]
    # import pdb; pdb.set_trace()
    return return_list


class BatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset_length, batchsize):
        self.dataset_length = dataset_length
        self.batchsize = batchsize

    def __iter__(self):
        list = shuffle_list(self.dataset_length, self.batchsize)
        return iter(list)

    def __len__(self):
        return sum(self.dataset_length)
