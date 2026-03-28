import os
from queue import LifoQueue
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

import torch

from utils.img_utils import crop_img, data_augmentation


import json

class FinetuneDataset(Dataset):
    def __init__(self, data_root, mode='train'):
        super(FinetuneDataset, self).__init__()
        self.data_root = data_root
        self.degraded_json_path = os.path.join(self.data_root, 'degradation_info.json')
        self.mode = mode

        self.transforms = Compose(
            [ToTensor()]
        )

        with open(self.degraded_json_path, 'r', encoding='utf-8') as f:
            self.degradation_info = []
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.degradation_info.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"degradation_info parse error at line {i}: {e}")

        self.images_pairs = self.degradation_info
        
        print(f"Loaded {len(self.images_pairs)} image pairs from {self.degraded_json_path}")

    def __len__(self):
        return len(self.images_pairs)
            
    def __getitem__(self, index):
        item = self.images_pairs[index]
        hq_path = os.path.join(self.data_root, item['hq_path'])
        lq_path = os.path.join(self.data_root, item['lq_path'])

        degradations = item['degradations']

        hq_image = Image.open(hq_path).convert('RGB')
        lq_image = Image.open(lq_path).convert('RGB')

        
        # 先转 numpy
        hq_image = np.array(hq_image)
        lq_image = np.array(lq_image)


        if self.mode == 'train':
                # data augmentation
                flag_aug = random.randint(0, 7)
                hq_image = data_augmentation(hq_image, flag_aug)
                lq_image = data_augmentation(lq_image, flag_aug)

        hq_image = crop_img(np.array(hq_image), base=16)
        lq_image = crop_img(np.array(lq_image), base=16)

        hq_image = self.transforms(hq_image)
        lq_image = self.transforms(lq_image)



        return hq_image, lq_image, degradations
        



if __name__ == '__main__':
    data_root = '/home/wanglixin/datasets/LSDIR'
    dataset = FinetuneDataset(data_root)
    for i in range(10):
        hq_image, lq_image, degradations = dataset[i]
        print(hq_image.shape, lq_image.shape, degradations)
        # break