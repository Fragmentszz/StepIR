import os
from queue import LifoQueue
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

import torch
import cv2
from utils.img_utils import crop_img, data_augmentation, random_crop


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

        # 随机切128x128的patch

        

        # 插值为512x512

        hq_image, lq_image = random_crop(lq_image, hq_image, crop_size=128)
        
        # hq_image = cv2.resize(hq_image, (128, 128), interpolation=cv2.INTER_CUBIC)
        # lq_image = cv2.resize(lq_image, (128, 128), interpolation=cv2.INTER_CUBIC)

        

        if self.mode == 'train':
                # data augmentation
                flag_aug = random.randint(0, 7)
                hq_image = data_augmentation(hq_image, flag_aug)
                lq_image = data_augmentation(lq_image, flag_aug)

        hq_image = crop_img(np.array(hq_image), base=16)
        lq_image = crop_img(np.array(lq_image), base=16)

        hq_image = self.transforms(hq_image)
        lq_image = self.transforms(lq_image)



        return hq_image, lq_image
        

import os
class TestDataset(Dataset):
    def __init__(self, data_root):
        super(TestDataset, self).__init__()
        self.data_root = data_root

        self.lq_images = []
        for group in os.listdir(self.data_root):
            if group == 'HQ':
                continue
            group_path = os.path.join(self.data_root, group)
            if os.path.isdir(group_path):
                for degradations in os.listdir(group_path):
                    if "jpeg compression" in degradations or "low resolution" in degradations:
                        continue

                    degradations_path = os.path.join(group_path, degradations)
                    if os.path.isdir(degradations_path):
                        for image_name in os.listdir(degradations_path):
                            image_path = os.path.join(degradations_path, image_name)
                            self.lq_images.append(image_path)
        print(f"Loaded {len(self.lq_images)} LQ images from {self.data_root}")
    def __len__(self):
        return len(self.lq_images)
    def __getitem__(self, index):
        lq_path = self.lq_images[index]
        lq_image = Image.open(lq_path).convert('RGB')
        lq_image = np.array(lq_image)
        lq_image = crop_img(np.array(lq_image), base=16)
        lq_image = ToTensor()(lq_image)


        hq_path = os.path.join(self.data_root, 'HQ', os.path.basename(lq_path))
        hq_image = Image.open(hq_path).convert('RGB')
        hq_image = np.array(hq_image)
        hq_image = crop_img(np.array(hq_image), base=16)
        hq_image = ToTensor()(hq_image)




        return lq_image, hq_image


if __name__ == '__main__':
    data_root = '/home/wanglixin/datasets/LSDIR'
    dataset = FinetuneDataset(data_root)
    for i in range(10):
        hq_image, lq_image = dataset[i]
        print(hq_image.shape, lq_image.shape)
        # break


    test_data_root = '/home/wanglixin/datasets/LSDIR/test'

    test_dataset = TestDataset(test_data_root)
    for i in range(10):
        lq_image, hq_image = test_dataset[i]
        print(lq_image.shape, hq_image.shape)
        # break