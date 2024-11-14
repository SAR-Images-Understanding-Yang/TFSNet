#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :sample_dataset.py
# @Time      :2024/6/3 9:27
# @Author    :Yangxinpeng
# @Introduce : a dataset code for sample dataset, it loads the images and attribution scattering centers

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import scipy.io as sio
from PIL import Image


class SAMPLEDataset(ImageFolder):
    def __init__(self,
                 root=None,
                 transform=None):
        if root is None:
            raise ValueError
        super(SAMPLEDataset, self).__init__(root, transform)

    # 重载 __getitem__ 函数来包含文件路径
    def __getitem__(self, index):
        result = super().__getitem__(index)
        # 文件路径
        image_path = self.imgs[index][0]
        image_format = image_path[-3:]

        ASC_path = image_path.replace(image_path.split('\\')[-3], image_path.split('\\')[-3] + "-ASC").replace(
            image_format, 'mat')
        ASC = sio.loadmat(ASC_path)['XX']
        ASC = torch.tensor(ASC, dtype=torch.float32)
        result = (result + (ASC,))
        # imaging, label, ASC
        return result
