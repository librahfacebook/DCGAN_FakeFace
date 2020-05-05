# -*- coding:utf-8 -*-
# @Time: 2020/5/4 23:19
# @Author: libra
# @Site: 
# @File: dataloader.py
# @Software: PyCharm

import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset,DataLoader

class CelebaDataset(Dataset):
    def __init__(self, root,transform=None):
        self.root = root
        self.transform = transform
        self.imgs = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root,self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


def get_dataloader(data_root,image_size,batch_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = CelebaDataset(root=data_root,transform=transform)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=2)
    return dataloader

