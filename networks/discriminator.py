# -*- coding:utf-8 -*-
# @Time: 2020/5/4 21:38
# @Author: libra
# @Site: A binary classification network that outputs a scalar probability
# @File: discriminator.py
# @Software: PyCharm

import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ndf, nc=3):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(nc,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8,1,4,1,0,bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)