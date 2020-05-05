# -*- coding:utf-8 -*-
# @Time: 2020/5/4 21:17
# @Author: libra
# @Site: Generator is designed to map the latent space vector to data-space
# @File: generator.py
# @Software: PyCharm

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc=3):
        """
        :param nz: size of latent vector
        :param ngf: size of feature maps in generator
        :param nc: number of channels
        """
        super(Generator,self).__init__()
        self.convTrans1 = nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False)
        self.batch1 = nn.BatchNorm2d(ngf*8)
        self.relu = nn.ReLU(True)
        self.convTrans2 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False)
        self.batch2 = nn.BatchNorm2d(ngf*4)
        self.convTrans3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.batch3 = nn.BatchNorm2d(ngf * 2)
        self.convTrans4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.batch4 = nn.BatchNorm2d(ngf)
        self.convTrans5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.batch1(self.convTrans1(x)))
        out = self.relu(self.batch2(self.convTrans2(out)))
        out = self.relu(self.batch3(self.convTrans3(out)))
        out = self.relu(self.batch4(self.convTrans4(out)))
        out = self.tanh(self.convTrans5(out))

        return out
