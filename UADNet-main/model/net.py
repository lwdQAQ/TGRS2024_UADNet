from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from utils import pca
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UADNet(nn.Module):
    def __init__(self, R, B, K, drop_out, use_ALDR=0):
        super(UADNet, self).__init__()
        self.use_ALDR = use_ALDR
        self.Encoder = nn.Sequential(
            nn.Conv2d(B, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),

            nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),

            nn.Conv2d(48, R, kernel_size=3, stride=1, padding=1),
            nn.Softmax(),
        )
        self.Decoder = nn.Conv2d(R, B, kernel_size=(1, 1), bias=False)

        if use_ALDR == 0:
            D = R
        elif use_ALDR == 1:
            D = R + 2
        self.Cluster = nn.Sequential(
            nn.Conv2d(D, 16 * K, kernel_size=(1, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(16 * K),
            nn.Dropout(drop_out),

            nn.Conv2d(16 * K, 8 * K, kernel_size=(1, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(8 * K),
            nn.Dropout(drop_out),

            nn.Conv2d(8 * K, K, kernel_size=(1, 1)),
            nn.Softmax(),
        )

    def forward(self, x, x_pca):
        a = self.Encoder(x)
        if self.use_ALDR == 0:
            a_aldr = a
        elif self.use_ALDR == 1:
            a_aldr = torch.cat((a, x_pca), 1)
        x_hat = self.Decoder(a)
        c = self.Cluster(a_aldr)
        return a, a_aldr, x_hat, c

if __name__=='__main__':
    R, B, drop_out, K = 4, 156, 0.2, 8
    device = 'cpu'
    net = UADNet(R, B, K, drop_out, use_ALDR=1)
    x = torch.randn(1, B, 5, 8)
    x_pca = torch.randn(1, 2, 5, 8)
    a, z, x_hat, c = net(x, x_pca)
    print(x.shape)
    print(x_hat.shape)
    print(a.shape)
    print(z.shape)
    print(c.shape)