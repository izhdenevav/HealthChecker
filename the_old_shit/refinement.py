import torch
import torch.nn as nn
import torch.nn.functional as F

from the_old_shit.tmsc_module import TMSCModule

class SignalRefinementSubNetwork(nn.Module):
    def __init__(self, in_channels):
        super(SignalRefinementSubNetwork, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            TMSCModule(in_channels),
            nn.AvgPool2d(kernel_size=(1,2))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            TMSCModule(in_channels),
            nn.AvgPool2d(kernel_size=(1,2))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            TMSCModule(in_channels),
            nn.AvgPool2d(kernel_size=(1,2))
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            TMSCModule(in_channels)
        )

    def forward(self, x):
        print(x.shape)
        x = self.block1(x)
        print(x.shape)
        x = self.block2(x)
        print(x.shape)
        x = self.block3(x)
        print(x.shape)
        x = self.block4(x)
        print(x.shape)
        return x