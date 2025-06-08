import torch
import torch.nn as nn
import torch.nn.functional as F

from tmsc_module import TMSCModule

class SignalRefinementSubNetwork(nn.module):
    def __init__(self, in_channels):
        super(SignalRefinementSubNetwork, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            TMSCModule(),
            nn.AvgPool2d()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            TMSCModule(),
            nn.AvgPool2d()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            TMSCModule(),
            nn.AvgPool2d()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            TMSCModule()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x