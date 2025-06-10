import torch
import torch.nn as nn
import torch.nn.functional as F

# сверточный TMSC модуль
class TMSCModule(nn.Module):
    def __init__(self, in_channels):
        super(TMSCModule, self).__init__()
        reduced_channels = max(1, in_channels // 4)
        out_channels = max(1, in_channels // 3)
        # свертка для уменьшения числа каналов
        self.conv1x1_in = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        # набор сверток разного размера для извлечения признаков разной длины
        self.conv3x1 = nn.Conv2d(reduced_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv5x1 = nn.Conv2d(reduced_channels, out_channels, kernel_size=(5, 1), padding=(2, 0))
        self.conv7x1 = nn.Conv2d(reduced_channels, out_channels, kernel_size=(7, 1), padding=(3, 0))
        # свертка для восстановления количества каналов
        self.conv1x1_out = nn.Conv2d(out_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        # [B, in_channels, R, T]
        x = self.conv1x1_in(x)
        x1 = self.conv3x1(x)
        x2 = self.conv5x1(x)
        x3 = self.conv7x1(x)
        # конкатим выходы сверток как в схеме
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv1x1_out(x)
        return x