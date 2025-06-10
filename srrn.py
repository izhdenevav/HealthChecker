import torch
import torch.nn as nn
import torch.nn.functional as F

from ssa_module import SSAModule 
from tmsc_module import TMSCModule

class SRRN(nn.Module):
    # по дефолту
    # R - количество зон = 4
    # T - количество кадров = 300 (10 секунд)  
    def __init__(self, in_channels=3, R=4, T=300):
        super(SRRN, self).__init__()
        self.in_channels = in_channels
        self.R = R
        self.T = T

        # объявляем подсеть фильтрации, состоящую из 4 блоков
        self.refinement_modules = nn.ModuleList()
        for i in range(4):
            module = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )

            # только первые три блока содержат AvgPool
            if i < 3:
                module.add_module("pooling", nn.AvgPool2d(kernel_size=(1, 2)))
            
            # заключающим в каждом блоке идет сверточный TMSC модуль
            module.add_module("tmsc", TMSCModule(in_channels))

            self.refinement_modules.append(module)

            if i < 3:
                T = T // 2

        # объявляем подсеть восстановления сигнала, которая состоит также из 4 блоков
        self.reconstruction_modules = nn.ModuleList()
        for _ in range(3):
            module = nn.Sequential(
                SSAModule(in_channels),
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(in_channels),
                nn.ELU()
            )
            self.reconstruction_modules.append(module)
            T = T * 2 

        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, T))

    def forward(self, x):
        refinement_outputs = []
        for i, module in enumerate(self.refinement_modules):
            x = module(x)
            if i < 3:
                refinement_outputs.append(x)

        reconstruction_output = x
        for module in self.reconstruction_modules:
            reconstruction_output = module(reconstruction_output)

        # учитывая конкаты в схеме, идущие из блоков подсети фильтрации в блоки подсети восстановления 
        if refinement_outputs:
            # [B, C, R, 37]
            ref = refinement_outputs[-1]
            # [B, C, R, 296]
            ref_up = F.interpolate(ref, size=reconstruction_output.shape[2:], mode='bilinear', align_corners=False)
            combined = ref_up + reconstruction_output
        else:
            combined = reconstruction_output

        out = self.final_conv(combined)
        out = self.global_pool(out).squeeze(2)
        
        return out.squeeze(1)