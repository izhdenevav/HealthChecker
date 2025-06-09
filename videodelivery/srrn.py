import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ssa_module
from . import tmsc_module

class SRRN(nn.Module):
    def __init__(self, in_channels=3, R=4, T=300):
        super(SRRN, self).__init__()
        self.in_channels = in_channels
        self.R = R
        self.T = T

        self.refinement_modules = nn.ModuleList()
        for i in range(4):
            module = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )
            if i < 3:
                module.add_module("pooling", nn.AvgPool2d(kernel_size=(1, 2)))
            module.add_module("tmsc", tmsc_module.TMSCModule(in_channels))
            self.refinement_modules.append(module)
            if i < 3:
                T = T // 2

        self.reconstruction_modules = nn.ModuleList()
        for _ in range(3):
            module = nn.Sequential(
                ssa_module.SSAModule(in_channels),
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

        if refinement_outputs:
            # [B, C, R, 37]
            ref = refinement_outputs[-1]
            # [B, C, R, 296]
            ref_up = F.interpolate(ref, size=reconstruction_output.shape[2:], mode='bilinear', align_corners=False)
            combined = ref_up + reconstruction_output
        else:
            combined = reconstruction_output

        out = self.final_conv(combined)
        print(f"out before final_conv {out.shape}")
        out = self.global_pool(out).squeeze(2)
        print(f"out before return {out.shape}")
        
        return out.squeeze(1)
