import torch
import torch.nn as nn
import torch.nn.functional as F

from refinement import SignalRefinementSubNetwork
from reconstruction import SignalReconstructionSubNetwork

class SRRN(nn.Module):
    def __init__(self):
        super(SRRN, self).__init__()
        self.refinement = SignalRefinementSubNetwork()
        self.reconstruction = SignalReconstructionSubNetwork()

    def forward(self, multi_band, spatial_temporal_map):
        C, K, R, T = multi_band.shape
        multi_band = multi_band.view(C * K, T, R)
        # [B, C*K + C, T, R]
        x = torch.cat([multi_band, spatial_temporal_map], dim=1)

        x = self.refinement(x)
        x = self.reconstruction(x)
        return x
