import torch
import torch.nn as nn
import torch.nn.functional as F

from the_old_shit.refinement import SignalRefinementSubNetwork
# from reconstruction import SignalReconstructionSubNetwork

class SRRN(nn.Module):
    def __init__(self, C=3, R=4, K=4, T=300):
        super(SRRN, self).__init__()
        self.refinement = SignalRefinementSubNetwork(in_channels=C*K + C)
        # self.reconstruction = SignalReconstructionSubNetwork()

    def forward(self, multi_band, spatial_temporal_map):
        B, C, K, R, T = multi_band.shape
        multi_band = multi_band.view(B, C * K, R, T)

        # [B, C*K + C, R, T]
        x = torch.cat([multi_band, spatial_temporal_map], dim=1)

        x = self.refinement(x)
        # x = self.reconstruction(x)
        return x
