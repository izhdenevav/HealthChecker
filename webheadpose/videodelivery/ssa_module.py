import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dct, idct
import numpy as np

def apply_dct(signal, axis=-1, norm='ortho'):
    signal_np = signal.cpu().detach().numpy()
    dct_result = dct(signal_np, axis=axis, norm=norm)
    return torch.tensor(dct_result, device=signal.device)

class SSAModule(nn.Module):
    # m - размерность после DCT
    def __init__(self, in_channels, m=64):
        super(SSAModule, self).__init__()
        inter_channels = max(1, in_channels // 4)

        self.m = m
        self.in_channels = in_channels

        self.conv1x1_relu = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.ReLU()
        )
        self.conv1x1_sigmoid = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels * 4, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, C, R, T = x.shape

        # [batch_size, C, R, T]
        x_dct = apply_dct(x, axis=-1)

        Q = x_dct.view(batch_size, C * R, T)
        K = x_dct.view(batch_size, C * R, T)
        V = x_dct.view(batch_size, C * R, T)

        # [batch_size, T, T]
        attention_scores = torch.bmm(Q.transpose(1, 2), K) / (T ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        # [batch_size, T, C*R]
        attention_output = torch.bmm(attention_weights, V.transpose(1, 2))
        attention_output = attention_output.transpose(1, 2).view(batch_size, C, R, T)

        weights = self.conv1x1_relu(attention_output)
        weights = self.conv1x1_sigmoid(weights)

        x = x * weights[:, :C, :, :T]
        return x