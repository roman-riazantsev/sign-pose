import torch
from torch import nn

from models.layers_s import FeatureExtractor, ConvUp2DNormedReLUBlock, Conv2DNormedReLUBlock, UvEstimator, \
    HourglassModuleS1
import torch.nn.functional as F


class ModelS1_3(nn.Module):
    def __init__(self, input_depth):
        super().__init__()
        self.hourglass_0 = HourglassModuleS1(input_depth)
        self.hourglass_1 = HourglassModuleS1(input_depth + 22)
        self.hourglass_2 = HourglassModuleS1(input_depth + 22)

    def forward(self, x):
        h0_res = self.hourglass_0(x)
        h1_in = torch.cat([h0_res, x], dim=1)
        h1_res = self.hourglass_1(h1_in)
        h2_in = torch.cat([h1_res, x], dim=1)
        h2_res = self.hourglass_2(h2_in)

        return h0_res, h1_res, h2_res
