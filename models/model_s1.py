import torch
from torch import nn

from models.layers_s import FeatureExtractor, ConvUp2DNormedReLUBlock, Conv2DNormedReLUBlock, UvEstimator, \
    HourglassModuleS1
import torch.nn.functional as F


class ModelS1(nn.Module):
    def __init__(self, stack_depth, input_depth):
        super().__init__()
        self.hourglass_0 = HourglassModuleS1(input_depth)

        self.refinements_nets = []
        for d in range(stack_depth - 1):
            self.refinements_nets.append(HourglassModuleS1(input_depth + 22))

    def forward(self, x):
        h0_res = self.hourglass_0(x)
        results = [h0_res]

        for net in self.refinements_nets:
            h_in = torch.cat([results[-1], x], 1)
            h_res = net(h_in)
            results.append(h_res)

        return results
