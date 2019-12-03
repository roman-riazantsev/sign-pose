import torch
#
from models.model_v0.layers_v0 import XYZUpsampler

import torch
from torch import nn

from models.model_v0.layers_v0 import XYZUpsampler


class ModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.xyz_upsampler = XYZUpsampler()

        self.hidden_forward_path = nn.Sequential(
            nn.Conv2d(52, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.end_part = nn.Sequential(
            nn.Linear(1152, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 63)
        )

    def forward(self, X):
        frame_0, frame_1, frame_0_2d, frame_1_2d, frame_0_xyz = X
        xyz_upsampled = self.xyz_upsampler(frame_0_xyz)
        x = torch.cat((frame_0, frame_1, frame_0_2d, frame_1_2d, xyz_upsampled), 1)

        x = self.hidden_forward_path(x)
        x = x.reshape(x.size(0), -1)
        x = self.end_part(x)
        return x

