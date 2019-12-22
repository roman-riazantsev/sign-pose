import torch
from torch import nn


class Conv2DNormedReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        self.forward_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.forward_path(x)


class LinearNormedReLUBlock(nn.Module):
    def __init__(self, in_neurons, out_neurons):
        super().__init__()
        self.forward_path = nn.Sequential(
            nn.Linear(in_neurons, out_neurons),
            nn.BatchNorm1d(out_neurons),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.forward_path(x)


class FeatureExtractorB(nn.Module):
    def __init__(self, in_neurons):
        super().__init__()
        self.forward_path = nn.Sequential(
            Conv2DNormedReLUBlock(in_neurons, 16, stride=2),
            Conv2DNormedReLUBlock(16, 16),
            Conv2DNormedReLUBlock(16, 16),
            Conv2DNormedReLUBlock(16, 32, stride=2),
            Conv2DNormedReLUBlock(32, 32),
            Conv2DNormedReLUBlock(32, 32),
            Conv2DNormedReLUBlock(32, 64, stride=2),
            Conv2DNormedReLUBlock(64, 64),
            Conv2DNormedReLUBlock(64, 64),
            Conv2DNormedReLUBlock(64, 96, stride=2),
            Conv2DNormedReLUBlock(96, 96),
            Conv2DNormedReLUBlock(96, 96),
            Conv2DNormedReLUBlock(96, 128, stride=2),
            Conv2DNormedReLUBlock(128, 128),
            Conv2DNormedReLUBlock(128, 128),
            Conv2DNormedReLUBlock(128, 196, stride=2),
            Conv2DNormedReLUBlock(196, 196),
            Conv2DNormedReLUBlock(196, 196)
        )

    def forward(self, x):
        return self.forward_path(x)


class FCPart1(nn.Module):
    def __init__(self, is_vid=False):
        super().__init__()

        self.is_vid = is_vid
        if is_vid:
            previous_frame = 22
        else:
            previous_frame = 0

        in_size = 3136 + previous_frame
        hidden_size = 1024 + previous_frame

        self.fc_0 = LinearNormedReLUBlock(in_size, 1024)
        self.fc_1 = LinearNormedReLUBlock(hidden_size, 1024)
        self.fc_2 = LinearNormedReLUBlock(hidden_size, 1024)

    def vid_op(self, features, prev_frame_depth):
        if self.is_vid:
            return torch.cat([features, prev_frame_depth], dim=1)
        else:
            return features

    def forward(self, features, prev_frame_depth):
        x = self.vid_op(features, prev_frame_depth)
        x = self.fc_0(x)
        x = self.vid_op(x, prev_frame_depth)
        x = self.fc_1(x)
        x = self.vid_op(x, prev_frame_depth)
        x = self.fc_2(x)
        return x


class FCPart2(nn.Module):
    def __init__(self, is_supervised=False):
        super().__init__()

        self.is_supervised = is_supervised

        self.fc_0 = LinearNormedReLUBlock(1024, 1024)
        self.fc_1 = LinearNormedReLUBlock(1024, 1024)
        self.fc_2 = LinearNormedReLUBlock(1024, 1024)

    def forward(self, x):
        x_0 = self.fc_0(x)
        x_1 = self.fc_1(x)
        x_2 = self.fc_2(x)

        if self.is_supervised:
            output = []
            for o in x_0, x_1, x_2:
                output.append(o[..., -22:])
            output.append(x_2)
            return output
        else:
            return [x_2]


class ModelB(nn.Module):
    def __init__(self, is_vid, is_supervised):
        super().__init__()
        self.is_supervised = is_supervised

        if is_vid:
            in_depth = 50
        else:
            in_depth = 25

        self.fe = FeatureExtractorB(in_depth)

        self.fc_part_1 = FCPart1(is_vid)
        self.fc_part_2 = FCPart2(is_supervised)

        self.final_layer = nn.Sequential(
            nn.Linear(1024, 22),
            nn.Sigmoid()
        )

    def forward(self, x, prev_frame_depth=None):
        x = self.fe(x)
        batch_size = x.shape[0]
        x = x.reshape([batch_size, -1])
        x = self.fc_part_1(x, prev_frame_depth)
        fc_p2_out = self.fc_part_2(x)

        x = self.final_layer(fc_p2_out[-1])

        if self.is_supervised:
            output = fc_p2_out[:-1]
            output.append(x)
            x = output
        else:
            x = [x]

        return x
