import torch
from torch import nn


class Conv2DNormedReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.forward_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.forward_path(x)


class DownsapleStep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.forward_path = nn.Sequential(
            Conv2DNormedReLUBlock(in_channels, out_channels, stride=2),
            Conv2DNormedReLUBlock(out_channels, out_channels),
            Conv2DNormedReLUBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.forward_path(x)


class UpsamleStep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.forward_path = nn.Sequential(
            ConvUp2DNormedReLUBlock(in_channels, out_channels),
            Conv2DNormedReLUBlock(out_channels, out_channels),
            Conv2DNormedReLUBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.forward_path(x)


class FeatureExtractor(nn.Module):
    def __init__(self, input_depth):
        super().__init__()
        self.dss_0 = DownsapleStep(input_depth, 22)
        self.dss_1 = DownsapleStep(22, 32)
        self.dss_2 = DownsapleStep(32, 64)
        self.dss_3 = DownsapleStep(64, 96)
        self.dss_4 = DownsapleStep(96, 128)
        self.dss_5 = DownsapleStep(128, 196)

    def forward(self, x):
        dss_0_res = self.dss_0(x)
        dss_1_res = self.dss_1(dss_0_res)
        dss_2_res = self.dss_2(dss_1_res)
        dss_3_res = self.dss_3(dss_2_res)
        dss_4_res = self.dss_4(dss_3_res)
        dss_5_res = self.dss_5(dss_4_res)
        results = [dss_0_res, dss_1_res, dss_2_res, dss_3_res, dss_4_res, dss_5_res]
        return results


class ResidualBlock(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.conv_1 = Conv2DNormedReLUBlock(depth, depth)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(depth, depth, 3, padding=1),
            nn.BatchNorm2d(depth)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x_in):
        x = self.conv_1(x_in)
        x = self.conv_2(x)
        x = x + x_in
        x = self.relu(x)
        return x


class UvEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_block = ResidualBlock(196)
        self.uss_0 = nn.Sequential(
            ConvUp2DNormedReLUBlock(196, 128),
            Conv2DNormedReLUBlock(128, 128, kernel_size=2, padding=0),
            Conv2DNormedReLUBlock(128, 128),
        )
        self.uss_1 = UpsamleStep(128, 96)
        self.uss_2 = UpsamleStep(96, 64)
        self.uss_3 = UpsamleStep(64, 32)
        self.uss_4 = UpsamleStep(32, 22)
        self.uss_5 = UpsamleStep(22, 22)
        self.final_conv = nn.Conv2d(22, 22, kernel_size=3, padding=1)

    def forward(self, features):
        res_block_res = self.res_block(features[-1])
        uss_0_res = self.uss_0(res_block_res)
        uss_1_in = features[-2] + uss_0_res
        uss_1_res = self.uss_1(uss_1_in)
        uss_2_in = features[-3] + uss_1_res
        uss_2_res = self.uss_2(uss_2_in)
        uss_3_in = features[-4] + uss_2_res
        uss_3_res = self.uss_3(uss_3_in)
        uss_4_in = features[-5] + uss_3_res
        uss_4_res = self.uss_4(uss_4_in)
        uss_5_in = features[-6] + uss_4_res
        uss_5_res = self.uss_5(uss_5_in)
        final_conv_res = self.final_conv(uss_5_res)
        return final_conv_res


class ConvUp2DNormedReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=0):
        super().__init__()
        self.forward_path = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.forward_path(x)


class HourglassModuleS1(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_shape)
        self.uv_estimator = UvEstimator()

    def forward(self, x):
        features = self.feature_extractor(x)
        x = self.uv_estimator(features)
        return torch.softmax(x, dim=1)
