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


class ModelS2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            Conv2DNormedReLUBlock(25, 16, stride=2),
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

        self.fc_net = nn.Sequential(
            LinearNormedReLUBlock(3136, 2048),
            LinearNormedReLUBlock(2048, 2048),
            LinearNormedReLUBlock(2048, 2048),
            LinearNormedReLUBlock(2048, 2048),
            nn.Linear(2048, 22),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_net(x)
        batch_size = x.shape[0]
        x = x.reshape([batch_size, -1])
        x = self.fc_net(x)
        return x
