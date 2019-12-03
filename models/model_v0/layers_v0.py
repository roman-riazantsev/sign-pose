from torch import nn


class XYZUpsampler(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(63, 64)

        self.conv_upnet = nn.Sequential(
            nn.ConvTranspose2d(1, 2, 2, padding=1, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.reshape(x.size(0), 1, 8, 8)
        x = self.conv_upnet(x)
        return x

