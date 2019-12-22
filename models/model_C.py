import torch
import torch.nn as nn
import torch.nn.functional as F

from manopth.manolayer import ManoLayer


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


class Encoder9(nn.Module):
    def __init__(self):
        super().__init__()
        self.mano_encoder = nn.Sequential(
            LinearNormedReLUBlock(63, 512),
            LinearNormedReLUBlock(512, 1024),
            LinearNormedReLUBlock(1024, 1024),
            LinearNormedReLUBlock(1024, 1024),
            LinearNormedReLUBlock(1024, 1024),
            LinearNormedReLUBlock(1024, 1024),
            LinearNormedReLUBlock(1024, 1024),
            LinearNormedReLUBlock(1024, 1024),
            LinearNormedReLUBlock(1024, 512),
            LinearNormedReLUBlock(512, 512),
            nn.Linear(512, 51)
        )

    def forward(self, x):
        return self.mano_encoder(x)


class Encoder7(nn.Module):
    def __init__(self):
        super().__init__()
        self.mano_encoder = nn.Sequential(
            LinearNormedReLUBlock(63, 512),
            LinearNormedReLUBlock(512, 1024),
            LinearNormedReLUBlock(1024, 1024),
            LinearNormedReLUBlock(1024, 1024),
            LinearNormedReLUBlock(1024, 1024),
            LinearNormedReLUBlock(1024, 1024),
            LinearNormedReLUBlock(1024, 512),
            LinearNormedReLUBlock(512, 512),
            nn.Linear(512, 51)
        )

    def forward(self, x):
        return self.mano_encoder(x)


class ModelC(nn.Module):
    def __init__(self, depth):
        super().__init__()

        if depth == 7:
            self.mano_encoder = Encoder7()
        else:
            self.mano_encoder = Encoder9()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mano_decoder = ManoLayer(
            mano_root='mano/models',
            use_pca=False,
            ncomps=45,
            flat_hand_mean=False
        ).to(self.device)

    def forward(self, x):
        mano_vector = self.mano_encoder(x)
        pose = mano_vector[..., :48]
        shape = mano_vector[..., 48:]

        batch_size = x.shape[0]

        new_shape = torch.zeros([batch_size, 10], dtype=torch.float32).to(self.device)

        new_shape[:, :3] = shape[:, :3]

        # new_shape.to(self.device)

        hand_verts, hand_joints = self.mano_decoder(pose, new_shape)
        hand_joints = hand_joints.reshape([-1, 63])
        return hand_verts, hand_joints, pose, shape
