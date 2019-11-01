import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from manopth.manolayer import ManoLayer


class Trainer(object):
    def __init__(self, batch_size, dataloader, model, save_path):
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.model = model
        self.save_path = save_path

        self.writer = SummaryWriter('results/network_5')
        input_example = next(iter(dataloader))['uv']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.writer.add_graph(self.model, input_example)
        self.writer.close()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)

        self.load_state()

        ncomps = 45
        self.mano_layer = ManoLayer(
            mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=False)

        self.mano_layer.to(self.device)
        self.mano_layer.eval()

    def train(self, epochs, save_rate):
        self.running_loss = 0.0

        for epoch in range(epochs):
            for i, sample in enumerate(self.dataloader, 0):
                uv = sample['uv']
                outputs_gt = sample['mano']

                uv = uv.to(self.device)
                outputs_gt = outputs_gt.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(uv) * 3.12

                # losses = []
                # losses.append(self.criterion(outputs, outputs_gt))

                loss = self.criterion(outputs, outputs_gt)

                # poses_pred = outputs[:, :48]  # .unsqueeze(0)
                # shapes_pred = outputs[:, 48:]  # .unsqueeze(0)
                # hand_verts_p, hand_joints_p = self.mano_layer(poses_pred, shapes_pred)
                #
                # poses_gt = outputs_gt[:, :48]
                # shapes_gt = outputs_gt[:, 48:]
                # hand_verts_gt, hand_joints_gt = self.mano_layer(poses_gt, shapes_gt)

                # losses.append(self.criterion(hand_verts_p, hand_verts_gt))
                # losses.append(self.criterion(hand_joints_p, hand_joints_gt))

                # loss = sum(losses)

                loss.backward()
                self.optimizer.step()

                self.running_loss += loss.item()
                self.g_step += 1

            self.running_loss /= 128
            self.save_state()
            self.writer.add_scalar('training loss',
                                   self.running_loss / 128,
                                   self.g_step)
            print(self.running_loss / 128, self.g_step)
            self.running_loss = 0.0

    def save_state(self):
        torch.save({
            'g_step': self.g_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'running_loss': self.running_loss / 128,
        }, self.save_path)

        print(f'Model saved at step {self.g_step} with running loss {self.running_loss / 128}.')

    def load_state(self):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.g_step = checkpoint['g_step'] + 1
            self.running_loss = checkpoint['running_loss']

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Model loaded. g_step: {self.g_step}; running_loss: {self.running_loss}')
        else:
            print(f'File "{self.save_path}" does not exist. Initializing parameters from scratch.')
            self.g_step = 0
            self.running_loss = 0.0
