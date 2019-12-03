import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from manopth.manolayer import ManoLayer
from utils.fh_utils import db_size
from utils.utils import to_numpy


class TrainerV0(object):
    def __init__(self, batch_size, dataloader, model, build_id, save_rate):
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.model = model
        self.save_rate = save_rate
        self.build_id = build_id

        self.save_path = f'results/{build_id}.pt'

        self.writer = SummaryWriter(f'results/{build_id}')
        input_example = next(iter(dataloader))[0]['img']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # self.writer.add_graph(self.model, input_example)
        # self.writer.close()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002)

        self.load_state()

    def train(self, epochs):
        self.running_loss = 0.0

        for epoch in range(epochs):
            for i, (sample_0, sample_1) in enumerate(tqdm(self.dataloader, 0)):
                frame_0 = sample_0['img'].to(self.device)
                frame_1 = sample_1['img'].to(self.device)

                frame_0_2d = sample_0['y_2d'].to(self.device)
                frame_1_2d = sample_1['y_2d'].to(self.device)

                frame_0_xyz = sample_0['xyz'].to(self.device)
                frame_1_xyz = sample_1['xyz'].to(self.device)

                X = [frame_0, frame_1, frame_0_2d, frame_1_2d, frame_0_xyz]

                self.optimizer.zero_grad()

                frame_1_xyz_ = self.model(X)

                loss = self.criterion(frame_1_xyz_, frame_1_xyz)
                print(loss.item())

                loss.backward()
                self.optimizer.step()

                # print()
                # print(frame_1_xyz_)
                # print(frame_1_xyz)
                self.running_loss += loss.item()
                self.g_step += 1

                if self.g_step % self.save_rate == self.save_rate - 1:
                    self.save_state()
                    self.writer.add_scalar('training loss',
                                           self.running_loss / self.save_rate,
                                           self.g_step)
                    print(self.running_loss / self.save_rate, self.g_step)
                    self.running_loss = 0.0

    def save_state(self):
        torch.save({
            'g_step': self.g_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'running_loss': self.running_loss / self.save_rate,
        }, self.save_path)

        print(f'Model saved at step {self.g_step} with running loss {self.running_loss / self.save_rate}.')

    def load_state(self):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.g_step = checkpoint['g_step'] + 1
            self.running_loss = checkpoint['running_loss']

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Model "{self.build_id}" loaded. g_step: {self.g_step}; running_loss: {self.running_loss}')
        else:
            print(f'File "{self.save_path}" does not exist. Initializing parameters from scratch.')
            self.g_step = 0
            self.running_loss = 0.0

    @staticmethod
    def render_results(y_2d, y_2d_):
        y_2d = to_numpy(y_2d)
        y_2d_ = to_numpy(y_2d_)
        point_pairs = np.column_stack((y_2d, y_2d_))
        im_to_show = point_pairs[1::4]
        res = np.concatenate(im_to_show, axis=1)
        res = np.expand_dims(res, axis=0)
        return res
