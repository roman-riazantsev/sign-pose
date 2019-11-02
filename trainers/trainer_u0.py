import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from manopth.manolayer import ManoLayer
from utils.fh_utils import db_size


class TrainerU0(object):
    def __init__(self, batch_size, dataloader, model, build_id, save_rate):
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.model = model
        self.save_rate = save_rate

        self.save_path = f'results/{build_id}.pt'

        self.writer = SummaryWriter(f'results/{build_id}')
        input_example = next(iter(dataloader))['img']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.writer.add_graph(self.model, input_example)
        self.writer.close()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)

        self.load_state()

    def train(self, epochs):
        self.running_loss = 0.0

        for epoch in range(epochs):
            for i, sample in enumerate(tqdm(self.dataloader, 0)):
                x = sample['img']
                y_2d = sample['y_2d']

                x = x.to(self.device)
                y_2d = y_2d.to(self.device)

                self.optimizer.zero_grad()

                y_2d_ = self.model(x)

                loss = self.criterion(y_2d_, y_2d)

                loss.backward()
                self.optimizer.step()

                self.running_loss += loss.item()
                self.g_step += 1

                if self.g_step % self.save_rate == self.save_rate - 1:
                    self.running_loss /= self.save_rate
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
            print(f'Model loaded. g_step: {self.g_step}; running_loss: {self.running_loss}')
        else:
            print(f'File "{self.save_path}" does not exist. Initializing parameters from scratch.')
            self.g_step = 0
            self.running_loss = 0.0
