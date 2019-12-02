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


class TrainerS2(object):
    def __init__(self, batch_size, train_loader, model, build_id, save_rate):
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.model = model
        self.save_rate = save_rate
        self.build_id = build_id

        self.save_path = f'results/s2/{build_id}/{build_id}.pt'
        self.writer = SummaryWriter(f'results/s2/{build_id}')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)
        self.load_state()

    def train(self, epochs):
        self.running_train_loss = 0.0

        for epoch in range(epochs):
            for i, train_sample in enumerate(tqdm(self.train_loader, 0)):
                self.optimizer.zero_grad()

                training_loss = self.get_loss(self.model, train_sample)

                training_loss.backward()
                self.optimizer.step()

                self.running_train_loss += training_loss.item()
                self.g_step += 1
                if self.g_step % self.save_rate == self.save_rate - 1:
                    self.save_state()

                    self.writer.add_scalar('training loss',
                                           self.running_train_loss / self.save_rate,
                                           self.g_step)

                    self.running_train_loss = 0.0

    def get_loss(self, model, sample):
        X = sample['X']
        depth = sample['depth']

        depth_ = model(X)

        loss = self.criterion(depth_, depth)
        return loss

    def save_state(self):
        train_loss = self.running_train_loss / self.save_rate

        torch.save({
            'g_step': self.g_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
        }, self.save_path)
        print()
        print(f'Model saved at step {self.g_step} with train loss {train_loss}.')

    def load_state(self):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.g_step = checkpoint['g_step'] + 1
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            train_loss = checkpoint['train_loss']
            print(f'Model "{self.build_id}" loaded. g_step: {self.g_step}; '
                  f'train_loss: {train_loss};')
        else:
            print(f'File "{self.save_path}" does not exist. Initializing parameters from scratch.')
            self.g_step = 0
