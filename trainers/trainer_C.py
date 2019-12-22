import time
from pprint import pprint

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


class TrainerC(object):
    def __init__(self, config, train_loader, test_loader, models):

        self.__dict__.update(config)

        self.models = models
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for net_id in self.models.keys():
            self.models[net_id].to(self.device)

        self.init_train_results()
        self.init_paths_and_writers()
        self.init_optimizers()
        self.criterion = nn.MSELoss()
        self.load_state()

    def train(self, epochs):
        self.reset_running_losses()
        for epoch in range(epochs):
            for i, _ in enumerate(tqdm(self.train_loader, 0)):
                self.train_step()
                if self.g_step % self.save_rate == self.save_rate - 1:
                    self.save_state()
                    self.write_logs()
                    self.reset_running_losses()

                self.g_step += 1

    def train_step(self):
        for net_id in self.models.keys():
            optimizer = self.optimizers[net_id]
            optimizer.zero_grad()

            train_sample = next(iter(self.train_loader))
            test_sample = next(iter(self.test_loader))

            train_loss = self.get_loss(self.models[net_id], train_sample)

            train_loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            test_loss = self.get_loss(self.models[net_id], test_sample)
            self.train_results[net_id]['running_train_loss'] += train_loss.item()
            self.train_results[net_id]['running_test_loss'] += test_loss.item()

    def get_loss(self, model, sample):
        hand_joints = sample['hand_joints']
        hand_verts = sample['hand_verts']
        poses = sample['poses']
        shapes = sample['shapes'][..., :3]

        hand_verts_, hand_joints_, poses_, shapes_ = model(hand_joints)

        loss_hand_joints = self.criterion(hand_joints_, hand_joints)
        loss_hand_verts = self.criterion(hand_verts_, hand_verts)
        loss_poses = self.criterion(poses_, poses)
        loss_shapes = self.criterion(shapes_, shapes)

        loss = loss_hand_joints + loss_hand_verts + loss_poses + loss_shapes
        return loss

    def init_paths_and_writers(self):
        mid = self.model_id
        bid = self.build_id
        base_path = f'results/experiments/{mid}/{bid}'
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        self.save_path = '/'.join([base_path, f'{bid}.pt'])
        self.writers = {}

        for net_id in self.models.keys():
            writer_path = '/'.join([base_path, net_id])
            self.writers[net_id] = SummaryWriter(writer_path)

    def save_state(self):
        save_dict = {}
        for net_id, net in self.models.items():
            save_dict[net_id] = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': self.optimizers[net_id].state_dict()
            }
        save_dict['g_step'] = self.g_step

        torch.save(save_dict, self.save_path)
        print(f'Models saved at step {self.g_step}.')

    def load_state(self):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.g_step = checkpoint['g_step'] + 1

            for net_id, net in self.models.items():
                net_saves = checkpoint[net_id]
                net.load_state_dict(net_saves['model_state_dict'])
                self.optimizers[net_id].load_state_dict(net_saves['optimizer_state_dict'])

            print(f'Models loaded; g_step: {self.g_step}.')
        else:
            print(f'File "{self.save_path}" does not exist. Initializing parameters from scratch.')
            self.g_step = 0

    def init_optimizers(self):
        self.optimizers = {}
        for net_id, net in self.models.items():
            o = optim.Adam(net.parameters(), lr=self.lr)
            self.optimizers[net_id] = o

    def reset_running_losses(self):
        for net_id, net in self.models.items():
            self.train_results[net_id]['running_train_loss'] = 0.
            self.train_results[net_id]['running_test_loss'] = 0.

    def write_logs(self):
        for net_id in self.models.keys():
            tr = self.train_results[net_id]
            self.writers[net_id].add_scalar('train_loss',
                                            tr['running_train_loss'] / self.save_rate,
                                            self.g_step)

            self.writers[net_id].add_scalar('test_loss',
                                            tr['running_test_loss'] / self.save_rate,
                                            self.g_step)

    def init_train_results(self):
        self.train_results = {}
        for net_id in self.models.keys():
            self.train_results[net_id] = {}
