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


class TrainerB(object):
    def __init__(self, config, loaders, models):
        self.__dict__.update(config)

        self.models = models
        self.loaders = loaders

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
            for i, _ in enumerate(tqdm(self.loaders["B0_PIC"][0], 0)):
                self.train_step()
                if self.g_step % self.save_rate == self.save_rate - 1:
                    self.save_state()
                    self.write_logs()
                    self.reset_running_losses()

                self.g_step += 1

    def train_step(self):
        for net_id in self.models.keys():
            is_vid = net_id.endswith('VID')
            optimizer = self.optimizers[net_id]
            optimizer.zero_grad()

            train_sample = next(iter(self.loaders[net_id][0]))
            test_sample = next(iter(self.loaders[net_id][1]))
            losses_train = self.get_loss(self.models[net_id], train_sample, is_vid)

            mse, train_loss, mse_output, renorm_loss_output = losses_train

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses_test = self.get_loss(self.models[net_id], test_sample, is_vid)
            mse_test, _, mse_output_test, renorm_loss_output_test = losses_test

            self.train_results[net_id]['running_train_loss'] += mse.item()
            self.train_results[net_id]['new_running_train_loss'] += train_loss.item()
            self.train_results[net_id]['running_train_output_loss'] += mse_output.item()
            self.train_results[net_id]['running_renorm_train'] += renorm_loss_output.item()

            self.train_results[net_id]['running_test_loss'] += mse_test.item()
            self.train_results[net_id]['running_test_output_loss'] += mse_output_test.item()
            self.train_results[net_id]['running_renorm_test'] += renorm_loss_output_test.item()

    def get_loss(self, model, sample, is_vid):
        X = sample['X']
        Y = sample['Y']
        if is_vid:
            prev_frame_depth = sample['prev_depth']
            Y_pred_list = model(X, prev_frame_depth)
        else:
            Y_pred_list = model(X)

        plain_mse, _, train_loss = self.get_renormalization_loss(Y_pred_list[-1], Y)
        plain_mse_output, renormalization_loss_output, _ = self.get_renormalization_loss(Y_pred_list[-1], Y)

        for Y_ in Y_pred_list[:-1]:
            plain_mse_a, _, train_loss_a = self.get_renormalization_loss(Y_, Y)
            plain_mse += plain_mse_a
            train_loss += train_loss_a

        losses = [plain_mse, train_loss, plain_mse_output, renormalization_loss_output]

        return losses

    def get_renormalization_loss(self, Y_, Y):
        def renormalize(depth_array):
            max_depth = depth_array[-1]
            normed_depth = depth_array[:-1]
            renormed = normed_depth * max_depth
            return renormed

        plain_mse = self.criterion(Y_, Y)
        print(Y_, Y)
        renormed_depth_ = renormalize(Y_)
        renormed_depth = renormalize(Y)

        renormalization_loss = self.criterion(renormed_depth_, renormed_depth)

        total_loss = plain_mse + renormalization_loss
        return plain_mse, renormalization_loss, total_loss

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
            self.train_results[net_id]['new_running_train_loss'] = 0.
            self.train_results[net_id]['running_train_output_loss'] = 0.
            self.train_results[net_id]['running_renorm_train'] = 0.

            self.train_results[net_id]['running_test_loss'] = 0.
            self.train_results[net_id]['running_test_output_loss'] = 0.
            self.train_results[net_id]['running_renorm_test'] = 0.

    def write_logs(self):
        for net_id in self.models.keys():
            tr = self.train_results[net_id]
            self.writers[net_id].add_scalar('running_train_loss',
                                            tr['running_train_loss'] / self.save_rate,
                                            self.g_step)
            self.writers[net_id].add_scalar('new_running_train_loss',
                                            tr['new_running_train_loss'] / self.save_rate,
                                            self.g_step)
            self.writers[net_id].add_scalar('running_train_output_loss',
                                            tr['running_train_output_loss'] / self.save_rate,
                                            self.g_step)
            self.writers[net_id].add_scalar('running_renorm_train',
                                            tr['running_renorm_train'] / self.save_rate,
                                            self.g_step)

            self.writers[net_id].add_scalar('running_test_loss',
                                            tr['running_test_loss'] / self.save_rate,
                                            self.g_step)
            self.writers[net_id].add_scalar('running_test_output_loss',
                                            tr['running_test_output_loss'] / self.save_rate,
                                            self.g_step)
            self.writers[net_id].add_scalar('running_renorm_test',
                                            tr['running_renorm_test'] / self.save_rate,
                                            self.g_step)

    def init_train_results(self):
        self.train_results = {}
        for net_id in self.models.keys():
            self.train_results[net_id] = {}

    @staticmethod
    def get_depth_info(xyz):
        depth = xyz[..., 2].copy()
        depth = (depth - depth.min())
        max_depth = depth.max()
        depth_normed = depth / max_depth
        depth_info = np.append(depth_normed, max_depth)
        return depth_info
