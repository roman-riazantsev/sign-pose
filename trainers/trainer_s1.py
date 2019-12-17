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


class TrainerS1(object):
    def __init__(self, config, train_loader, test_loader, model):
        self.__dict__.update(config)
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

        mid = self.model_id
        bid = self.build_id

        if not os.path.exists(f'results/experiments/{mid}/{bid}'):
            os.makedirs(f'results/experiments/{mid}/{bid}')

        self.save_path = f'results/experiments/{mid}/{bid}/{bid}.pt'
        self.writer = SummaryWriter(f'results/experiments/{mid}/{bid}')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.load_state()

    def train(self, epochs):
        self.reset_running_losses()
        for epoch in range(epochs):
            start = time.time()
            for i, train_sample in enumerate(tqdm(self.train_loader, 0)):
                self.optimizer.zero_grad()

                train_loss, final_loss, uv_maps_, uv_maps = self.get_loss(self.model, train_sample)

                test_sample = next(iter(self.test_loader))
                _, final_test_loss, _, _ = self.get_loss(self.model, test_sample)

                train_loss.backward()
                self.optimizer.step()

                self.running_train_loss += train_loss.item()
                self.running_final_loss += final_loss.item()
                self.running_final_test_loss += final_test_loss.item()
                self.g_step += 1

                if self.g_step % self.save_rate == self.save_rate - 1:
                    self.save_state()
                    self.writer.add_scalar('running_train_loss',
                                           self.running_train_loss / self.save_rate,
                                           self.g_step)

                    self.writer.add_scalar('running_final_loss',
                                           self.running_final_loss / self.save_rate,
                                           self.g_step)

                    self.writer.add_scalar('running_final_test_loss',
                                           self.running_final_test_loss / self.save_rate,
                                           self.g_step)

                    self.writer.add_image('predictions vs. actuals',
                                          self.render_results(uv_maps[0], uv_maps_[0]),
                                          global_step=self.g_step)

                    print(self.running_train_loss / self.save_rate, self.g_step)
                    self.reset_running_losses()

                    print(len(self.train_loader) / self.save_rate)
                    print("it took", time.time() - start, "seconds. ", i)

    def get_loss(self, model, sample):
        input_img = sample['img']
        uv_maps = sample['uv_maps']
        uv_maps_list = model(input_img)

        train_loss = self.criterion(uv_maps_list[-1], uv_maps)
        final_loss = train_loss

        for uv_maps_ in uv_maps_list[:-1]:
            train_loss += self.criterion(uv_maps_, uv_maps)

        return train_loss, final_loss, uv_maps_list[-1], uv_maps

    # def maps_to_uv(self, maps):
    #     maps = maps[:, :-1, ...]
    #     batch_size, depth, height, width = maps.shape
    #     flatten_maps = maps.reshape(batch_size, maps.shape[1], -1)
    #     idx = flatten_maps.argmax(2)
    #     x_ = idx % height
    #     y_ = idx / height
    #     uv = torch.stack((x_, y_), 1).transpose(2, 1).float()
    #     return uv.to(self.device)

    def save_state(self):
        torch.save({
            'g_step': self.g_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'running_loss': self.running_train_loss / self.save_rate,
        }, self.save_path)

        print(f'Model saved at step {self.g_step} with running loss {self.running_train_loss / self.save_rate}.')

    def load_state(self):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.g_step = checkpoint['g_step'] + 1
            self.running_train_loss = checkpoint['running_loss']

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Model "{self.build_id}" loaded. g_step: {self.g_step}; running_loss: {self.running_train_loss}')
        else:
            print(f'File "{self.save_path}" does not exist. Initializing parameters from scratch.')
            self.g_step = 0

    def reset_running_losses(self):
        self.running_train_loss = 0.0
        self.running_final_loss = 0.0
        self.running_final_test_loss = 0.0

    @staticmethod
    def render_results(y_2d, y_2d_):
        y_2d = to_numpy(y_2d)
        y_2d_ = to_numpy(y_2d_)
        point_pairs = np.column_stack((y_2d, y_2d_))
        im_to_show = point_pairs[1::4]
        res = np.concatenate(im_to_show, axis=1)
        res = np.expand_dims(res, axis=0)
        return res
