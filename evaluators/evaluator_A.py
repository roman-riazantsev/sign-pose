import pickle
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


class EvaluatorA(object):
    def __init__(self, config, loaders, models):
        self.__dict__.update(config)

        self.models = models
        self.loaders = loaders

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for net_id in self.models.keys():
            self.models[net_id].to(self.device)

        self.mse_loss = nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.load_models()

        self.init_evaluation_results()

    def evaluate(self):
        for subset in ['val']:
            for i, _ in enumerate(tqdm(self.loaders["PIC"]["val"], 0)):
                for net_id in self.models.keys():
                    data_type = net_id[-3:]
                    sample = self.get_samples(data_type, subset, i)

                    mse_maps, uv_maps_ = self.get_mse_loss(self.models[net_id], sample)
                    l1, mse = self.get_uv_errors(uv_maps_, sample)

                    self.evaluation_results[subset][net_id]['mse_maps'].append(to_numpy(mse_maps))
                    self.evaluation_results[subset][net_id]['l1'].append(to_numpy(l1))
                    self.evaluation_results[subset][net_id]['mse'].append(to_numpy(mse))

            saves_path = '/'.join([self.save_to, f'evaluation_{subset}.pickle'])
            with open(saves_path, 'wb') as handle:
                pickle.dump(self.evaluation_results[subset], handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_uv_errors(self, uv_maps_, sample):
        uv = sample['uv']
        uv_ = self.maps_to_uv(uv_maps_)
        l1_loss = self.l1_loss(uv_, uv)
        mse = self.mse_loss(uv_, uv)

        return l1_loss, mse

    def get_samples(self, data_type, subset, i):
        loaders_ = self.loaders[data_type]

        loader = loaders_[subset]

        sample = next(iter(loader))

        return sample

    def get_mse_loss(self, model, sample):
        X = sample['X']
        uv_maps = sample['uv_maps']
        uv_maps_list_ = model(X)
        uv_maps_ = uv_maps_list_[-1]
        loss = self.mse_loss(uv_maps_, uv_maps)

        return loss, uv_maps_

    def maps_to_uv(self, maps):
        maps = maps[:, :-1, ...]
        batch_size, depth, height, width = maps.shape
        flatten_maps = maps.reshape(batch_size, maps.shape[1], -1)
        idx = flatten_maps.argmax(2)
        x_ = idx % height
        y_ = idx / height
        uv = torch.stack((x_, y_), 1).transpose(2, 1).float()
        return uv.to(self.device)

    def load_models(self):
        if os.path.exists(self.weights_path):
            checkpoint = torch.load(self.weights_path)
            self.g_step = checkpoint['g_step']

            for net_id, net in self.models.items():
                net_saves = checkpoint[net_id]
                net.load_state_dict(net_saves['model_state_dict'])

            print(f'Models loaded; g_step: {self.g_step}.')
        else:
            print(f'File "{self.weights_path}" does not exist.')

    def init_evaluation_results(self):
        self.evaluation_results = {}
        for subset in ['train', 'test', 'val']:
            self.evaluation_results[subset] = {}
            for net_id in self.models.keys():
                self.evaluation_results[subset][net_id] = {
                    'mse_maps': [],
                    'l1': [],
                    'mse': []
                }
