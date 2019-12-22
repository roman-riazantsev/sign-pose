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


class EvaluatorB(object):
    def __init__(self, config, loaders, models):
        self.__dict__.update(config)
        self.subsets = ['train', 'test', 'val', 'val_augmented']
        self.models = models
        self.loaders = loaders

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for net_id in self.models.keys():
            self.models[net_id].to(self.device)
            self.models[net_id].eval()

        self.mse_loss = nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.load_models()

        self.init_evaluation_results()

    def evaluate(self):
        for subset in self.subsets:
            for i, _ in enumerate(tqdm(self.loaders["PIC"]["val"], 0)):
                for net_id in self.models.keys():
                    is_vid = net_id.endswith('VID')
                    data_type = net_id[-3:]
                    sample = self.get_samples(data_type, subset)

                    mse, l_1, renorm_loss = self.get_losses(self.models[net_id], sample, is_vid)

                    self.evaluation_results[subset][net_id]['mse'].append(to_numpy(mse))
                    self.evaluation_results[subset][net_id]['l_1'].append(to_numpy(l_1))
                    self.evaluation_results[subset][net_id]['renorm_loss'].append(to_numpy(renorm_loss))

            saves_path = '/'.join([self.save_to, f'evaluation_{subset}.pickle'])
            with open(saves_path, 'wb') as handle:
                pickle.dump(self.evaluation_results[subset], handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_losses(self, model, sample, is_vid):
        X = sample['X']
        Y = sample['Y']

        if is_vid:
            prev_frame_depth = sample['prev_depth']
            Y_pred_list = model(X, prev_frame_depth)
        else:
            Y_pred_list = model(X)

        mse_loss = self.mse_loss(Y_pred_list[-1], Y)
        l1_loss = self.l1_loss(Y_pred_list[-1], Y)

        renorm_loss = self.get_renormalization_loss(Y_pred_list[-1], Y)

        return mse_loss, l1_loss, renorm_loss

    def get_renormalization_loss(self, Y_, Y):
        def renormalize(depth_array):
            max_depth = depth_array[-1]
            normed_depth = depth_array[:-1]
            renormed = normed_depth * max_depth
            return renormed

        renormed_depth_ = renormalize(Y_[0])
        renormed_depth = renormalize(Y[0])

        renormalization_loss = self.l1_loss(renormed_depth_, renormed_depth)
        return renormalization_loss

    def get_samples(self, data_type, subset):
        loaders_ = self.loaders[data_type]

        loader = loaders_[subset]

        sample = next(iter(loader))

        return sample

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
        for subset in self.subsets:
            self.evaluation_results[subset] = {}
            for net_id in self.models.keys():
                self.evaluation_results[subset][net_id] = {
                    'mse': [],
                    'l_1': [],
                    'renorm_loss': []
                }
