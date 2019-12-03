from pprint import pprint

from torch.utils.data import DataLoader

from datasets.mano_xyz_dataset_mae0 import ManoXYZDataset
from models.model_mae0 import ModelMAE0
from trainers.trainer_mae0 import TrainerMAE0
from transforms.np_to_tensor import NpToTensor

from configs.config_mae0 import CONFIG_MAE0
from models.model_m0 import ModelM0
from trainers.trainer_m0 import TrainerM0

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

if __name__ == '__main__':
    build_id = CONFIG_MAE0['build_id']

    dataset = ManoXYZDataset(
        CONFIG_MAE0['dataset_path'],
        transform=NpToTensor()
    )

    batch_size = CONFIG_MAE0['batch_size']
    test_split = .2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=test_sampler)

    model = ModelMAE0()
    trainer = TrainerMAE0(CONFIG_MAE0['batch_size'], train_loader, test_loader, model, build_id,
                          CONFIG_MAE0['save_rate'])
    trainer.train(epochs=10000)
