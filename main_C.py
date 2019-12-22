import torch
from torch.utils.data import SubsetRandomSampler, DataLoader

import numpy as np
from configs.config_A import CONFIG_A
from configs.config_C import CONFIG_C
from datasets.frei_dataset_s1 import FreiDatasetS1
from datasets.mano_dataset_C import ManoDatasetC

from datasets.mano_xyz_dataset_mae0 import ManoXYZDataset
from models.model_C import ModelC
from models.model_mae0 import ModelMAE0
from models.model_s1 import ModelS1
from models.model_s1_2 import ModelS1_2
from models.model_s1_3 import ModelS1_3
from models.model_u0 import ModelU0
from trainers.trainer_A import TrainerA
from trainers.trainer_C import TrainerC
from trainers.trainer_mae0 import TrainerMAE0
from trainers.trainer_s1 import TrainerS1
from transforms.np_to_tensor import NpToTensor
from utils.data_utils import get_loaders
from utils.utils import init_build_id

if __name__ == '__main__':
    config = init_build_id(CONFIG_C)
    batch_size = config['batch_size']

    print(f"INITIATED BUILD WITH ID: {config['build_id']}")

    test_split = .2
    shuffle_dataset = True
    random_seed = 42
    # Creating data indices for training and validation splits:
    dataset_size = 32560
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_dataset = ManoDatasetC(config['train_dataset_path'], NpToTensor(), train_indices)
    test_dataset = ManoXYZDataset(config['test_dataset_path'], NpToTensor())

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              sampler=test_sampler)

    models = {
        "C0": ModelC(7),
        "C1": ModelC(9)
    }
    trainer = TrainerC(config, train_loader, test_loader, models)
    trainer.train(epochs=10000)