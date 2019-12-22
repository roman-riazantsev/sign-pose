import torch
from torch.utils.data import SubsetRandomSampler

import numpy as np
from configs.config_A import CONFIG_A
from configs.config_B import CONFIG_B
from datasets.frei_dataset_B import FreiDatasetB
from datasets.frei_dataset_s1 import FreiDatasetS1
from models.model_B import ModelB
from models.model_s1 import ModelS1
from models.model_s1_2 import ModelS1_2
from models.model_s1_3 import ModelS1_3
from models.model_u0 import ModelU0
from trainers.trainer_A import TrainerA
from trainers.trainer_B import TrainerB
from trainers.trainer_s1 import TrainerS1
from transforms.np_to_tensor import NpToTensor
from utils.data_utils import get_loaders
from utils.utils import init_build_id


def init_loaders(is_vid, config):
    # TODO: REFACTOR LOGIC @!
    batch_size = config['batch_size']
    test_split = .2
    shuffle_dataset = True
    random_seed = 42

    dataset_size = 32560
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataset = FreiDatasetB(
        config=config,
        excluded_indices=test_indices,
        transform=NpToTensor(),
        is_vid=is_vid,
    )

    test_dataset = FreiDatasetB(
        config=config,
        excluded_indices=train_indices,
        transform=NpToTensor(),
        is_vid=is_vid,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              sampler=test_sampler)

    return train_loader, test_loader


if __name__ == '__main__':
    config = init_build_id(CONFIG_B)

    print(f"INITIATED BUILD WITH ID: {config['build_id']}")

    train_loader_pic, test_loader_pic = init_loaders(is_vid=False, config=config)
    train_loader_vid, test_loader_vid = init_loaders(is_vid=True, config=config)

    loaders = {
        "B0_PIC": (train_loader_pic, test_loader_pic),
        "B0_VID": (train_loader_vid, test_loader_vid),
        "B1_PIC": (train_loader_pic, test_loader_pic),
        "B1_VID": (train_loader_vid, test_loader_vid),
    }

    models = {
        "B0_VID": ModelB(True, False),
        "B0_PIC": ModelB(False, False),
        "B1_VID": ModelB(True, True),
        "B1_PIC": ModelB(False, True),
    }

    trainer = TrainerB(config, loaders, models)
    trainer.train(config['epochs'])
