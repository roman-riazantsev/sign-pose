import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


def get_loaders(batch_size, dataset):
    test_split = .2
    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=test_sampler)

    return train_loader, test_loader
