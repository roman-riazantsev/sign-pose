import pickle

import numpy as np
from torch.utils.data import Dataset


class ManoXYZDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.transform = transform

        with open(base_path, 'rb') as saved_dict:
            self.dataset_records = pickle.load(saved_dict)

    def __len__(self):
        return self.dataset_records['poses'].shape[0]

    def __getitem__(self, idx):
        sample = {}
        for k, v in self.dataset_records.items():
            sample[k] = v[idx]

        sample['hand_joints'] = sample['hand_joints'].flatten()

        if self.transform:
            sample = self.transform(sample)

        return sample
