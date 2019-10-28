import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'uv': torch.FloatTensor(sample['uv']),
                'mano': torch.FloatTensor(sample['mano'])}