import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if torch.cuda.is_available():
            transform = torch.cuda.FloatTensor
        else:
            transform = torch.FloatTensor

        return {'uv': transform(sample['uv']),
                'mano': transform(sample['mano'])}
