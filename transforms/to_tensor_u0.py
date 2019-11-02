import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if torch.cuda.is_available():
            transform = torch.cuda.FloatTensor
        else:
            transform = torch.FloatTensor

        for key in sample:
            if key in ['img', 'y_2d']:
                sample[key] = sample[key].transpose((2, 0, 1))
            sample[key] = transform(sample[key])

        return sample
