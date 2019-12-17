import torch


class NpToTensor(object):
    def __call__(self, sample):
        transform = self.get_system_compatable_transform()

        if isinstance(sample, dict):
            sample = self.map_dict(sample, transform)
        else:
            sample = transform(sample)

        return sample

    @staticmethod
    def get_system_compatable_transform():
        if torch.cuda.is_available():
            transform = torch.cuda.FloatTensor
        else:
            transform = torch.FloatTensor
        return transform

    @staticmethod
    def map_dict(dictionary, transform):
        for k, v in dictionary.items():
            if len(v.shape) == 3:
                v = v.transpose((2, 0, 1))
            dictionary[k] = transform(v)
        return dictionary
