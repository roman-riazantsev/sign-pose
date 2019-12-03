import torch
import numpy as np

from configs.config_projective import CONFIG_PROJECTIVE
from models.model_projective import ModelProjective
#
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_v0 import CONFIG_V0
from datasets.frei_dataset_v0 import FreiHandVideoDataset
from models.model_u0 import ModelU0
from models.model_v0.model_v0 import ModelV0
from trainers.trainer_u0 import TrainerU0
from trainers.trainer_v0 import TrainerV0
from transforms.to_tensor_u0 import ToTensor
import numpy as np


def get_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A = torch.tensor([[95.5140, 6.3689, 6.1803],
                      [110.6639, -28.9414, 4.4762],
                      [119.5424, -54.9427, 18.2492],
                      [137.0819, -70.2538, 32.0437],
                      [144.9314, -89.9586, 59.5302],
                      [171.8986, -38.8672, 23.5651],
                      [200.3899, -46.0596, 38.5124],
                      [198.2221, -57.1665, 57.5840],
                      [186.7307, -72.3618, 70.4729],
                      [178.5609, -22.2365, 41.1710],
                      [202.2075, -24.8915, 62.0991],
                      [189.4097, -38.9878, 75.4409],
                      [169.1707, -54.2943, 80.1873],
                      [165.1613, -4.0509, 55.3695],
                      [175.1544, -17.5514, 78.7594],
                      [164.4818, -31.4912, 96.2139],
                      [155.1297, -50.4788, 105.9110],
                      [150.3887, 7.2596, 66.6637],
                      [144.4910, -5.9284, 82.0104],
                      [137.2938, -17.6871, 94.9699],
                      [132.4863, -32.7465, 113.6388]]).to(device)
    b = torch.tensor([[81.6912, 136.1244],
                      [93.0584, 109.9410],
                      [100.1987, 91.8695],
                      [113.6654, 81.9669],
                      [119.1983, 70.1710],
                      [139.4812, 104.4189],
                      [160.4549, 100.6390],
                      [157.9915, 94.5050],
                      [148.5879, 87.4507],
                      [143.7224, 117.3830],
                      [160.1517, 116.7967],
                      [150.6411, 107.7159],
                      [138.0303, 96.6124],
                      [133.7154, 130.7154],
                      [140.2368, 122.3100],
                      [132.6299, 113.7178],
                      [123.9726, 100.2617],
                      [123.5582, 139.2142],
                      [119.5465, 130.4882],
                      [114.5742, 123.0273],
                      [109.8051, 112.0727]]).to(device)

    return A, b


if __name__ == '__main__':
    A, b = get_data()

    config = CONFIG_V0

    build_id = config['build_id']

    dataset = FreiHandVideoDataset(
        config['dataset_path'],
        config['data_version'],
        transform=ToTensor()
    )

    sample = dataset[124][0]

    xyz = sample['xyz'].reshape([21, 3])
    K = sample['K']
    uv = sample['uv']

    print(K)
    print(xyz)
    print(uv)
    print(sample['mano'])

    model = ModelProjective(CONFIG_PROJECTIVE)
    model.fit(A, b)
