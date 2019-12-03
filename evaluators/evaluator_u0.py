import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from configs.config_u0 import CONFIG_U0
from datasets.frei_dataset_u0 import FreiHandDataset
from models.model_u0 import ModelU0
from transforms.to_tensor_u0 import ToTensor
from utils.utils import to_numpy, plot_hand

build_id = CONFIG_U0['build_id']

dataset = FreiHandDataset(
    CONFIG_U0['dataset_path'],
    CONFIG_U0['data_version'],
    transform=ToTensor()
)

dataloader = DataLoader(dataset)
model = ModelU0()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(f'results/{build_id}.pt', map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

sample = next(iter(dataloader))
img = sample['img']
y_2d = to_numpy(sample['y_2d'])
uv = to_numpy(sample['uv'])

y_2d_ = model(img)
y_2d_ = to_numpy(y_2d_)
img = to_numpy(img)
img = img[0].transpose((1, 2, 0))
print(img.shape)
fig, ax = plt.subplots()
ax.imshow(img)
plot_hand(ax, uv[0], order='uv')
plt.show()
