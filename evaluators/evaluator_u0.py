import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config_u0 import CONFIG_U0
from datasets.frei_dataset_u0 import FreiHandDataset
from models.model_u0 import ModelU0
from transforms.to_tensor_u0 import ToTensor

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

img = next(iter(dataloader))['img']
points = model(img)
result = np.sum(points.detach().numpy(), axis=1)[0]

# y_2d = next(iter(dataloader))['y_2d']
# y_2d = np.sum(y_2d.detach().numpy()[0], axis=0)


cv2.imshow('result', result)
cv2.waitKey(-1)
