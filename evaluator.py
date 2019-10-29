import torch

from config import CONFIG
from frei_dataset import FreiDataset
from network_1 import Network1
from utils import ToTensor

dataset = FreiDataset(
    CONFIG['dataset_path'],
    transform=ToTensor()
)

device = torch.device("cuda")
model = Network1()
checkpoint = torch.load('results/network_1.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
#
#
# # max: 3.1128
# # min: ~ -3.12

sample = dataset[1]
uv = sample['uv']
mano_true = sample['mano']
mano_pred = model(uv)

print(mano_true)
print("-------")
print(mano_pred)

