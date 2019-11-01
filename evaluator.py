import torch
import torch
from manopth.manolayer import ManoLayer
from manopth import demo

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
checkpoint = torch.load('results/network_5.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


# # max: 3.1128
# # min: ~ -3.12

sample = dataset[1]
uv = sample['uv']
mano_true = sample['mano']
mano_pred = model(uv) * 3.12

print(mano_true)
print("-------")
print(mano_pred)

poses = mano_pred[:48].unsqueeze(0)
shapes = mano_pred[48:].unsqueeze(0)

batch_size = 1
# Select number of principal components for pose space
ncomps = 45

# Initialize MANO layer
mano_layer = ManoLayer(
    mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mano_layer.to(device)

hand_verts, hand_joints = mano_layer(poses, shapes)
demo.display_hand({
    'verts': hand_verts.cpu().detach(),
    'joints': hand_joints.cpu().detach()
},
    mano_faces=mano_layer.th_faces)

poses_true = mano_true[:48].unsqueeze(0)
shapes_true = mano_true[48:].unsqueeze(0)

hand_verts, hand_joints = mano_layer(poses_true, shapes_true)
demo.display_hand({
    'verts': hand_verts.cpu().detach(),
    'joints': hand_joints.cpu().detach()
},
    mano_faces=mano_layer.th_faces)
