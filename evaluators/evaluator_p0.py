import torch
from manopth.manolayer import ManoLayer
from manopth import demo

from configs.config_p0 import CONFIG_P0
from datasets.frei_dataset_p0 import FreiDataset
from models.model_p0 import ModelP0
from transforms.to_tensor_p0 import ToTensorP0

dataset = FreiDataset(
    CONFIG_P0['dataset_path'],
    transform=ToTensorP0()
)

build_id = CONFIG_P0['build_id']

device = torch.device("cuda")
model = ModelP0()
checkpoint = torch.load(f'results/{build_id}.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# # max: 3.1128
# # min: ~ -3.12


sample = dataset[1]
uv = sample['uv']
print(uv)
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
