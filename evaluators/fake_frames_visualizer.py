import cv2

from configs.config_u0 import CONFIG_U0
from analysis.frames_generator import FramesGenerator
from utils.utils import to_numpy, plot_hand
import matplotlib.pyplot as plt

from configs.config_m0 import CONFIG_M0
from datasets.frei_dataset_u0 import FreiHandDataset
from transforms.to_tensor_u0 import ToTensor

data_path = CONFIG_U0['dataset_path']
build_mode = CONFIG_U0['build_mode']
frames_generator = FramesGenerator(data_path, build_mode)

oredered_sequence = frames_generator.order_frames('xyz', 2345, 340)

build_id = CONFIG_M0['build_id']

dataset = FreiHandDataset(
    CONFIG_M0['dataset_path'],
    CONFIG_M0['data_version'],
    transform=ToTensor()
)

xyz_array = frames_generator.get_xyz_array()
K_array = frames_generator.get_K_array()
uv_array = frames_generator.get_uv_array(xyz_array, K_array)

for idx in oredered_sequence:
    # uv_21x2 = np.array(uv_list).reshape(32560, 21, 2)
    img_tensor = dataset[idx]['img']
    img_np = to_numpy(img_tensor).transpose((1, 2, 0))
    im_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    cv2.imshow('df', im_rgb)
    cv2.waitKey(-1)
