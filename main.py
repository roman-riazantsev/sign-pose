import numpy as np
import torch

from config import CONFIG
from fh_utils import *
import matplotlib.pyplot as plt


def load_samples(dataset_path, build_mode, subset_size):
    db_data_anno = load_db_annotation(dataset_path, build_mode)
    for idx in range(subset_size):
        idx = 2
        img = read_img(idx, dataset_path, build_mode)
        msk = read_msk(idx, dataset_path)

        # annotation for this frame
        K, mano, xyz = db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)

        poses, shapes, uv_root, scale = split_theta(mano)
        poses = torch.tensor(poses, dtype=torch.float32)
        shapes = torch.tensor(shapes, dtype=torch.float32)

        print(poses, shapes)

        msk_rendered = None
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(img)
        ax2.imshow(msk if msk_rendered is None else msk_rendered)
        plot_hand(ax1, uv, order='uv')
        plot_hand(ax2, uv, order='uv')
        plt.show()


if __name__ == '__main__':
    dataset_path = CONFIG['dataset_path']
    build_mode = CONFIG['build_mode']
    subset_size = CONFIG['subset_size']
    load_samples(dataset_path, build_mode, subset_size)
