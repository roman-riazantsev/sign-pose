import torch
import numpy as np


def init_build_id(config):
    model_id = config['model_id']
    experiment_number = config['experiment_number']
    build_id = "_".join([model_id, str(experiment_number)])
    config['build_id'] = build_id
    return config


def to_numpy(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cpu()

    return tensor.detach().numpy()

#
# def get_focal_pp(K):
#     """ Extract the camera parameters that are relevant for an orthographic assumption. """
#     focal = 0.5 * (K[0, 0] + K[1, 1])
#     pp = K[:2, 2]
#     return focal, pp
#
#
# def backproject_ortho(uv, scale,  # kind of the predictions
#                       focal, pp):  # kind of the camera calibration
#     """ Calculate 3D coordinates from 2D coordinates and the camera parameters. """
#     uv = uv.copy()
#     uv -= pp
#     xyz = np.concatenate([np.reshape(uv, [-1, 2]),
#                           np.ones_like(uv[:, :1]) * focal], 1)
#     xyz /= scale
#     return xyz
#
#
# def recover_root(uv_root, scale,
#                  focal, pp):
#     uv_root = np.reshape(uv_root, [1, 2])
#     xyz_root = backproject_ortho(uv_root, scale, focal, pp)
#     return xyz_root
#
#
# def plot_hand(axis, coords_hw, vis=None, color_fixed=None, linewidth='1', order='hw', draw_kp=True):
#     """ Plots a hand stick figure into a matplotlib figure. """
#     if order == 'uv':
#         coords_hw = coords_hw[:, ::-1]
#
#     colors = np.array([[0.4, 0.4, 0.4],
#                        [0.4, 0.0, 0.0],
#                        [0.6, 0.0, 0.0],
#                        [0.8, 0.0, 0.0],
#                        [1.0, 0.0, 0.0],
#                        [0.4, 0.4, 0.0],
#                        [0.6, 0.6, 0.0],
#                        [0.8, 0.8, 0.0],
#                        [1.0, 1.0, 0.0],
#                        [0.0, 0.4, 0.2],
#                        [0.0, 0.6, 0.3],
#                        [0.0, 0.8, 0.4],
#                        [0.0, 1.0, 0.5],
#                        [0.0, 0.2, 0.4],
#                        [0.0, 0.3, 0.6],
#                        [0.0, 0.4, 0.8],
#                        [0.0, 0.5, 1.0],
#                        [0.4, 0.0, 0.4],
#                        [0.6, 0.0, 0.6],
#                        [0.7, 0.0, 0.8],
#                        [1.0, 0.0, 1.0]])
#
#     colors = colors[:, ::-1]
#
#     # define connections and colors of the bones
#     bones = [((0, 1), colors[1, :]),
#              ((1, 2), colors[2, :]),
#              ((2, 3), colors[3, :]),
#              ((3, 4), colors[4, :]),
#
#              ((0, 5), colors[5, :]),
#              ((5, 6), colors[6, :]),
#              ((6, 7), colors[7, :]),
#              ((7, 8), colors[8, :]),
#
#              ((0, 9), colors[9, :]),
#              ((9, 10), colors[10, :]),
#              ((10, 11), colors[11, :]),
#              ((11, 12), colors[12, :]),
#
#              ((0, 13), colors[13, :]),
#              ((13, 14), colors[14, :]),
#              ((14, 15), colors[15, :]),
#              ((15, 16), colors[16, :]),
#
#              ((0, 17), colors[17, :]),
#              ((17, 18), colors[18, :]),
#              ((18, 19), colors[19, :]),
#              ((19, 20), colors[20, :])]
#
#     if vis is None:
#         vis = np.ones_like(coords_hw[:, 0]) == 1.0
#
#     for connection, color in bones:
#         if (vis[connection[0]] == False) or (vis[connection[1]] == False):
#             continue
#
#         coord1 = coords_hw[connection[0], :]
#         coord2 = coords_hw[connection[1], :]
#         coords = np.stack([coord1, coord2])
#         if color_fixed is None:
#             axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
#         else:
#             axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)
#
#     if not draw_kp:
#         return
#
#     for i in range(21):
#         if vis[i] > 0.5:
#             axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=colors[i, :])
