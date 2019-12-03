import torch

from configs.config_affine0 import CONFIG_AFFINE0
from models.model_affine0 import ModelAffine0


def get_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A = torch.tensor([[96.3155, 6.4408, 6.2320, 1],
                      [80.5500, -18.2667, 31.7445, 1],
                      [59.6476, -21.7967, 54.3762, 1],
                      [49.9231, -11.3441, 77.6003, 1],
                      [23.8699, 2.8700, 95.4979, 1],
                      [59.0175, 3.9633, 89.4892, 1],
                      [43.0577, 8.9765, 118.1685, 1],
                      [29.8227, 11.5733, 135.9823, 1],
                      [23.2641, 13.6527, 158.6822, 1],
                      [51.6065, 28.1710, 87.6219, 1],
                      [34.8829, 37.7173, 113.0928, 1],
                      [20.6123, 46.2407, 129.6086, 1],
                      [10.1921, 58.5846, 150.8793, 1],
                      [47.9414, 45.0360, 66.8808, 1],
                      [32.8974, 50.6703, 91.1090, 1],
                      [17.8783, 60.6983, 108.3596, 1],
                      [11.4558, 70.0814, 129.8244, 1],
                      [43.4404, 54.3447, 47.4658, 1],
                      [31.4775, 63.6320, 62.4144, 1],
                      [19.0282, 73.5074, 73.0234, 1],
                      [15.5087, 86.7201, 89.3633, 1]]).to(device)
    b = torch.tensor([[8.2529e-02, -1.4945e-01, 5.6315e-01, 1],
                      [6.6335e-02, -1.7623e-01, 5.8878e-01, 1],
                      [4.4146e-02, -1.8020e-01, 6.1253e-01, 1],
                      [3.3096e-02, -1.6868e-01, 6.3744e-01, 1],
                      [2.7584e-03, -1.5482e-01, 6.5776e-01, 1],
                      [4.3642e-02, -1.5129e-01, 6.5138e-01, 1],
                      [2.6643e-02, -1.4592e-01, 6.8187e-01, 1],
                      [1.2250e-02, -1.4310e-01, 7.0081e-01, 1],
                      [3.6873e-03, -1.3518e-01, 7.2599e-01, 1],
                      [3.5133e-02, -1.2599e-01, 6.4969e-01, 1],
                      [1.7186e-02, -1.1599e-01, 6.7677e-01, 1],
                      [1.9427e-03, -1.0687e-01, 6.9442e-01, 1],
                      [-5.4359e-03, -9.3021e-02, 7.1830e-01, 1],
                      [3.0754e-02, -1.0840e-01, 6.2777e-01, 1],
                      [1.4456e-02, -1.0251e-01, 6.5401e-01, 1],
                      [-1.7585e-03, -9.1628e-02, 6.7300e-01, 1],
                      [-1.3311e-02, -8.3420e-02, 6.9725e-01, 1],
                      [2.5966e-02, -9.8290e-02, 6.0717e-01, 1],
                      [1.3304e-02, -8.8364e-02, 6.2325e-01, 1],
                      [-7.6417e-07, -7.7943e-02, 6.3471e-01, 1],
                      [-7.6329e-03, -6.4999e-02, 6.5365e-01, 1]]).to(device)
    K = torch.tensor([[388.9018, 0.0000, 76.3069],
                      [0.0000, 388.7123, 195.5198],
                      [0.0000, 0.0000, 1.0000]]).to(device)

    return A, b, K


if __name__ == '__main__':
    A, b, K = get_data()

    model = ModelAffine0(CONFIG_AFFINE0)
    model.fit(A, b, K)
