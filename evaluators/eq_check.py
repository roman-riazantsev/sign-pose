import numpy as np

A = np.array([[96.3155, 6.4408, 6.2320, 1],
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
              [15.5087, 86.7201, 89.3633, 1]])


#
# x = np.array([[7.1897e-04, -2.5783e-03, 1.8849e-04, 3.7177e-01],
#               [-1.5689e-04, 4.5643e-05, 9.8314e-05, 1.8859e-01],
#               [-1.6242e-04, -1.1618e-03, 1.1492e-03, 9.4405e-01],
#               [1.1049e-02, 7.1954e-02, 5.3836e-01, 9.8845e-01]])

print(A.dot(x))
