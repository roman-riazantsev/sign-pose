import torch
from manopth.manolayer import ManoLayer
from manopth import demo

batch_size = 1
# Select number of principal components for pose space
ncomps = 45

# Initialize MANO layer
mano_layer = ManoLayer(
    mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=False)

# Generate random shape parameters
random_shape = torch.rand(batch_size, 10)
# Generate random pose parameters, including 3 values for global axis-angle rotation
random_pose = torch.rand(batch_size, ncomps + 3)
# print(random_pose.type())
random_pose = torch.tensor([[0.5576, 0.0780, 1.9508, 0.1080, 0.2545, -0.2010, 0.2553, 0,
                             -0.3103, 0.2027, 0.5, -0.1910, 0.0104, 0.2927, 0.7490, -0.1511,
                             0.5059, 0.6051, -0.3941, 0.5, 1, -0.3091, 0.4936, 0.7028,
                             1, 0.02, 0.0147, -0.6654, -0.0777, -0.7503, -0.1413, -0.0093,
                             0.5808, -0.1387, 0.2958, 0.4595, -0.3251, 0.2328, 0.9202, -0.5215,
                             0.3213, -0.0823, 0.0062, -0.2645, 0.5065, 0.2708, -0.7465, -0.4358]],
                           dtype=torch.float32)
random_shape = torch.tensor([[-0.0821, 0.0223, -0.1239, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                              0.0000, 0.0000]], dtype=torch.float32)
# print(random_pose)

# Forward pass through MANO layer
hand_verts, hand_joints = mano_layer(random_pose, random_shape)
demo.display_hand({
    'verts': hand_verts,
    'joints': hand_joints
},
    mano_faces=mano_layer.th_faces)

print(hand_verts)
print(hand_joints)
