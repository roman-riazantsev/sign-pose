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
random_pose = torch.tensor([[6.3183e-01, 2.6757e+00, -1.0330e+00, 1.4474e-01, -1.9188e-01,
                             -2.6770e-01, -8.3857e-02, -2.8929e-03, 2.0615e-01, -7.9884e-04,
                             -2.9529e-02, 3.5706e-01, 1.6369e-01, -1.8449e-01, -3.4624e-01,
                             -2.2747e-01, -3.0279e-02, 6.9063e-01, -1.4951e-03, -1.0566e-03,
                             1.8206e-01, -3.5073e-01, -1.8281e-01, 4.5052e-01, 6.7868e-01,
                             -1.1985e-01, -9.7984e-02, -2.5002e-01, -9.3788e-02, -5.2464e-01,
                             -4.1274e-02, -2.4522e-01, 9.7890e-02, -4.2962e-02, -1.4314e-01,
                             -5.2920e-02, 8.1496e-03, 1.8553e-01, 6.0142e-02, 2.8373e-01,
                             1.5624e-01, -7.2765e-04, -2.2385e-01, -2.0514e-01, 2.3299e-01,
                             1.1829e-01, 1.1474e-01, 4.1660e-03]],
                           dtype=torch.float32)
random_shape = torch.tensor([[1, 1, 1, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                              0.0000e+00, 0.0000e+00, 0.0000e+00]], dtype=torch.float32)
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
