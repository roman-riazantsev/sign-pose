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
random_pose = torch.tensor([[0.44984702, -0.22015289, 0.54938289, -0.00494608, -0.01699188,
                             0.30066987, 0.13834799, 0.03047967, -0.08054493, -0.11717121,
                             0.03270623, -0.37028669, -0.2045858, 0.15049707, 0.59146934,
                             -0.04468567, -0.06943699, -0.46513121, 0.01072934, 0.03271886,
                             -0.27121537, -0.51538225, 0.51169612, 0.78337064, -0.5108254,
                             0.10383249, 0.37561964, -0.08759982, 0.0623596, 0.1352752,
                             -0.27866673, 0.2572354, 0.798621, -0.25857588, 0.05991794,
                             0.59108595, -0.07769459, 0.10952272, 0.19827406, 0.34415588,
                             -0.00251028, -0.06033212, -0.10143131, -0.06303516, 0.22536544,
                             0.3939942, -0.09946802, 0.16713043]],
                           dtype=torch.float32)
random_shape = torch.tensor(
    [[0.19801947, -0.02240238,
      -0.0073368, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
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
