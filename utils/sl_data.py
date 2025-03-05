import pickle
import torch
import sys, os

sys.path.append(os.getcwd())
from vehicle import Vehicle

with open("results/sl_data/2_nn_inputs_300.pkl", "rb") as f:
    data = pickle.load(f)
    inputs = data[:300]
with open("results/sl_data/2_nn_targets_explicit_300.pkl", "rb") as f:
    data = pickle.load(f)
    targets_explicit = data[:300]
with open("results/sl_data/2_nn_targets_shift_300.pkl", "rb") as f:
    data = pickle.load(f)
    targets_shift = data[:300]

v_target_norm = (
    (Vehicle.v_max - Vehicle.v_min) * inputs[:, :, :, 2]
    + Vehicle.v_min
    - inputs[:, :, :, 1]
)
v_target_norm = (v_target_norm - Vehicle.v_min) / (Vehicle.v_max - Vehicle.v_min)
inputs = torch.cat(
    (inputs[:, :, :, :3], v_target_norm.unsqueeze(-1), inputs[:, :, :, 3:]), dim=-1
)

with open("results/sl_data/2_nn_inputs_augmented_300.pkl", "wb") as f:
    pickle.dump(inputs, f)
with open("results/sl_data/2_nn_targets_explicit_300.pkl", "wb") as f:
    pickle.dump(targets_explicit, f)
