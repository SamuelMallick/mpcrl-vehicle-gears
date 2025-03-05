import sys, os
import numpy as np

sys.path.append(os.getcwd())
from vehicle import Vehicle

max_v_per_gear = [
    (Vehicle.w_e_max * Vehicle.r_r * 2 * np.pi) / (Vehicle.z_t[i] * Vehicle.z_f * 60)
    for i in range(6)
]
min_v_per_gear = [
    (Vehicle.w_e_idle * Vehicle.r_r * 2 * np.pi) / (Vehicle.z_t[i] * Vehicle.z_f * 60)
    for i in range(6)
]

for i in range(6):
    v = min_v_per_gear[i]
    T = (
        Vehicle.r_r
        * (Vehicle.C_wind * v**2 + Vehicle.m * Vehicle.g * Vehicle.mu)
        / (Vehicle.z_t[i] * Vehicle.z_f)
    )
    print(f"Gear {i+1}: {v} m/s, {T} Nm")
    v = max_v_per_gear[i]
    T = (
        Vehicle.r_r
        * (Vehicle.C_wind * v**2 + Vehicle.m * Vehicle.g * Vehicle.mu)
        / (Vehicle.z_t[i] * Vehicle.z_f)
    )
    print(f"Gear {i+1}: {v} m/s, {T} Nm")
