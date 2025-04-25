import sys, os
import numpy as np
import casadi as cs

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

alpha = 0.05

H = cs.DM(0)
g = cs.DM([1, 1])
lbx = np.array([Vehicle.T_e_idle, 0])
ubx = np.array([Vehicle.T_e_max, Vehicle.F_b_max])
lp = {'h': cs.Sparsity(2, 2), 'a': cs.Sparsity.dense(1, 2)}
lpsolver = cs.conic("lp_solver", "clp", lp)

for i in range(6):
    v = max_v_per_gear[i]
    A = cs.DM([[(Vehicle.z_t[i] * Vehicle.z_f)/Vehicle.r_r, -1]])
    b = cs.DM([Vehicle.C_wind * v**2 + Vehicle.m * Vehicle.g * Vehicle.mu  * np.cos(alpha) + Vehicle.m * Vehicle.g * np.sin(alpha)])
    sol = lpsolver(h=0, g=g, a=A, lba=b, uba=b, lbx=lbx, ubx=ubx)
    print(f"Gear {i+1}: {v} m/s, {sol['x'][0]} T, {sol['x'][1]} F_b")
    v = min_v_per_gear[i]
    A = cs.DM([[(Vehicle.z_t[i] * Vehicle.z_f)/Vehicle.r_r, -1]])
    b = cs.DM([Vehicle.C_wind * v**2 + Vehicle.m * Vehicle.g * Vehicle.mu * np.cos(alpha) + Vehicle.m * Vehicle.g * np.sin(alpha)])
    sol = lpsolver(h=0, g=g, a=A, lba=b, uba=b, lbx=lbx, ubx=ubx)
    print(f"Gear {i+1}: {v} m/s, {sol['x'][0]} T, {sol['x'][1]} F_b")
