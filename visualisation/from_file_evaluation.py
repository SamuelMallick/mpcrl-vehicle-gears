import pickle
import sys, os

sys.path.append(os.getcwd())
from visualisation.plot import plot_evaluation, plot_training

file_name = "results/evaluations/MIQP_MPC_N_20.pkl"
with open(file_name, "rb") as f:
    data = pickle.load(f)

# cost = data["cost"]
fuel = data["fuel"]
R = data["R"]
X = data["X"]
U = data["U"]
x_ref = data["x_ref"]
engine_torque = data["T_e"]
engine_speed = data["w_e"]
mpc_solve_time = data["mpc_solve_time"]
num_eps = len(R)

print(f"average cost = {sum([sum(R[i]) for i in range(len(R))]) / len(R)}")
print(f"average fuel = {sum([sum(fuel[i]) for i in range(len(fuel))]) / len(fuel)}")
print(f"total mpc solve times = {sum(mpc_solve_time)}")

# for ep in range(num_eps):
#     plot_evaluation(
#         x_ref[ep], X[ep], U[ep], R[ep], fuel[ep], engine_torque[ep], engine_speed[ep]
#     )
