import pickle
import sys, os

sys.path.append(os.getcwd())
from visualisation.plot import plot_evaluation, plot_training

type = "heuristic_mpc"
baseline_type = "miqp_mpc"
N = 5
file_name = f"results/evaluations/easy_{type}_N_{N}.pkl"
baseline_file_name = f"results/evaluations/easy_{baseline_type}_N_{N}.pkl"
with open(file_name, "rb") as f:
    data = pickle.load(f)
with open(baseline_file_name, "rb") as f:
    baseline_data = pickle.load(f)

baseline_feul = baseline_data["fuel"]
baseline_R = baseline_data["R"]
baseline_x_ref = baseline_data["x_ref"]
baseline_engine_torque = baseline_data["T_e"]
baseline_engine_speed = baseline_data["w_e"]
baseline_solvetime = baseline_data["mpc_solve_time"]


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

print(
    f"average cost increase = {sum([(sum(R[i]) - sum(baseline_R[i]))/sum(baseline_R[i]) for i in range(len(R))]) / len(R)}"
)
print(
    f"average fuel increase = {sum([(sum(fuel[i]) - sum(baseline_feul[i]))/sum(baseline_feul[i]) for i in range(len(fuel))]) / len(fuel)}"
)
print(f"average mpc solve time increase = {sum(mpc_solve_time)}")

for ep in range(num_eps):
    plot_evaluation(
        x_ref[ep], X[ep], U[ep], R[ep], fuel[ep], engine_torque[ep], engine_speed[ep]
    )
