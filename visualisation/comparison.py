import pickle
import sys, os

sys.path.append(os.getcwd())
from visualisation.plot import plot_evaluation, plot_comparison

type_1 = "rl_mpc_eval"
type_2 = "heuristic_mpc"
N = 5
file_name_1 = f"results/evaluations/easy_expert_{type_1}_N_{N}.pkl"
file_name_2 = f"results/evaluations/easy_{type_2}_N_{N}.pkl"
with open(file_name_1, "rb") as f:
    data_1 = pickle.load(f)
with open(file_name_2, "rb") as f:
    data_2 = pickle.load(f)

# cost = data["cost"]
# fuel = data["fuel"]
# R = data["R"]
# X = data["X"]
# U = data["U"]
# x_ref = data["x_ref"]
# engine_torque = data["T_e"]
# engine_speed = data["w_e"]
# mpc_solve_time = data["mpc_solve_time"]
# num_eps = len(R)

ep = 20
plot_comparison(
    [data_1["x_ref"][ep], data_2["x_ref"][ep]],
    [data_1["X"][ep], data_2["X"][ep]],
    [data_1["U"][ep], data_2["U"][ep]],
    [data_1["R"][ep], data_2["R"][ep]],
    [data_1["fuel"][ep], data_2["fuel"][ep]],
    [data_1["T_e"][ep], data_2["T_e"][ep]],
    [data_1["w_e"][ep], data_2["w_e"][ep]],
)
