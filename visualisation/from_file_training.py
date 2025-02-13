import pickle
import sys, os

sys.path.append(os.getcwd())
from visualisation.plot import plot_evaluation, plot_training

file_name = "results/N_5_prev_slope_and_IC/data_ep_49999.pkl"
# file_name = "results/N_5/data_ep_49999.pkl"
with open(file_name, "rb") as f:
    data = pickle.load(f)

cost = data["cost"]
fuel = data["fuel"]
R = data["R"]
X = data["X"]
U = data["U"]
x_ref = data["x_ref"]
engine_torque = data["T_e"]
engine_speed = data["w_e"]

# ep = 49671
# plot_evaluation(
#     x_ref[ep], X[ep], U[ep], R[ep], fuel[ep], engine_torque[ep], engine_speed[ep]
# )

plot_training(
    [sum(cost[i]) for i in range(len(cost))],
    [sum(fuel[i]) for i in range(len(fuel))],
    [sum(R[i]) - sum(fuel[i]) for i in range(len(R))],
    [sum(cost[i]) - sum(R[i]) for i in range(len(R))],
    [sum(R[i]) for i in range(len(R))],
)
