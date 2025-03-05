import pickle
import sys, os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
# from utils.tikz import save2tikz
from visualisation.plot import plot_comparison, plot_evaluation, plot_training

types = [
    "miqp_mpc_N_15",
    "sl_mpc_eval_N_15_ep_300_c_5",
    "heuristic_mpc_low_N_15",
]
baseline_type = "minlp_mpc_N_15"
file_names = [f"results/evaluations/seed_10/windy/{type}.pkl" for type in types]
baseline_file_name = f"results/evaluations/seed_10/windy/{baseline_type}.pkl"

X = []
U = []
R = []
T = []
fuel = []
x_ref = []
engine_torque = []
engine_speed = []
for file_name in file_names:
    with open(file_name, "rb") as f:
        data = pickle.load(f)
        R.append(data["R"])
        T.append(data["mpc_solve_time"])
        fuel.append(data["fuel"])
        X.append(data["X"])
        U.append(data["U"])
        x_ref.append(data["x_ref"])
        engine_torque.append(data["T_e"])
        engine_speed.append(data["w_e"])
        infeasible = data["infeasible"] if "infeasible" in data else None
with open(baseline_file_name, "rb") as f:
    baseline_data = pickle.load(f)
    baseline_R = baseline_data["R"]
    baseline_t = baseline_data["mpc_solve_time"]
    baseline_X = baseline_data["X"]
    baseline_U = baseline_data["U"]
    baseline_fuel = baseline_data["fuel"]
    baseline_x_ref = baseline_data["x_ref"]
    baseline_engine_torque = baseline_data["T_e"]
    baseline_engine_speed = baseline_data["w_e"]

num_eps = len(R[0])
R_rel = [
    [
        100 * (sum(r[i]) - sum(baseline_R[i])) / sum(baseline_R[i])
        for i in range(num_eps)
    ]
    for r in R
]
o = [np.array_split(np.array(t), num_eps) for t in T]
t_ep = [[sum(t) / len(t) for t in ep] for ep in o]

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].boxplot(R_rel, labels=types)
box = ax[1].boxplot(
    t_ep,
    labels=types,
)
for i in range(len(types)):
    ax[1].plot(i + 1, max(T[i]), marker="^", color="red")
ax[1].set_yscale("log")
# save2tikz(plt.gcf())
plt.show()

for ep in range(3, len(R[0])):
    print(f"Episode {ep}")
    plot_comparison(
        [x_ref[i][ep] for i in range(len(types))] + [baseline_x_ref[ep]],
        [X[i][ep] for i in range(len(types))] + [baseline_X[ep]],
        [U[i][ep] for i in range(len(types))] + [baseline_U[ep]],
        [R[i][ep] for i in range(len(types))] + [baseline_R[ep]],
        [fuel[i][ep] for i in range(len(types))] + [baseline_fuel[ep]],
        [engine_torque[i][ep] for i in range(len(types))]
        + [baseline_engine_torque[ep]],
        [engine_speed[i][ep] for i in range(len(types))] + [baseline_engine_speed[ep]],
    )
