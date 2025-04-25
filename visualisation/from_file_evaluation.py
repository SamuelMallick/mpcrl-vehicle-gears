import pickle
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pgf import _tex_escape as mpl_common_texification
sys.path.append(os.getcwd())
# from utils.tikz import save2tikz
from visualisation.plot import plot_comparison, plot_evaluation, plot_training

N = 15
types = [
    f"miqp_mpc_N_{N}",
    f"l_mpc_eval_N_{N}_new",
    f"heuristic_mpc_low_N_{N}_new",
]
baseline_type = f"minlp_mpc_N_{N}"
# baseline_type = f"heuristic_mpc_low_N_{N}"
file_names = [f"results/evaluations/{type}.pkl" for type in types]
baseline_file_name = f"results/evaluations/{baseline_type}.pkl"

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
        if "infeasible" in data and data["infeasible"]:
            print(
                f"Infeasible count: {sum(sum(data["infeasible"][i]) for i in range(len(data["infeasible"])))}"
            )
            infeas_ep = [
                sum(data["infeasible"][i]) for i in range(len(data["infeasible"]))
            ]
            print(f"Average infeasible per episode: {sum(infeas_ep) / len(infeas_ep)}")
        if "gear_diff" in data and data["gear_diff"]:
            o = [i for l in data['gear_diff'] for i in l]
            print(f"Average gear diff: {sum(o) / len(o)}")
            
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

labels = ["MIQP-MPC", "L-MPC", "H-MPC", "MINLP-MPC"]

num_eps = len(R[0])
R_rel = [
    [
        100 * (sum(r[i]) - sum(baseline_R[i])) / sum(baseline_R[i])
        for i in range(num_eps)
    ]
    for r in R
]
o = [np.array_split(np.array(t), num_eps) for t in T]
o.append(
    baseline_t
)  # handled differently because baseline data is already split into eps
T.append(np.concatenate(baseline_t))
t_ep = [[sum(t) / len(t) for t in ep] for ep in o]

fig, ax = plt.subplots(2, 1, sharex=False)
ax[0].boxplot(R_rel, labels=labels[:-1])
for i in range(len(types)):
    ax[0].hlines(
        np.median(R_rel[i]),
        xmin=i + 0.7,
        xmax=i + 1.3,
        color="gray",
        linestyle="dashed",
        linewidth=0.5,
    )
    ax[0].text(
        i + 1.3,
        np.median(R_rel[i]),
        f"{np.median(R_rel[i]):.2f}",
        va="center",
        ha="left",
        color="black",
    )
ax[0].set_ylabel("J")
ax[0].xaxis.set_ticks_position("top")  # Move ticks to the top
ax[0].xaxis.set_label_position("top")  # Move labels to the top
# ax[0].set_yscale("log")
box = ax[1].boxplot(
    t_ep,
    labels=labels,
)
for i in range(len(t_ep)):
    ax[1].hlines(
        np.median(t_ep[i]),
        xmin=i + 0.7,
        xmax=i + 1.3,
        color="gray",
        linestyle="dashed",
        linewidth=0.5,
    )
    ax[1].text(
        i + 1.3,
        np.median(t_ep[i]),
        f"{np.median(t_ep[i]):.2f}",
        va="center",
        ha="left",
        color="black",
    )
    ax[1].plot(i + 1, max(T[i]), marker="^", color="red")
    ax[1].text(
        i + 1.1, max(T[i]), f"{ max(T[i]):.2f}", va="center", ha="left", color="red"
    )
ax[1].set_yscale("log")
ax[1].set_ylabel("Time (s)")
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
