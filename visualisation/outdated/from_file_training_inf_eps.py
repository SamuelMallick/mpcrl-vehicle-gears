import pickle
import sys, os
from matplotlib import pyplot as plt


sys.path.append(os.getcwd())
# from utils.tikz import save2tikz
from visualisation.plot import plot_evaluation, plot_training

# file_names = [
#     "dev/results/1_seeds/1/data_step_5000000.pkl",
# ]
# file_names = [
#     "dev/results/2/data_step_50000.pkl"
# ]  # , "dev/results/31/data_step_50000.pkl"]
file_names = [
    # "dev/results/1_seeds/1/data_step_5000000.pkl",
    # "dev/results/1_seeds/2/data_step_5000000.pkl",
    # "dev/results/1_seeds/3/data_step_5000000.pkl",
    # "dev/results/1_seeds/4/data_step_3800000.pkl",
    # "dev/results/1_seeds/5/data_step_3875000.pkl",
    "dev/results/4/data_step_3825000.pkl",
]
fig, ax = plt.subplots(4, 1, sharex=True)

for file_name in file_names:
    # file_name = "dev/results/evaluations/l_mpc_eval_N_15_c_25.pkl"
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
    if "infeasible" in data:
        infeasible = data["infeasible"]
    if "heuristic" in data:
        heuristic = data["heuristic"]

    # plt.plot([len(X[i]) for i in range(len(X))])
    # plt.show()
    # len = 500
    # ep = 0
    # plot_evaluation(
    #     x_ref[ep][:len, :, :],
    #     X[ep][:len+1, :, :],
    #     U[ep][:len, :],
    #     R[ep][:len],
    #     fuel[ep][:len],
    #     engine_torque[ep][:len],
    #     engine_speed[ep][:len],
    #     infeasible[ep][:len] if "infeasible" in data else None,
    # )
    # ep = -1
    # plot_evaluation(
    #     x_ref[ep][-len:, :, :],
    #     X[ep][-len-1:, :, :],
    #     U[ep][-len:, :],
    #     R[ep][-len:],
    #     fuel[ep][-len:],
    #     engine_torque[ep][-len:],
    #     engine_speed[ep][-len:],
    #     infeasible[ep][-len:] if "infeasible" in data else None,
    # )

    plot_training(
        [item for sublist in cost for item in sublist],
        [item for sublist in fuel for item in sublist],
        [r - f for sub_r, sub_f in zip(R, fuel) for r, f in zip(sub_r, sub_f)],
        [c - r for sub_c, sub_r in zip(cost, R) for c, r in zip(sub_c, sub_r)],
        [r for sub_r in R for r in sub_r],
        infeasible=(
            [item for sublist in infeasible for item in sublist]
            if "infeasible" in data
            else None
        ),
        only_averages=True,
        log_scales=False,
        average_interval=10000,
        ax=ax,
    )
    # ax[0].legend(["1", "2", "3", "4", "5", "no_bi"])
# save2tikz(plt.gcf())
plt.show()
