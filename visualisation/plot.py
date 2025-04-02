import numpy as np
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
# from utils.tikz import save2tikz


def plot_training(
    cost: list,
    fuel: list,
    tracking: list,
    penalty: list,
    reward: list,
    infeasible: list | None = None,
    average_interval: int = 100,
    log_scales: bool = False,
    only_averages: bool = False,
):
    fig, ax = plt.subplots(5 if infeasible is None else 6, 1, sharex=True)
    if not only_averages:
        ax[0].plot(cost)
        ax[1].plot(fuel)
        ax[2].plot(tracking)
        ax[3].plot(penalty)
        ax[4].plot(reward)
        if infeasible is not None:
            ax[5].plot(infeasible)
    if log_scales:
        for i in range(5):
            ax[i].set_yscale("log")
    ax[0].plot(
        np.convolve(cost, np.ones(average_interval) / average_interval, mode="valid"),
    )
    ax[0].set_ylabel("RL cost")
    ax[1].plot(
        np.convolve(fuel, np.ones(average_interval) / average_interval, mode="valid")
    )
    ax[1].set_ylabel("Fuel")
    ax[2].plot(
        np.convolve(
            tracking, np.ones(average_interval) / average_interval, mode="valid"
        )
    )
    ax[2].set_ylabel("Tracking")
    ax[3].plot(
        np.convolve(penalty, np.ones(average_interval) / average_interval, mode="valid")
    )
    ax[3].set_ylabel("Penalty")
    ax[4].plot(
        np.convolve(reward, np.ones(average_interval) / average_interval, mode="valid")
    )
    ax[4].set_ylabel("L")
    # ax[4].set_ylim([700, 1000])
    if infeasible is not None:
        ax[5].plot(
            np.convolve(
                infeasible, np.ones(average_interval) / average_interval, mode="valid"
            )
        )
        ax[5].set_ylabel("Infeasible")
    plt.show()


def plot_evaluation(
    x_ref: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    R: np.ndarray,
    fuel: np.ndarray,
    T_e: np.ndarray,
    w_e: np.ndarray,
    infeasible: np.ndarray = None,
):
    fig, ax = plt.subplots(5, 1, sharex=True)
    ax[0].plot(x_ref[:, 0] - X[:-1, 0])
    ax[0].set_ylabel("d_e (m)")
    ax[1].plot(X[:, 0])
    ax[1].plot(x_ref[:, 0])
    ax[1].legend(["actual", "desired"])
    ax[2].plot(X[:, 1])
    ax[2].plot(x_ref[:, 1])
    ax[2].legend(["actual", "desired"])
    ax[2].set_ylabel("v (m/s)")
    ax[3].plot(np.cumsum(fuel))
    ax[3].set_ylabel("Fuel (L)")
    ax[4].plot(np.cumsum(R))
    # ax[4].plot(R)
    ax[4].set_ylabel("Reward")

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(T_e)  # T_e actual
    ax[0].plot(U[:, 0])  # T_e desired
    ax[0].legend(["actual", "desired"])
    ax[0].set_ylabel("T_e (Nm)")
    ax[1].plot(w_e)
    ax[1].set_ylabel("w_e (rpm)")
    ax[2].plot(U[:, 1])
    ax[2].set_ylabel("F_b (N)")
    ax[3].plot(U[:, 2])
    ax[3].set_ylabel("gear")
    if infeasible is not None:
        ax[3].fill_between(
            np.arange(U.shape[0]),
            0,
            5,
            where=infeasible,
            color="red",
            alpha=1,
            label="Shaded Region",
        )
    ax[3].set_xticks([i for i in range(len(U))])
    ax[3].set_yticks([i for i in range(6)])
    ax[3].grid(visible=True, which="major", color="gray", linestyle="-", linewidth=0.8)
    plt.show()


def plot_comparison(
    x_ref: list[np.ndarray],
    X: list[np.ndarray],
    U: list[np.ndarray],
    R: list[np.ndarray],
    fuel: list[np.ndarray],
    T_e: list[np.ndarray],
    w_e: list[np.ndarray],
):
    linestyles = ["--", "-", "-.", ":", "-"] * 3
    colors = ["red", "blue", "green", "orange", "black"] * 3
    labels = ["H-MPC", "L2-MPC", "L-MPC", "MIQP-MPC", "MINLP-MPC"]
    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[1].plot(x_ref[0][:, 1], color="black")
    ax[1].legend(["ref"])
    ax[1].set_ylabel("v (m/s)")
    for i in range(len(x_ref)):
        ax[0].plot(
            x_ref[i][:, 0] - X[i][:-1, 0], linestyle=linestyles[i], color=colors[i]
        )
        # ax[1].plot(X[i][:, 0])
        ax[1].plot(
            X[i][:, 1], linestyle=linestyles[i], label="_nolegend_", color=colors[i]
        )
        ax[2].plot(np.cumsum(fuel[i]), linestyle=linestyles[i], color=colors[i])
        ax[3].plot(np.cumsum(R[i]), linestyle=linestyles[i], color=colors[i])

    ax[0].set_ylabel("d_e (m)")
    ax[0].legend(labels)

    ax[2].set_ylabel("Fuel (L)")

    ax[3].set_ylabel("Reward")
    # save2tikz(plt.gcf())

    fig, ax = plt.subplots(4, 1, sharex=True)
    for i in range(len(x_ref)):
        ax[0].plot(T_e[i], linestyle=linestyles[i], color=colors[i])
        ax[1].plot(w_e[i], linestyle=linestyles[i], color=colors[i])
        ax[2].plot(U[i][:, 1], linestyle=linestyles[i], color=colors[i])
        ax[3].plot(U[i][:, 2], linestyle=linestyles[i], color=colors[i])
    ax[0].set_ylabel("T_e (Nm)")
    ax[1].set_ylabel("w_e (rpm)")
    ax[2].set_ylabel("F_b (N)")
    ax[3].set_ylabel("gear")
    # save2tikz(plt.gcf())

    plt.show()


def plot_reference_traj(x_ref: list[np.ndarray], change_points=None) -> None:
    """
    Plot reference trajectory (position and speed).
    :param: x_ref: n x 2 ndarray with reference position trajectory (index 0) and
    reference speed trajectory (index 1)
    :param: change_points: m x 1 ndarray of acceleration change points (as time steps)
    """
    _, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x_ref[:, 0])
    ax[0].set_ylabel("d (m)")
    ax[1].plot(x_ref[:, 1])
    ax[1].set_ylabel("v (m/s)")
    if change_points is not None:
        for cp in change_points:
            ax[0].scatter(cp, x_ref[cp, 0], color="r")
            ax[1].scatter(cp, x_ref[cp, 1], color="r")
    plt.show()
