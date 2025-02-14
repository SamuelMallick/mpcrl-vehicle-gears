import numpy as np
import matplotlib.pyplot as plt


def plot_training(
    cost: list,
    fuel: list,
    tracking: list,
    penalty: list,
    reward: list,
    average_interval: int = 100,
    log_scales: bool = False,
    only_averages: bool = False,
):
    # TODO add docstring
    fig, ax = plt.subplots(5, 1, sharex=True)
    if not only_averages:
        ax[0].plot(cost)
        ax[1].plot(fuel)
        ax[2].plot(tracking)
        ax[3].plot(penalty)
        ax[4].plot(reward)
    if log_scales:
        for i in range(5):
            ax[i].set_yscale("log")
    ax[0].plot(
        np.convolve(cost, np.ones(average_interval) / average_interval, mode="valid")
    )
    ax[0].set_ylabel("Cost")
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
    ax[4].set_ylabel("Reward")
    plt.show()


def plot_evaluation(
    x_ref: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    R: np.ndarray,
    fuel: np.ndarray,
    T_e: np.ndarray,
    w_e: np.ndarray,
):
    # TODO add docstring
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
    fig, ax = plt.subplots(5, 1, sharex=True)
    ax[0].plot(x_ref[0][:, 0] - X[0][:-1, 0])
    ax[0].plot(x_ref[1][:, 0] - X[1][:-1, 0])
    ax[0].set_ylabel("d_e (m)")
    ax[1].plot(X[0][:, 0])
    ax[1].plot(X[1][:, 0])
    ax[1].plot(x_ref[0][:, 0])
    ax[1].legend(["actual 1", "actual 2", "desired"])
    ax[2].plot(X[0][:, 1])
    ax[2].plot(X[1][:, 1])
    ax[2].plot(x_ref[0][:, 1])
    ax[2].legend(["actual 1", "actual 2", "desired"])
    ax[2].set_ylabel("v (m/s)")
    ax[3].plot(np.cumsum(fuel[0]))
    ax[3].plot(np.cumsum(fuel[1]))
    ax[3].set_ylabel("Fuel (L)")
    ax[4].plot(np.cumsum(R[0]))
    ax[4].plot(np.cumsum(R[1]))
    # ax[4].plot(R)
    ax[4].set_ylabel("Reward")

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(T_e[0])
    ax[0].plot(T_e[1])
    ax[0].set_ylabel("T_e (Nm)")
    ax[1].plot(w_e[0])
    ax[1].plot(w_e[1])
    ax[1].set_ylabel("w_e (rpm)")
    ax[2].plot(U[0][:, 1])
    ax[2].plot(U[1][:, 1])
    ax[2].set_ylabel("F_b (N)")
    ax[3].plot(U[0][:, 2])
    ax[3].plot(U[1][:, 2])
    ax[3].set_ylabel("gear")
    ax[3].set_xticks([i for i in range(len(U))])
    ax[3].set_yticks([i for i in range(6)])
    ax[3].grid(visible=True, which="major", color="gray", linestyle="-", linewidth=0.8)
    plt.show()
