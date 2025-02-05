import numpy as np
import matplotlib.pyplot as plt


def plot_training(
    cost: list,
    fuel: list,
    tracking: list,
    penalty: list,
    average_interval: int = 100,
):
    # TODO add docstring
    fig, ax = plt.subplots(4, 1, sharex=True)
    # ax[0].plot(cost)
    ax[0].plot(
        np.convolve(cost, np.ones(average_interval) / average_interval, mode="valid")
    )
    ax[0].set_ylabel("Cost")
    ax[0].set_yscale("log")
    # ax[1].plot(fuel)
    ax[1].plot(
        np.convolve(fuel, np.ones(average_interval) / average_interval, mode="valid")
    )
    ax[1].set_ylabel("Fuel")
    ax[1].set_yscale("log")
    # ax[2].plot(tracking)
    ax[2].plot(
        np.convolve(
            tracking, np.ones(average_interval) / average_interval, mode="valid"
        )
    )
    ax[2].set_ylabel("Tracking")
    ax[2].set_yscale("log")
    # ax[3].plot(penalty)
    ax[3].plot(
        np.convolve(penalty, np.ones(average_interval) / average_interval, mode="valid")
    )
    ax[3].set_ylabel("Penalty")
    ax[3].set_yscale("log")
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
    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(X[:, 0])
    ax[0].plot(x_ref[:, 0])
    ax[0].legend(["actual", "desired"])
    ax[0].set_ylabel("d (m)")
    ax[1].plot(X[:, 1])
    ax[1].plot(x_ref[:, 1])
    ax[1].legend(["actual", "desired"])
    ax[1].set_ylabel("v (m/s)")
    ax[2].plot(np.cumsum(fuel))
    ax[2].set_ylabel("Fuel (L)")
    ax[3].plot(np.cumsum(R))
    ax[3].set_ylabel("Reward")

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
    plt.show()
