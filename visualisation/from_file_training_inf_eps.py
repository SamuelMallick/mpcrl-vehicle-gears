import pickle
import sys, os
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from visualisation.plot import plot_evaluation, plot_training

# file_name = "dev/results/25/data_step_4050000.pkl"
file_name = "dev/results/evaluations/l_mpc_eval_N_15_c_25.pkl"
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
if "infeasible" in data:
    infeasible = data["infeasible"]

plt.plot([len(X[i]) for i in range(len(X))])
plt.show()

for ep in range(0, 100):
    plot_evaluation(
        x_ref[ep],
        X[ep],
        U[ep],
        R[ep],
        fuel[ep],
        engine_torque[ep],
        engine_speed[ep],
        infeasible[ep] if "infeasible" in data else None,
    )

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
    average_interval=100,
)
