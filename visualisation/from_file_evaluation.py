import pickle
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from visualisation.plot import plot_comparison, plot_evaluation, plot_training

# type = "heuristic_mpc_low"
# type = "miqp_mpc"
types = [
    "sl_exp_mpc_eval_clipped",
    "heuristic_mpc_low",
    "heuristic_mpc_mid",
    "heuristic_mpc_high",
]
baseline_type = "miqp_mpc"
N = 15
config = "c_1"
file_names = [
    f"results/evaluations/seed_10/{type}_N_{N}_{config}.pkl" for type in types
]
baseline_file_name = f"results/evaluations/seed_10/{baseline_type}_N_{N}_{config}.pkl"

R = []
T = []
for file_name in file_names:
    with open(file_name, "rb") as f:
        data = pickle.load(f)
        R.append(data["R"])
        T.append(data["mpc_solve_time"])
with open(baseline_file_name, "rb") as f:
    baseline_data = pickle.load(f)
    baseline_R = baseline_data["R"]
    baseline_t = baseline_data["mpc_solve_time"]

num_eps = len(R[0])
R_rel = [
    [(sum(r[i]) - sum(baseline_R[i])) / sum(baseline_R[i]) for i in range(num_eps)]
    for r in R
]
t_rel = [
    (sum(t) / len(t) - sum(baseline_t) / len(baseline_t))
    / (sum(baseline_t) / len(baseline_t))
    for t in T
]

fig, ax = plt.subplots(1, 1, sharex=True)
ax.boxplot(R_rel, labels=types)
plt.show()
exit()


baseline_X = baseline_data["X"]
baseline_U = baseline_data["U"]
baseline_fuel = baseline_data["fuel"]
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
infeasible = data["infeasible"] if "infeasible" in data else None


if "valid_episodes" in baseline_data and baseline_data["valid_episodes"] is not None:
    valid_episodes = baseline_data["valid_episodes"]
    R = [R[i] for i in valid_episodes]
    fuel = [fuel[i] for i in valid_episodes]

num_eps = len(R)

print(
    f"average cost increase = {sum([(sum(R[i]) - sum(baseline_R[i]))/sum(baseline_R[i]) for i in range(num_eps)]) / num_eps}"
)
print(
    f"average fuel increase = {sum([(sum(fuel[i]) - sum(baseline_fuel[i]))/sum(baseline_fuel[i]) for i in range(num_eps)]) / num_eps}"
)
print(f"average mpc solve time increase = {sum(mpc_solve_time)}")

for ep in range(num_eps):
    plot_comparison(
        [x_ref[ep], baseline_x_ref[ep]],
        [X[ep], baseline_X[ep]],
        [U[ep], baseline_U[ep]],
        [R[ep], baseline_R[ep]],
        [fuel[ep], baseline_fuel[ep]],
        [engine_torque[ep], baseline_engine_torque[ep]],
        [engine_speed[ep], baseline_engine_speed[ep]],
    )

for ep in range(num_eps):
    plot_evaluation(
        x_ref[ep],
        X[ep],
        U[ep],
        R[ep],
        fuel[ep],
        engine_torque[ep],
        engine_speed[ep],
        infeasible[ep] if infeasible is not None else None,
    )
