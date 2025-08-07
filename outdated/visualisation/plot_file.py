import os
import pickle
import sys

sys.path.append(os.getcwd())
from outdated.visualisation.plot import plot_evaluation

with open("miqp_mpc_N_15_c_1.pkl", "rb") as f:
    # with open("dev/results/evaluations/platoon_minlp_mpc_N_15_c_1_s_1_ms_2_t_50.pkl", "rb") as f:
    # with open("dev/results/evaluations/platoon_l_mpc_N_15_c_1_s_1_nocnstr.pkl", "rb") as f:
    # with open("dev/results/evaluations/platoon_heuristic_2_mpc_N_15_c_1_s_1.pkl", "rb") as f:
    data = pickle.load(f)
x_ref = data["x_ref"]
X = data["X"]
U = data["U"]
R = data["R"]
fuel = data["fuel"]
engine_torque = data["T_e"]
engine_speed = data["w_e"]
times = data["mpc_solve_time"]
if "heuristic" in data:
    heuristic = data["heuristic"]
    if isinstance(heuristic[0][0], list):
        heuristic = [
            [sum(heuristic[i][j]) for j in range(len(heuristic[i]))]
            for i in range(len(heuristic))
        ]
        print(f"heuristic count: {[sum(heuristic[i]) for i in range(len(heuristic))]}")
    else:
        print(
            f"heuristic count: {sum(sum(data["heuristic"][i]) for i in range(len(data["heuristic"])))}"
        )

ep = 0
plot_evaluation(
    x_ref[ep],
    X[ep],
    U[ep],
    R[ep],
    fuel[ep],
    engine_torque[ep],
    engine_speed[ep],
    mark=heuristic[ep] if "heuristic" in data else None,
)
