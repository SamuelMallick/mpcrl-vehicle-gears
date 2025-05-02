import pickle
import sys, os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sys.path.append(os.getcwd())
# from utils.tikz import save2tikz

skip = 10000
average_interval = 10000

file_names = [
    "dev/results/1_seeds/1/data_step_5000000.pkl",
    "dev/results/1_seeds/2/data_step_5000000.pkl",
    "dev/results/1_seeds/3/data_step_5000000.pkl",
    "dev/results/1_seeds/4/data_step_3800000.pkl",
    "dev/results/1_seeds/5/data_step_3875000.pkl",
]

L = []
L_t = []
L_f = []
kappa = []

for file_name in file_names:
    with open(file_name, "rb") as f:
        data = pickle.load(f)

    cost = data["cost"]
    fuel = data["fuel"]
    R = data["R"]
    tracking = [r - f for sub_r, sub_f in zip(R, fuel) for r, f in zip(sub_r, sub_f)]
    if "infeasible" in data:
        infeasible = data["infeasible"]
    if "heuristic" in data:
        heuristic = data["heuristic"]

    L.append([l for sub_l in cost for l in sub_l])
    L_t.append(tracking)
    L_f.append(fuel)
    kappa.append(infeasible)
    # kappa.append(heuristic)

L_avg = np.array(
    [
        np.convolve(l, np.ones(average_interval) / average_interval, mode="valid")
        for l in L
    ]
)[:, ::skip]
L_df = pd.DataFrame(L_avg.T, columns=file_names)
L_df["x"] = np.arange(len(L_df))
df_long = L_df.melt(id_vars="x", var_name="seed", value_name="L")
sns.lineplot(data=df_long, x="x", y="L", errorbar="sd")
plt.show()


# np.random.seed(0)
# df = pd.DataFrame({
#     "x": np.tile(np.arange(10), 3),  # 10 x-values repeated 20 times
#     "y": np.random.randn(30) + np.tile(np.arange(10), 3),
# })

# # Lineplot with standard deviation as shaded region
# sns.lineplot(data=df, x="x", y="y", errorbar='sd')  # Use 'sd' for standard deviation shading
# plt.show()
