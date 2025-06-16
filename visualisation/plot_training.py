import os
import pickle
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
# from utils.tikz import save2tikz

skip = 10000
average_interval = 10000
show_individual_lines = False

file_names = [
    "dev/results/1/seeds/1/data_step_5000000.pkl",
    "dev/results/1/seeds/2/data_step_5000000.pkl",
    "dev/results/1/seeds/3/data_step_5000000.pkl",
    "dev/results/1/seeds/4/data_step_5000000.pkl",
    "dev/results/1/seeds/5/data_step_5000000.pkl",
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
    L_f.append([f for sub_f in fuel for f in sub_f])
    kappa.append([i for sub_i in infeasible for i in sub_i])
    # kappa.append(heuristic)

data = [L, L_t, L_f, kappa]
data_avg = [
    np.array(
        [
            np.convolve(l, np.ones(average_interval) / average_interval, mode="valid")
            for l in d
        ]
    )[:, ::skip]
    for d in data
]
data_df = [pd.DataFrame(d.T, columns=file_names) for d in data_avg]
for d in data_df:
    d["x"] = np.arange(len(d))
data_df_long = [d.melt(id_vars="x", var_name="seed", value_name="L") for d in data_df]

# Plot results
if show_individual_lines:
    fig, ax = plt.subplots(4, 1, sharex=True)
    sns.lineplot(
        data=data_df_long[0], x="x", y="L", errorbar="sd", ax=ax[0], hue="seed"
    )
    ax[0].set_ylabel("L")
    sns.lineplot(
        data=data_df_long[1], x="x", y="L", errorbar="sd", ax=ax[1], hue="seed"
    )
    ax[1].set_ylabel("L_t")
    sns.lineplot(
        data=data_df_long[2], x="x", y="L", errorbar="sd", ax=ax[2], hue="seed"
    )
    ax[2].set_ylabel("L_f")
    sns.lineplot(
        data=data_df_long[3], x="x", y="L", errorbar="sd", ax=ax[3], hue="seed"
    )
    ax[3].set_ylabel("kappa")
    plt.show()
else:
    fig, ax = plt.subplots(4, 1, sharex=True)
    sns.lineplot(data=data_df_long[0], x="x", y="L", errorbar="sd", ax=ax[0])
    ax[0].set_ylabel("L")
    sns.lineplot(data=data_df_long[1], x="x", y="L", errorbar="sd", ax=ax[1])
    ax[1].set_ylabel("L_t")
    sns.lineplot(data=data_df_long[2], x="x", y="L", errorbar="sd", ax=ax[2])
    ax[2].set_ylabel("L_f")
    sns.lineplot(data=data_df_long[3], x="x", y="L", errorbar="sd", ax=ax[3])
    ax[3].set_ylabel("kappa")
    plt.show()
