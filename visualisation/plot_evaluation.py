"""
Generate the box plot or violin plot from the evaluation results of a single controller.
"""

import os
import pickle
import sys

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

# Select experiments to plot
# List must be formatted as ["folder name", eval_name] where eval_name is the label
# used for the plot.
eval_list = [
    ["eval_l_mpc/eval_c1_s1_t5000000", "c1_s1"],
    ["eval_l_mpc/eval_c2_s3_t4000000", "c2_s3"],
    ["eval_l_mpc/eval_c2_s4_t4000000", "c2_s4"],
    ["eval_l_mpc/eval_c3_s3_t5000000", "c3_s3"],
    ["eval_l_mpc/eval_c3_s5_t5000000", "c3_s5"],
    ["eval_l_mpc/eval_c4_s3_t2900000", "c4_s3"],
    ["eval_l_mpc/eval_c4_s4_t2900000", "c4_s4"],
]
eval_metric = "ep_mean"  # {"ep_mean", "ep_sum", "ts"}

# TODO: remove and use eval_list instead
eval_name = "eval_l_mpc/eval_c1_s1_t5000000"

##### Collect data from .pkl files #####
# Locate data files
pkl_files = os.listdir(f"results/{eval_name}")
print(f"Found {len(pkl_files)} files in results/{eval_name}")

# Initialize variables
reward: list[np.ndarray] = []
time: list[list] = []

# Extract all data from .pkl files
for file in pkl_files:
    if file.endswith(".pkl"):
        with open(f"results/{eval_name}/{file}", "rb") as f:
            data = pickle.load(f)
            reward.append(data["R"][0])
            time.append(data["mpc_solve_time"])
    else:
        print(f"Skipping {file}, not a .pkl file")

# Episode-wise analysis
reward_ep = np.array([np.mean(r) for r in reward])
time_ep = np.array([np.mean(t) for t in time])
n_ep = len(reward_ep)

# Timestep-wise analysis
reward_ts = np.concatenate(reward, axis=0)
time_ts = np.concatenate(time, axis=0)
n_ts = len(reward_ts)

# Group data for plotting
df_reward = pd.DataFrame(
    {
        "Group": ["EP"] * n_ep * 2 + ["TS"] * n_ts * 2,
        "Type": ["reward"] * n_ep
        + ["dummy"] * n_ep
        + ["reward"] * n_ts
        + ["dummy"] * n_ts,
        "Value": np.concatenate(
            [reward_ep, -10e6 * np.ones(n_ep), reward_ts, -10e6 * np.ones(n_ts)]
        ),
    }
)
df_time = pd.DataFrame(
    {
        "Group": ["EP"] * n_ep * 2 + ["TS"] * n_ts * 2,
        "Type": ["dummy"] * n_ep + ["time"] * n_ep + ["dummy"] * n_ts + ["time"] * n_ts,
        "Value": np.concatenate(
            [-10e6 * np.ones(n_ep), time_ep, -10e6 * np.ones(n_ts), time_ts]
        ),
    }
)


##### Plot results #####
# Set seaborn style
c_reward = "royalblue"  # color for reward axis
c_time = "orange"  # color for time axis

# Initialize figure
print("Generating figure...")
fig, ax_r = plt.subplots(figsize=(10, 6))
ax_r.patch.set_visible(False)
ax_t = ax_r.twinx()  # time axis on the right side
ax_t.set_yscale("log")
ax_t.grid(True, which="major", linestyle="-", linewidth=0.6)
ax_t.grid(True, which="minor", linestyle=":", linewidth=0.4)
ax_t.set_axisbelow(True)
ax_r.set_zorder(ax_t.get_zorder() + 1)
# NOTE: Currently, adding a grid for the reward axis would draw it on top of the time
# distribution. This could probably be fixed by creating 2 more axes objects that only
# contain the grid lines, and draw them at the bottom of the z-order.
# TODO: Check if this is worth implementing.

# Set labels and limits
ax_r.set_xlabel("Policy")
ax_r.set_ylabel("Reward", color=c_reward)
ax_t.set_ylabel("Time (Log Scale)", color=c_time)
ax_r.set_ylim(0, 40)
ax_t.set_ylim(0.001, 10)
ax_t.set_xticks([0, 1], labels=["episode mean", "timesteps"])

# Generate the violin plots
# TODO: The violin plot could be replaced with a kde plot since the current
# implementation of the violin plot is equivalent to it.
# See:
# - https://seaborn.pydata.org/generated/seaborn.kdeplot.html
# - https://github.com/mwaskom/seaborn/issues/3619
gap: float = 0.025
inner = "quartile"  # "quartile", None

sns.violinplot(
    data=df_reward,
    x="Group",
    y="Value",
    hue="Type",
    ax=ax_r,
    color=c_reward,
    palette=[c_reward, c_time],
    orient="v",
    split=True,
    gap=gap,
    inner=inner,
    log_scale=False,
    legend=False,
    # cut=0,
)

sns.violinplot(
    data=df_time,
    x="Group",
    y="Value",
    hue="Type",
    ax=ax_t,
    color=c_time,
    palette=[c_reward, c_time],
    orient="v",
    split=True,
    gap=gap,
    inner=inner,
    log_scale=True,
    legend=False,
    # cut=0,
)

# add legend
h_reward = mpatches.Patch(color=c_reward, label=f"Reward")
h_time = mpatches.Patch(color=c_time, label=f"Time")
handles = [h_reward, h_time]
labels = ["Reward", "Time"]
ax_r.legend(
    handles,
    labels,
    loc="upper right",
    frameon=True,
)

plt.tight_layout()

print("Saving figure...")
fig.savefig(f"results/violin_plot.png", dpi=300, bbox_inches="tight")
