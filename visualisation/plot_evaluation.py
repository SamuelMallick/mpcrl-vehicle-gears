"""
Generate the box plot or violin plot from the evaluation results of a single controller.
"""

# TODO: add 2 new axis objects to manage grid lines separately from the data plots

import os
import pickle
import sys

import seaborn as sns
import matplotlib
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
grouping_r = "ep_sum"  # {ep_sum, ep_mean, ts} default is "ep_sum"
grouping_t = "ts"  # {ep_sum, ep_mean, ts} default is "ts"

# Initialize data containers
df_reward = pd.DataFrame(columns=["Policy", "Variable", "Value"])
df_time = pd.DataFrame(columns=["Policy", "Variable", "Value"])
max_time = []
xticks_labels = []

# Collect data
for eval_name, eval_label in eval_list:

    # Reset data containers
    reward: list[np.ndarray] = []
    time: list[list] = []

    # Locate data files for current evaluation
    pkl_files = os.listdir(f"results/{eval_name}")
    print(f"Found {len(pkl_files)} files in results/{eval_name}")

    # Extract all data from .pkl files
    for file in pkl_files:
        if file.endswith(".pkl"):
            with open(f"results/{eval_name}/{file}", "rb") as f:
                data = pickle.load(f)
                reward.append(data["R"][0])
                time.append(data["mpc_solve_time"])
        else:
            print(f"Skipping {file}, not a .pkl file")

    # Extract information to plot
    match grouping_r:

        case "ep_sum":
            reward = np.array([np.sum(r) for r in reward])

        case "ep_mean":
            reward = np.array([np.mean(r) for r in reward])

        case "ts":
            reward = np.concatenate(reward, axis=0)

        case _:
            raise ValueError(f"Unknown evaluation metric: {grouping_r}")

    match grouping_t:

        case "ep_sum":
            time = np.array([np.sum(t) for t in time])

        case "ep_mean":
            time = np.array([np.mean(t) for t in time])

        case "ts":
            time = np.concatenate(time, axis=0)
            max_time.append(np.max(time))

        case _:
            raise ValueError(f"Unknown evaluation metric: {grouping_t}")

    # Build temporary dataframes
    # NOTE: dummy np.empty(1) are used to ensure that violin hue split works properly
    n_r = len(reward)
    n_t = len(time)
    df_reward_temp = pd.DataFrame(
        {
            "Group": [eval_label] * (n_r + 1),
            "Type": ["reward"] * n_r + ["dummy"] * 1,
            "Value": np.concatenate([reward, 10e-10 * np.ones(1)]),
        }
    )
    df_time_temp = pd.DataFrame(
        {
            "Group": [eval_label] * (n_t + 1),
            "Type": ["dummy"] * 1 + ["time"] * n_t,
            "Value": np.concatenate([10e-10 * np.ones(1), time]),
        }
    )

    # Append data to the data structures
    if df_reward.empty:
        df_reward = df_reward_temp
        df_time = df_time_temp
    else:
        df_reward = pd.concat([df_reward, df_reward_temp])
        df_time = pd.concat([df_time, df_time_temp])

    # Store the label for the x-axis
    xticks_labels.append(eval_label)

##### Plot results #####

# Set plot colors
c_reward = "lightblue"
c_reward_dark = "cadetblue"
c_time = "salmon"
c_time_dark = "orangered"
c_time_dark2 = "darkred"
# c_time = "gold"
# c_time_dark = "goldenrod"
# c_time_dark2 = "darkgoldenrod"

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
# TODO: Currently, adding a grid for the reward axis would draw it on top of the time
# distribution. This could probably be fixed by creating 2 more axes objects that only
# contain the grid lines, and draw them at the bottom of the z-order. Check if this is
# worth implementing.

# Set labels and limits
match grouping_r:

    case "ep_mean":
        ax_r.set_ylim(0, 10)
        ax_r.set_ylabel(
            "Average timestep reward over episode",
            color=c_reward_dark,
        )
        cut_r = 2

    case "ep_sum":
        ax_r.set_ylim(5_000, 11_000)
        ax_r.set_ylabel(
            "Episode cumulative reward",
            color=c_reward_dark,
        )
        cut_r = 2

    case "ts":
        ax_r.set_ylim(0, 40)
        ax_r.set_ylabel(
            "Timestep reward",
            color=c_reward_dark,
        )
        cut_r = 0

match grouping_t:

    case "ep_mean":
        ax_t.set_ylim(0.01, 1)
        ax_t.set_ylabel(
            "Average timestep time over episode (Log Scale) [s]",
            color=c_time_dark,
        )
        cut_t = 2

    case "ep_sum":
        ax_t.set_ylim(10, 1000)
        ax_t.set_ylabel(
            "Episode cumulative time (Log Scale) [s]",
            color=c_time_dark,
        )
        cut_t = 2

    case "ts":
        ax_t.set_ylim(0.01, 10)
        ax_t.set_ylabel(
            "Timestep time (Log Scale) [s]",
            color=c_time_dark,
        )
        cut_t = 0

ax_t.set_xticks(list(range(len(xticks_labels))), labels=xticks_labels)
ax_r.set_xlabel("Policy")
ax_r.set_title("Policies Evaluation")

# Generate the violin plots
# TODO: The violin plot could be replaced with a kde plot since the current
# implementation of the violin plot is equivalent to it.
# - https://seaborn.pydata.org/generated/seaborn.kdeplot.html
# - https://github.com/mwaskom/seaborn/issues/3619.

# Violin plot parameters
gap = 0.05
inner = "quartile"  # {"quartile", None}
linewidth = 1.5

sns.violinplot(
    data=df_reward,
    x="Group",
    y="Value",
    hue="Type",
    ax=ax_r,
    color=c_reward,
    palette=[c_reward, c_time],
    fill=True,
    linewidth=linewidth,
    linecolor=c_reward_dark,
    orient="v",
    split=True,
    gap=gap,
    inner=inner,
    log_scale=False,
    legend=False,
    cut=cut_r,
)

sns.violinplot(
    data=df_time,
    x="Group",
    y="Value",
    hue="Type",
    ax=ax_t,
    color=c_time,
    fill=True,
    linewidth=linewidth,
    linecolor=c_time_dark,
    palette=[c_reward, c_time],
    orient="v",
    split=True,
    gap=gap,
    inner=inner,
    log_scale=True,
    legend=False,
    cut=cut_t,
)

# Add max time marker
if grouping_t == "ts":
    x_ticks = np.arange(len(xticks_labels)) + gap / 2
    plt.plot(
        x_ticks,
        max_time,
        marker=matplotlib.markers.CARETLEFT,
        color=c_time_dark,
        linestyle="None",
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
