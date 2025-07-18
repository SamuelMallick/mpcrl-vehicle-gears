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
from scipy.integrate import simpson
from scipy.stats import gaussian_kde

sys.path.append(os.getcwd())

# Select experiments to plot
# List must be formatted as ["folder name", eval_name] where eval_name is the label
# used for the plot. The experiment folder name (first layer under results/) can be
# specified collectively in the variable `experiment_folder`.
experiment_folder = "eval_single_agent"
eval_list = [
    # ["eval_l_mpc/eval_c1_s1_t5000000", "c1_s1"],
    # ["eval_l_mpc/eval_c2_s3_t4000000", "c2_s3"],
    # ["eval_l_mpc/eval_c2_s4_t4000000", "c2_s4"],
    # ["eval_l_mpc/eval_c3_s1_t5000000", "c3_s1"],
    # ["eval_l_mpc/eval_c3_s3_t5000000", "c3_s3"],
    ["eval_l_mpc/eval_c3_s5_t5000000", "c3_s5"],
    ["eval_l_mpc/eval_c4_s3_t2900000", "c4_s3"],
    ["eval_l_mpc/eval_c4_s4_t2900000", "c4_s4"],
    ["eval_miqp", "miqp"],
    ["eval_heuristic_mpc_1", "h_1"],
    ["eval_heuristic_mpc_2", "h_2"],
    ["eval_heuristic_mpc_3", "h_3"],
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
    pkl_files = os.listdir(f"results/{experiment_folder}/{eval_name}")
    print(f"Found {len(pkl_files)} files in results/{experiment_folder}/{eval_name}")

    # Extract all data from .pkl files
    for file in pkl_files:
        if file.endswith(".pkl"):
            with open(f"results/{experiment_folder}/{eval_name}/{file}", "rb") as f:
                data = pickle.load(f)
                reward.append(data["R"][0])
                time.append(data["mpc_solve_time"])
        else:
            print(f"Skipping {file}, not a .pkl file")

    # Extract information to plot
    match grouping_r:

        case "ep_sum":
            # Divide by 1000 for ease of plot (e.g., 1000 instead of 1000000)
            reward = np.array([np.sum(r) / 1000 for r in reward])

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
# ax_t.grid(True, which="major", linestyle="-", linewidth=0.6)
# ax_t.grid(True, which="minor", linestyle=":", linewidth=0.4)
# ax_t.set_axisbelow(True)

# Set labels and limits
match grouping_r:

    case "ep_mean":
        ax_r.set_ylim(0, 10)
        ax_r.set_ylabel(
            "Average timestep cost over episode",
            color=c_reward_dark,
        )
        cut_r = 0

    case "ep_sum":
        ax_r.set_ylim(5, 11)
        ax_r.set_ylabel(
            "Episode cumulative cost (x 1000)",
            color=c_reward_dark,
        )
        cut_r = 0

    case "ts":
        ax_r.set_ylim(0, 40)
        ax_r.set_ylabel(
            "Timestep cost",
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
        cut_t = 0

    case "ep_sum":
        ax_t.set_ylim(10, 1000)
        ax_t.set_ylabel(
            "Episode cumulative time (Log Scale) [s]",
            color=c_time_dark,
        )
        cut_t = 0

    case "ts":
        ax_t.set_ylim(0.01, 2000)
        ax_t.set_ylabel(
            "Timestep time (Log Scale) [s]",
            color=c_time_dark,
        )
        cut_t = 0

ax_t.set_xticks(list(range(len(xticks_labels))), labels=xticks_labels)
ax_r.set_xlabel("Policy")
ax_r.set_title("Policies Evaluation")

# Reward grid lines
ax_grid_r = fig.add_axes(ax_r.get_position(), frameon=False)
ax_grid_r.set_xticks([])
ax_grid_r.set_yticks([])
ax_grid_r.set_facecolor("none")
ax_grid_r.set_xlim(ax_r.get_xlim())
ax_grid_r.set_ylim(ax_r.get_ylim())
ax_grid_r.set_yticks(ax_r.get_yticks(), minor=False)
ax_grid_r.yaxis.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=1)

# Time grid lines
ax_grid_t = ax_grid_r.twinx()
ax_grid_t.set_yscale("log")
ax_grid_t.set_xticks([])
ax_grid_t.set_yticks([])
ax_grid_t.set_facecolor("none")
ax_grid_t.set_xlim(ax_t.get_xlim())
ax_grid_t.set_ylim(ax_t.get_ylim())
ax_grid_t.set_yticks(ax_t.get_yticks()[1:-2], minor=False)  # hacky but it works
ax_grid_t.yaxis.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=1)
ax_grid_t.yaxis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.8)
ax_t.set_axisbelow(True)

# add vertical line for 1st violin plot
for i in range(len(xticks_labels)):
    ax_t.axvline(i, color="gray", linestyle="-", linewidth=0.6, alpha=1)

# Generate the violin plots
# NOTE: The violin plot could be replaced with a kde plot since the current
# implementation of the violin plot is equivalent to it.
# - https://seaborn.pydata.org/generated/seaborn.kdeplot.html
# - https://github.com/mwaskom/seaborn/issues/3619.

# Violin plot parameters
gap = 0.1
inner = "quartile"  # {"quartile", None}
linewidth = 1.2

# Reward violin plot
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

# Time violin plot
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

# Add mean reward marker
if grouping_r == "ep_sum":
    avg_reward = []
    df = df_reward[df_reward["Type"] == "reward"]
    for _, eval_label in eval_list:
        data = df[df["Group"] == eval_label]["Value"]
        avg_reward.append(np.mean(data))
    x_marker_avg_reward = np.arange(len(xticks_labels)) - gap / 2
    ax_r.plot(
        x_marker_avg_reward,
        avg_reward,
        marker=matplotlib.markers.CARETRIGHT,
        color=c_reward_dark,
        linestyle="None",
    )

    for i, r in enumerate(avg_reward):
        ax_r.annotate(
            f"{r:.2f}",
            xy=(x_marker_avg_reward[i], r),
            xytext=(-6, 0),  # offset text
            textcoords="offset points",
            color=c_reward_dark,
            fontsize=8,
            ha="right",
            va="center",
        )

# Add max time marker
if grouping_t == "ts":
    x_marker_max_time = np.arange(len(xticks_labels)) + gap / 2
    ax_t.plot(
        x_marker_max_time,
        max_time,
        marker=matplotlib.markers.CARETLEFT,
        color=c_time_dark,
        linestyle="None",
    )

    for i, t in enumerate(max_time):
        ax_t.annotate(
            f"{t:.2f}",
            xy=(x_marker_max_time[i], t),
            xytext=(6, 0),  # offset text
            textcoords="offset points",
            color=c_time_dark,
            fontsize=8,
            ha="left",
            va="center",
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

# set zorder of the axes
ax_grid_r.set_zorder(1)
ax_grid_t.set_zorder(2)
ax_t.set_zorder(3)
ax_r.set_zorder(4)

# Save figure
print("Saving figure...")
fig.savefig(f"results/violin_plot.png", dpi=300, bbox_inches="tight")
