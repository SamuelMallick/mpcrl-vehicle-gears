"""
Generate the box plot or violin plot from the evaluation results of a single controller.

Note: some parts of the script have been tailored to the plotting of the results of the
experiments obtained during the evaluation experiment used in the paper, for example
when defining the offset of the markers from the y line to avoid overlapping with the
distribution. These values may need to be updated when plotting the results of different
experiments in order to obtain a clean plot.
"""

import os
import pickle
import sys

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from packaging import version

sys.path.append(os.getcwd())
from utils.plot_fcns import cm2inch

##### Plot settings ####################################################################

# Save settings
save_png = True
save_pgf = True
save_tikz = False

# Plot settings
fig_size_x = 16.5  # cm
fig_size_y = 6  # cm
show_legend = False
show_title = False

# Select experiments to plot
eval_type = "eval_platoon"
# AVAILABLE OPTIONS:
# - eval_single
# - eval_platoon
# - eval_single_seed_10
# - eval_platoon_seed_10

# Select the plot parameters
use_relative_performance = True  # Show the relative performance of the policies
baseline_solution = "MINLP"

# Other options (better not change -- alternative values have not really been tested)
plot_type = "violin"  # {"box", "violin"} default is "violin"
show_r_mean_marker = True  # Show the mean reward marker on the reward plot
show_t_max_marker = True  # Show the max time marker on the time plot
color_grid_lines = False  # Color the gridlines of the axes colors
grouping_r = "ep_sum"  # {ep_sum, ep_mean, ts} default is "ep_sum"
grouping_t = "ts"  # {ep_sum, ep_mean, ts} default is "ts"

# Match eval_type
# List must be formatted as ["folder name", "label"] where the label is the one used
# for the plot. The experiment folder name (first folder layer under results/) has to
# match the variable `eval_type`.
match eval_type:

    case "eval_single":
        eval_list = [
            ["eval_l_mpc/c3_seed1", "LC-1"],
            ["eval_l_mpc/c4_seed4", "LC-2"],
            # ["eval_l_mpc/c4_seed4_laptop", "LC-2 PC"],
            ["eval_miqp", "MIQP"],
            ["eval_miqp_1s", "MIQP-tl"],
            ["eval_minlp", "MINLP"],
            ["eval_minlp_1s", "MINLP-tl"],
            ["eval_heuristic_mpc_1", "HD"],
            ["eval_heuristic_mpc_2", "HC"],
            ["eval_heuristic_mpc_3", "HS"],
        ]

    case "eval_platoon":
        eval_list = [
            ["eval_l_mpc/c3_seed1", "LC-1"],
            ["eval_l_mpc/c4_seed4", "LC-2"],
            # ["eval_l_mpc/c4_seed4_laptop", "LC-2 PC"],
            # ["eval_l_mpc/c4_seed4_laptop_1s", "LC-2 PC tl"],
            ["eval_miqp", "MIQP"],
            ["eval_miqp_1s", "MIQP-tl"],
            ["eval_minlp_720", "MINLP"],
            # ["eval_minlp_3600", "MINLP-long"],
            # ["eval_minlp_1s", "MINLP-tl"],  # TODO
            ["eval_heuristic_mpc_1", "HD"],
            ["eval_heuristic_mpc_2", "HC"],
            ["eval_heuristic_mpc_3", "HS"],
        ]

    case "eval_single_seed_10":
        eval_list = [
            ["eval_l_mpc/c3_seed1", "LC-1"],
            ["eval_l_mpc/c4_seed4", "LC-2"],
            ["eval_l_mpc/c4_seed4_laptop", "LC-2 PC"],
            ["eval_miqp", "MIQP"],
            ["eval_miqp_1s", "MIQP-tl"],
            ["eval_minlp", "MINLP"],
            # ["eval_minlp_1s", "MINLP-tl"],  # TODO
            ["eval_heuristic_mpc_1", "HD"],
            ["eval_heuristic_mpc_2", "HC"],
            ["eval_heuristic_mpc_3", "HS"],
        ]

    case "eval_platoon_seed_10":
        eval_list = [
            ["eval_l_mpc/c3_seed1", "LC-1"],
            ["eval_l_mpc/c4_seed4", "LC-2"],
            # ["eval_l_mpc/c4_seed4_laptop", "LC-2 PC"],
            ["eval_miqp", "MIQP"],
            ["eval_miqp_1s", "MIQP-tl"],
            ["eval_minlp", "MINLP"],
            ["eval_minlp", "MINLP-tl"],
            # ["eval_minlp_1s", "MINLP 1s"],  TODO
            ["eval_heuristic_mpc_1", "HD"],
            ["eval_heuristic_mpc_2", "HC"],
            ["eval_heuristic_mpc_3", "HS"],
        ]

    case _:
        print("Unknown evaluation type")
        sys.exit()

##### Preprocess data ##################################################################

# List of specific experiments to include (by their seed, i.e., last 4 digits of name)
experiments_list = [
    # "1001",
    # "1002",
    # "1003",
    # "1004",
    # "1005",
    "1006",
    "1007",
    "1008",
    "1009",
    "1010",
    "1011",
    "1012",
]

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
    pkl_files = os.listdir(f"results/{eval_type}/{eval_name}")
    print(f"Found {len(pkl_files)} files in results/{eval_type}/{eval_name}")

    # Extract all data from .pkl files
    for _, file in enumerate(pkl_files):
        if file.endswith(".pkl"):

            if experiments_list and file[-8:-4] not in experiments_list:
                continue

            with open(f"results/{eval_type}/{eval_name}/{file}", "rb") as f:
                data = pickle.load(f)
                reward.append(data["R"][0])
                if eval_name == "eval_minlp_1s":
                    t = data["t_primary_mpc"]
                else:
                    t = data["mpc_solve_time"]

                # For platooning, sum solve time of each vehicle to get platoon time
                if len(t) > 1000:
                    t = [np.sum(o) for o in np.split(t, 1000)]

                # Append solve time
                time.append(t)

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
    # NOTE: dummy 10e-10 * np.ones(1) ensures that violin hue split works properly
    n_r = len(reward)
    n_t = len(time)
    df_reward_temp = pd.DataFrame(
        {
            "Group": [eval_label] * (n_r + 1),
            "Type": ["reward"] * n_r + ["dummy"] * 1,
            "Value": np.concatenate([reward, -10e10 * np.ones(1)]),
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

# Compute relative performance if required
if use_relative_performance is True:
    J_baseline = df_reward[
        (df_reward["Group"] == baseline_solution) & (df_reward["Type"] == "reward")
    ]["Value"].values

    # Update dataframe with relative performance
    for eval_name, eval_label in eval_list:
        mask = (df_reward["Group"] == eval_label) & (df_reward["Type"] == "reward")
        J_policy = df_reward[mask]["Value"].values
        relative_J = (J_policy - J_baseline) / J_baseline * 100
        df_reward.loc[mask, "Value"] = (J_policy - J_baseline) / J_baseline * 100

##### Plot results #####################################################################

if version.parse(mpl.__version__) <= version.parse("3.7"):
    if save_pgf is True:
        save_pgf = True
    if save_tikz is True:
        save_tikz = True
else:
    if save_pgf is True:
        save_pgf = False
        print("PGF export is not supported in this version of matplotlib.")
    if save_tikz is True:
        save_tikz = False
        print("TikZ export is not supported in this version of matplotlib.")

# Set plot colors
c_reward = "lightblue"
c_reward_dark = "cadetblue"
c_time = "salmon"
c_time_dark = "orangered"
c_time_dark2 = "darkred"
# c_time = "gold"
# c_time_dark = "goldenrod"
# c_time_dark2 = "darkgoldenrod"

# Plot parameters
linewidth = 1.2
gap = 0.1
inner = "quartile"  # {"quartile", None} -- only for violin plots
labels_font_size = 10
tick_labels_font_size = 8
markers_font_size = 7

# Set matplotlib parameters
mpl.rcParams.update(
    {
        "pgf.texsystem": "xelatex",  # or any other engine you want to use
        "text.usetex": True,  # use TeX for all texts
        "font.family": "serif",
        "font.size": labels_font_size,
        "axes.labelsize": labels_font_size,
        "legend.fontsize": labels_font_size,
        "xtick.labelsize": tick_labels_font_size,
        "ytick.labelsize": tick_labels_font_size,
        "pgf.rcfonts": False,
        "pgf.preamble": "\\usepackage[T1]{fontenc}",  # extra preamble for LaTeX
    }
)

# Initialize figure
print("Generating figure...")
fig_size_x = cm2inch(fig_size_x)
fig_size_y = cm2inch(fig_size_y)
fig, ax_r = plt.subplots(figsize=(fig_size_x, fig_size_y))
fig.tight_layout()  # Avoid plt.tight_layout() as it messes with the multiple axes
ax_r.patch.set_visible(False)
ax_t = ax_r.twinx()  # time axis on the right side
ax_t.set_yscale("log")

# Set reward labels and limits
cut_r = 0
if use_relative_performance is True:
    if eval_type == "eval_single":
        ax_r.set_ylim(-1, 28)
    elif eval_type == "eval_single_seed_10":
        ax_r.set_ylim(-1, 31)
    elif eval_type == "eval_platoon":
        ax_r.set_ylim(-1, 25)  # TODO: update values
    elif eval_type == "eval_platoon_seed_10":
        ax_r.set_ylim(-1, 28)
    ax_r.set_ylabel(
        "$\\Delta J$ [\\%]",  # Relative performance drop
        color=c_reward_dark,
    )
    cut_r = 0

else:
    match grouping_r:

        case "ep_mean":
            ax_r.set_ylim(0, 10)
            ax_r.set_ylabel(
                "Average timestep cost over episode",
                color=c_reward_dark,
            )
            cut_r = 0

        case "ep_sum":
            if eval_type == "eval_single":
                ax_r.set_ylim(5, 10)
            elif eval_type == "eval_platoon":
                ax_r.set_ylim(25, 55)
            else:
                ax_r.set_ylim(0, 100)
            ax_r.set_ylabel(
                "Episode cumulative cost",
                color=c_reward_dark,
            )
            ax_r.text(
                -0.01,
                1.03,
                r"$\times 10^3$",
                transform=ax_r.transAxes,
                ha="right",
                va="bottom",
                fontsize=tick_labels_font_size,
            )
            cut_r = 0

        case "ts":
            ax_r.set_ylim(0, 40)
            ax_r.set_ylabel(
                "Timestep cost",
                color=c_reward_dark,
            )
            cut_r = 0

# Set time labels and limits
cut_t = 0
match grouping_t:

    case "ep_mean":
        ax_t.set_ylim(0.01, 1)
        ax_t.set_ylabel(
            "Average timestep time over episode [s]",
            color=c_time_dark,
        )
        cut_t = 0

    case "ep_sum":
        ax_t.set_ylim(10, 1000)
        ax_t.set_ylabel(
            "Episode cumulative time [s]",
            color=c_time_dark,
        )
        cut_t = 0

    case "ts":
        ax_t.set_ylim(0.008, 3000)
        ax_t.set_ylabel(
            "Timestep solution time [s]",
            color=c_time_dark,
        )
        cut_t = 0

# Reward axis settings
ax_r.set_xticks(list(range(len(xticks_labels))))
ax_r.set_xticklabels(xticks_labels)
ax_r.tick_params(axis="x", labelrotation=0)  # 90
if show_title is True:
    ax_r.set_xlabel("Policy")
    ax_r.set_title("Policies Evaluation")
elif eval_type in ["eval_single", "eval_single_seed_10"]:
    ax_r.set_xlabel("(a) $M=1$")
elif eval_type in ["eval_platoon", "eval_platoon_seed_10"]:
    ax_r.set_xlabel("(b) $M=5$")

# Vertical grid lines
for i in range(len(xticks_labels)):
    ax_r.axvline(i, color="gray", linestyle="-", linewidth=0.6, alpha=1)

# Reward grid lines
ax_grid_r = fig.add_axes(ax_r.get_position(), frameon=False)
ax_grid_r.set_xticks([])
ax_grid_r.set_yticks([])
ax_grid_r.set_facecolor("none")
ax_grid_r.set_yticks(ax_r.get_yticks(), minor=False)
ax_grid_r.set_xlim(ax_r.get_xlim())
ax_grid_r.set_ylim(ax_r.get_ylim())
if color_grid_lines is True:
    ax_grid_r.yaxis.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.6,
        alpha=1,
        color=c_reward_dark,
    )
else:
    ax_grid_r.yaxis.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.6,
        alpha=1,
        color="gray",
    )

# Time grid lines
ax_grid_t = ax_grid_r.twinx()
ax_grid_t.set_yscale("log")
ax_grid_t.set_xticks([])
ax_grid_t.set_yticks([])
ax_grid_t.set_facecolor("none")
ax_grid_t.set_xlim(ax_t.get_xlim())
ax_grid_t.set_ylim(ax_t.get_ylim())
ax_grid_t.set_yticks(ax_t.get_yticks()[2:-2], minor=False)  # a bit hacky but it works
if color_grid_lines is True:
    ax_grid_t.yaxis.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.6,
        alpha=1,
        color=c_time_dark,
    )
    ax_grid_t.yaxis.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.4,
        alpha=0.8,
        color=c_time_dark,
    )
else:
    ax_grid_t.yaxis.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.6,
        alpha=1,
    )
    ax_grid_t.yaxis.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.4,
        alpha=0.8,
    )
ax_grid_t.set_axisbelow(True)

# Set the plot type
if plot_type == "violin":

    # NOTE: The violin plot could be replaced with a kde plot since the current
    # implementation of the violin plot is equivalent to it.
    # - https://seaborn.pydata.org/generated/seaborn.kdeplot.html
    # - https://github.com/mwaskom/seaborn/issues/3619.

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

elif plot_type == "box":

    # Reward box plot
    sns.boxplot(
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
        gap=gap,
        log_scale=False,
        legend=False,
    )

    # Time box plot
    sns.boxplot(
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
        gap=gap,
        log_scale=True,
        legend=False,
    )

else:
    raise ValueError(f"Unknown plot type: {plot_type}")

# Mean reward marker
if show_r_mean_marker is True:

    # Calculate the average reward for each evaluation
    avg_reward = []
    df = df_reward[df_reward["Type"] == "reward"]  # Remove dummy data
    for _, eval_label in eval_list:
        avg_reward.append(np.mean(df[df["Group"] == eval_label]["Value"]))

    # Set offset for the mean marker (to avoid overlap with the plot)
    # The current values have been manually determined for each evaluation type.
    match eval_type:
        case "eval_single":
            r_marker_offset = np.array(
                [
                    0.16,  # LC-2
                    0.35,  # LC-2
                    # 0.35,  # LC-2 PC
                    0.13,  # MIQP
                    0.14,  # MIQP-tl
                    -0.04,  # MINLP
                    0.075,  # MINLP-tl
                    0.025,  # HD
                    0.17,  # HC
                    0.14,  # HS
                ]
            )
        case "eval_platoon":
            r_marker_offset = -np.ones(9) * 0.00  # TODO: update with correct values
        case "eval_single_seed_10":
            r_marker_offset = -np.ones(9) * 0.04
        case "eval_platoon_seed_10":
            r_marker_offset = -np.ones(9) * 0.045  # TODO: update to 8 after adding 1s
    if len(avg_reward) != len(r_marker_offset):
        r_marker_offset = np.zeros(len(avg_reward))  # no offset if wrong dimensions

    # Define x marker positions
    x_marker_avg_reward = np.arange(len(xticks_labels)) - gap / 2 - r_marker_offset

    # Plot the mean reward markers
    ax_r.plot(
        x_marker_avg_reward,
        avg_reward,
        marker=mpl.markers.CARETRIGHT,
        markersize=5,
        color=c_reward_dark,
        linestyle="None",
    )

    # Annotate the mean reward markers
    for i, r in enumerate(avg_reward):
        if eval_type in ["eval_single_seed_10", "eval_platoon_seed_10"]:
            y_offset = 3.5
        else:
            if eval_list[i][1] == "MINLP":
                y_offset = 3
            else:
                y_offset = 0
        ax_r.annotate(
            f"{r:.2f}",
            xy=(x_marker_avg_reward[i], r),
            xytext=(-4, y_offset),  # offset text
            textcoords="offset points",
            color=c_reward_dark,
            fontsize=markers_font_size,
            ha="right",
            va="center",
        )

# Max time marker
if show_t_max_marker is True:

    match eval_type:
        case "eval_single":
            t_marker_offset = np.array(
                [
                    0,  # LC-2
                    0,  # LC-2
                    # 0,  # LC-2 PC
                    0,  # MIQP
                    0.17,  # MIQP-tl
                    0,  # MINLP
                    0.14,  # MINLP-tl
                    0,  # HD
                    0,  # HC
                    0,  # HS
                ]
            )
        case "eval_platoon":
            t_marker_offset = np.ones(9) * 0.00  # TODO
        case "eval_single_seed_10":
            t_marker_offset = np.ones(9) * 0.00  # TODO
        case "eval_platoon_seed_10":
            t_marker_offset = np.array(
                [
                    0,  # LC-2
                    0,  # LC-2
                    # 0,  # LC-2 PC
                    0,  # MIQP
                    0.36,  # MIQP-tl
                    0,  # MINLP
                    0,  # MINLP-tl
                    0,  # HD
                    0,  # HC
                    0,  # HSC
                ]
            )
    if len(t_marker_offset) != len(max_time):
        t_marker_offset = np.zeros(len(max_time))  # no offset if wrong dimensions

    if grouping_t == "ts":
        x_marker_max_time = np.arange(len(xticks_labels)) + gap / 2 + t_marker_offset
        ax_t.plot(
            x_marker_max_time,
            max_time,
            marker=mpl.markers.CARETLEFT,
            markersize=5,
            color=c_time_dark,
            linestyle="None",
        )

        for i, t in enumerate(max_time):
            ax_t.annotate(
                f"{t:.2f}",
                xy=(x_marker_max_time[i], t),
                xytext=(7, 0),  # offset text
                textcoords="offset points",
                color=c_time_dark,
                fontsize=markers_font_size,
                ha="left",
                va="center",
            )

# Add legend
if show_legend is True:
    h_reward = mpatches.Patch(color=c_reward, label="Reward")
    h_time = mpatches.Patch(color=c_time, label="Time")
    handles = [h_reward, h_time]
    labels = ["Reward", "Time"]
    ax_r.legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
    )

# Set zorder of the axes
ax_grid_r.set_zorder(1)
ax_grid_t.set_zorder(2)
ax_t.set_zorder(3)
ax_r.set_zorder(4)

# Save figures
if save_png:
    print("Saving png...")
    fig.savefig(f"plots/{eval_type}.png", dpi=300, bbox_inches="tight")

if save_tikz:
    from utils.tikz import save2tikz  # import tikzplotlib only if supported

    print("Saving tikz...")
    save2tikz(plt.gcf(), name=f"plots/{eval_type}.tex")

if save_pgf:
    mpl.use("pgf")  # This line must be after the execution of save2tikz (?)
    print("Saving pgf...")
    fig.savefig(f"plots/{eval_type}.pgf", bbox_inches="tight")

# Generate and print table
# [mean, std, median, min, max]
table_r = pd.DataFrame(
    columns=["mean", "std", "median", "min", "max"],
    index=xticks_labels,
)
for i, j in enumerate(eval_list):
    eval_name = j[0]
    eval_label = j[1]
    r = df_reward[(df_reward["Group"] == eval_label) & (df_reward["Type"] == "reward")][
        "Value"
    ].values
    table_r.loc[eval_label, "mean"] = np.mean(r)
    table_r.loc[eval_label, "std"] = np.std(r)
    table_r.loc[eval_label, "median"] = np.median(r)
    table_r.loc[eval_label, "min"] = np.min(r)
    table_r.loc[eval_label, "max"] = np.max(r)
print(table_r.round(2))
