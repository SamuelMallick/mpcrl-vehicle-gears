"""
Generate the comparison plot for MIQP-tl and RL-MPC with different horizons.
"""

import os
import pickle
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from packaging import version

sys.path.append(os.getcwd())
from utils.plot_fcns import cm2inch

##### Plot settings ####################################################################

# Save settings
show_figure = False
save_png = True
save_pgf = True
save_tikz = False

# Plot settings
fig_size_x = 8  # cm
fig_size_y = 3.5  # cm
show_legend = True

# Plot parameters
labels_font_size = 10
tick_labels_font_size = 8

# Select experiments to plot
eval_list = [
    ["eval_minlp_N15", "MINLP", 15],  # baseline
    ["eval_miqp_N15", "GUROBI", 15],
    ["eval_miqp_N15_cplex", "CPLEX", 15],
    ["eval_l_mpc_c4_seed4_N15", "RL-MPC", 15],
    ["eval_miqp_N20", "GUROBI", 20],
    ["eval_miqp_N20_cplex", "CPLEX", 20],
    ["eval_l_mpc_c4_seed4_N20", "RL-MPC", 20],
    ["eval_miqp_N25", "GUROBI", 25],
    ["eval_miqp_N25_cplex", "CPLEX", 25],
    ["eval_l_mpc_c4_seed4_N25", "RL-MPC", 25],
    ["eval_miqp_N30", "GUROBI", 30],
    ["eval_miqp_N20_cplex", "CPLEX", 30],  # TODO: update data when available
    ["eval_l_mpc_c4_seed4_N30", "RL-MPC", 30],
    ["eval_miqp_N35", "GUROBI", 35],
    ["eval_miqp_N20_cplex", "CPLEX", 35],  # TODO: update data when available
    ["eval_l_mpc_c4_seed4_N35", "RL-MPC", 35],
]
##### Preprocess data ##################################################################

# Initialize data containers
df = pd.DataFrame(columns=["Policy", "Variable", "Value"])
horizon_list = []
experiments_list = list(range(1001, 1025 + 1))  # Experiments to consider
# Collect data
for eval_name, controller, horizon in eval_list:

    # Reset data containers
    reward: list[np.ndarray] = []

    # Locate data files for current evaluation
    pkl_files = os.listdir(f"results/eval_platoon/{eval_name}")
    print(f"Found {len(pkl_files)} files in results/eval_platoon/{eval_name}")

    # Extract all data from .pkl files
    for _, file in enumerate(pkl_files):
        if file.endswith(".pkl"):

            if int(file[-8:-4]) not in experiments_list:
                continue

            with open(f"results/eval_platoon/{eval_name}/{file}", "rb") as f:
                data = pickle.load(f)
                reward.append(data["R"][0])

        else:
            pass  # Skip non-pkl files

    # Compute cumulative episode reward
    # Divide by 1000 for ease of plot (e.g., 1000 instead of 1000000)
    reward = np.array([np.sum(r) / 1000 for r in reward])

    # Build temporary dataframes
    # NOTE: dummy 10e-10 * np.ones(1) ensures that violin hue split works properly
    n_r = len(reward)
    df_temp = pd.DataFrame(
        {
            "Group": [controller] * n_r,
            "Type": [str(horizon)] * n_r,
            "Value": reward,
        }
    )

    # Append data to the data structures
    if df.empty:
        df = df_temp
    else:
        df = pd.concat([df, df_temp])

    # Update horizon list
    if horizon not in horizon_list:
        horizon_list.append(horizon)

# Compute relative performance
J_baseline = df[df["Group"] == "MINLP"]["Value"].values

# Update dataframe with relative performance
for eval_name, controller, horizon in eval_list:
    mask = (df["Group"] == controller) & (df["Type"] == str(horizon))
    J_policy = df[mask]["Value"].values
    relative_J = (J_policy - J_baseline) / J_baseline * 100
    df.loc[mask, "Value"] = (J_policy - J_baseline) / J_baseline * 100

# Calculate the average reward for each evaluation
avg_reward_rlmpc = []
avg_reward_gurobi = []
avg_reward_cplex = []
for h in horizon_list:
    avg_reward_rlmpc.append(
        np.mean(df[(df["Group"] == "RL-MPC") & (df["Type"] == str(h))]["Value"])
    )
    avg_reward_gurobi.append(
        np.mean(df[(df["Group"] == "GUROBI") & (df["Type"] == str(h))]["Value"])
    )
    if h in [15, 20, 25]:
        avg_reward_cplex.append(
            np.mean(df[(df["Group"] == "CPLEX") & (df["Type"] == str(h))]["Value"])
        )
    else:
        avg_reward_cplex.append(1e5)  # Placeholder for missing data

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

# Set plot colors
color_1 = "#005b8f"
color_2 = "#8f0000"
color_3 = "#8f6900"

# Initialize figure
print("Generating figure...")
fig_size_x = cm2inch(fig_size_x)
fig_size_y = cm2inch(fig_size_y)
fig, ax = plt.subplots(figsize=(fig_size_x, fig_size_y))
fig.tight_layout()  # Avoid plt.tight_layout() as it messes with the multiple axes

# Set reward labels and limits
ax.set_xlim(14, 36)
ax.set_yscale("symlog", linthresh=2, linscale=1)  # CHANGE THESE SETTINGS IF NEEDED
ax.set_ylim(-1.1, 2000)
ax.minorticks_on()
ax.set_yticks([-1, 0, 1, 10, 100, 1000])
ax.set_yticks([-0.5, 0.5, 2, 3, 5, 50, 500], minor=True)
ax.set_yticklabels(["-1", "0", "1", "$10^1$", "$10^2$", "$10^3$"])
ax.set_xlabel("$N$")
ax.set_ylabel("$\\Delta J$ [\\%]")  # Relative performance drop
ax.set_axisbelow(True)  # Set grid below the plot elements
ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=1)
ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)
ax.set_zorder(1)

# Plot RLMPC2
h_rlmpc = ax.plot(
    horizon_list,
    avg_reward_rlmpc,
    marker=".",
    markersize=8,
    color=color_1,
    linestyle="None",
)

# PLOT GUROBI MIQP-tl
h_gurobi = ax.plot(
    horizon_list,
    avg_reward_gurobi,
    marker="D",
    markersize=3,
    color=color_2,
    linestyle="None",
)

# PLOT CPLEX MIQP-tl
h_cplex = ax.plot(
    horizon_list,
    avg_reward_cplex,
    marker="v",
    markersize=3,
    color=color_3,
    linestyle="None",
)

# Add legend
if show_legend is True:
    handles = [h_rlmpc[0], h_gurobi[0], h_cplex[0]]
    labels = ["LC-2", "GUROBI", "CPLEX"]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        fontsize=labels_font_size,
        framealpha=None,
        edgecolor="white",
        frameon=True,
        mode=None,
        ncols=3,
        columnspacing=0.7,
        borderpad=0,
        labelspacing=0.2,
        handletextpad=-0.3,
    )

if show_figure:
    plt.show()

# Save figures
if save_png:
    print("Saving png...")
    fig.savefig("plots/horizon_comparison.png", dpi=300, bbox_inches="tight")

if save_tikz:
    from utils.tikz import save2tikz  # import tikzplotlib only if supported

    print("Saving tikz...")
    save2tikz(plt.gcf(), name="plots/horizon_comparison.tex")

if save_pgf:
    mpl.use("pgf")  # This line must be after the execution of save2tikz (?)
    print("Saving pgf...")
    fig.savefig("plots/horizon_comparison.pgf", bbox_inches="tight")
