"""
Generate the comparison plot for MIQP-tl and RL-MPC25 with different horizons.
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
save_png = True
save_pgf = True
save_tikz = True

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
    # ["eval_miqp_N10", "MIQP", 10],
    # ["eval_l_mpc_c4_seed4_N10", "RL-MPC", 10],
    ["eval_miqp_N15", "MIQP", 15],
    ["eval_l_mpc_c4_seed4_N15", "RL-MPC", 15],
    ["eval_miqp_N20", "MIQP", 20],
    ["eval_l_mpc_c4_seed4_N20", "RL-MPC", 20],
    ["eval_miqp_N25", "MIQP", 25],
    ["eval_l_mpc_c4_seed4_N25", "RL-MPC", 25],
    ["eval_miqp_N30", "MIQP", 30],
    ["eval_l_mpc_c4_seed4_N30", "RL-MPC", 30],
    ["eval_miqp_N35", "MIQP", 35],
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
avg_reward_miqp = []
for h in horizon_list:
    avg_reward_rlmpc.append(
        np.mean(df[(df["Group"] == "RL-MPC") & (df["Type"] == str(h))]["Value"])
    )
    avg_reward_miqp.append(
        np.mean(df[(df["Group"] == "MIQP") & (df["Type"] == str(h))]["Value"])
    )
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
color_1 = "cadetblue"
color_2 = "darkred"

# Initialize figure
print("Generating figure...")
fig_size_x = cm2inch(fig_size_x)
fig_size_y = cm2inch(fig_size_y)
fig, ax = plt.subplots(figsize=(fig_size_x, fig_size_y))
fig.tight_layout()  # Avoid plt.tight_layout() as it messes with the multiple axes

# Set reward labels and limits
ax.set_ylim(13, 37)
ax.set_ylim(-1, 5)
ax.set_ylabel("$\\Delta J$ [\\%]")  # Relative performance drop
ax.set_xlabel("$N$")
ax.set_axisbelow(True)  # Set grid below the plot elements
ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=1)
ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)
ax.minorticks_on()
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

# PLOT MIQP-tl
h_miqp = ax.plot(
    horizon_list,
    avg_reward_miqp,
    marker="D",
    markersize=3,
    color=color_2,
    linestyle="None",
)

# Add legend
if show_legend is True:
    handles = [h_rlmpc[0], h_miqp[0]]
    labels = ["LC-2", "MIQP-tl"]
    ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
    )

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
