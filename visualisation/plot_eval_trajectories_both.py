import os
import pickle
import sys
from itertools import chain

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from packaging import version

sys.path.append(os.getcwd())
from utils.plot_fcns import cm2inch

##### Plot settings ####################################################################

# Plot trajectories of a platoon evaluation.
# Select experiment to plot
filename = "results/eval_platoon/eval_miqp/platoon_miqp_N_15_c_eval_seed_1003.pkl"
fig_name = "trajectories_platoon_miqp_1003"

# Save settings
save_png = True
save_pgf = False
save_tikz = False

# Plot settings
t_end = 1000  # time steps for plotting abs values
fig_size_x = 9  # cm
fig_size_y = 10  # cm


##### Generate Plot ####################################################################

# Update save settings based on matplotlib version
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

# Load data
print("Loading data...")
with open(filename, "rb") as f:
    data = pickle.load(f)
x = np.squeeze(np.array(list(chain.from_iterable(data["X"]))))
x_ref = np.squeeze(np.array(list(chain.from_iterable(data["x_ref"]))))
t = np.arange(t_end)  # time steps for plotting


# Compute relative errors (lead vehicle from ref, other vehicles from preceding one)
# The +25 compensates for the safety distance between vehicles
x_err = np.zeros_like(x)
x_err[0:t_end, 0, 0] = x[0:t_end, 0, 0] - x_ref[0:t_end, 0]
x_err[0:t_end, 1, 0] = x[0:t_end, 1, 0] - x_ref[0:t_end, 1]
for i in range(1, 5):
    x_err[0:t_end, 0, i] = x[0:t_end, 0, i] - x[0:t_end, 0, i - 1] + 25
    x_err[0:t_end, 1, i] = x[0:t_end, 1, i] - x[0:t_end, 1, i - 1]

# Set matplotlib parameters
mpl.rcParams.update(
    {
        "pgf.texsystem": "xelatex",  # or any other engine you want to use
        "text.usetex": True,  # use TeX for all texts
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pgf.rcfonts": False,
        "pgf.preamble": "\\usepackage[T1]{fontenc}",  # extra preamble for LaTeX
    }
)

# Initialize plots
print("Generating plots...")
fig_size_x = cm2inch(fig_size_x)
fig_size_y = cm2inch(fig_size_y)

# Absolute values plot
fig, ax = plt.subplots(
    4,
    1,
    sharex=True,
    figsize=(fig_size_x, fig_size_y),
)
ax = ax.flatten()

# Plot trajectories
linewidth = 0.7
n_agents = x.shape[2]
for i in range(n_agents):
    ax[0].plot(t, x[0:t_end, 0, i], linewidth=linewidth)
    ax[1].plot(t, x_err[0:t_end, 0, i], linewidth=linewidth)
    ax[2].plot(t, x[0:t_end, 1, i], linewidth=linewidth)
    ax[3].plot(t, x_err[0:t_end, 1, i], linewidth=linewidth)
ax[0].plot(t, x_ref[0:t_end, 0], linewidth=linewidth, linestyle="--", color="darkred")
ax[1].plot(np.zeros(t_end), linewidth=linewidth, linestyle="--", color="darkred")
ax[2].plot(t, x_ref[0:t_end, 1], linewidth=linewidth, linestyle="--", color="darkred")
ax[3].plot(np.zeros(t_end), linewidth=linewidth, linestyle="--", color="darkred")

# Add legend
ax[0].legend(
    [f"Vehicle {i+1}" for i in range(n_agents)] + ["Reference"],
    loc="upper center",
    bbox_to_anchor=(0.47, 1.75),
    fontsize=8,
    frameon=False,
    fancybox=True,
    ncol=3,
)

# Add grid, limits, ticks, and labels
for i in range(4):
    ax[i].set_axisbelow(True)
    ax[i].grid(True, which="major", linestyle="-", linewidth=0.6, alpha=1)
    ax[i].grid(
        True, which="minor", axis="both", linestyle=":", linewidth=0.5, alpha=0.7
    )
    ax[i].minorticks_on()
    ax[i].set_zorder(1)

ax[0].set_xticks(list(np.linspace(0, 200, 5, dtype=int)))
ax[1].set_xticks(ax[0].get_xticks())
ax[2].set_xticks(ax[0].get_xticks())
ax[3].set_xticks(ax[0].get_xticks())
ax[0].set_ylim([-200, 2700])
ax[0].set_yticks([0, 1000, 2000])
ax[0].set_yticklabels([0, 1, 2])
ax[0].set_ylabel("$x^{[1]}_i$ [m]")
ax[0].text(
    0.095,
    1.02,
    r"$\times 10^3$",
    transform=ax[0].transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
)
ax[1].set_ylim([-20, 20])
ax[1].set_yticks([-20, 0, 20])
ax[1].set_ylabel("$\\Delta x^{[1]}_i$ [m]")
ax[2].set_ylim([0, 35])
ax[2].set_yticks([0, 10, 20, 30])
ax[2].set_xlabel("$t$ [s]")
ax[2].set_ylabel("$x^{[2]}_i$ [m/s]")
ax[3].set_ylim([-7.7, 7])
ax[3].set_yticks([-5, 0, 5])
ax[3].set_xlabel("$t$ [s]")
ax[3].set_ylabel("$\\Delta x^{[2]}_i$ [m/s]")
fig.align_ylabels(ax)

##### Save Figures #####################################################################
if save_png:
    print("Saving png...")
    fig.savefig(f"plots/{fig_name}.png", dpi=300, bbox_inches="tight")

if save_tikz:
    from utils.tikz import save2tikz  # import tikzplotlib only if supported

    print("Saving tikz...")
    save2tikz(fig, name=f"plots/{fig_name}.tex")

if save_pgf:
    mpl.use("pgf")
    print("Saving pgf...")
    fig.savefig(f"plots/{fig_name}.pgf", bbox_inches="tight")
