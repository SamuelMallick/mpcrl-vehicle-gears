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
filename = (
    "results/eval_platoon_seed_10/eval_miqp/platoon_miqp_mpc_N_15_c_eval_seed_10.pkl"
)
fig_name_abs = "trajectories_platoon_miqp"
fig_name_err = "trajectories_platoon_err_miqp"

# Save settings
save_png = True
save_pgf = True
save_tikz = False

# Plot settings
t_end_abs = 1000  # time steps for plotting abs values
t_end_err = 1000  # time steps for plotting errors
fig_size_x = 9  # cm
fig_size_y = 5  # cm


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
t_abs = np.arange(t_end_abs)  # time steps for plotting
t_err = np.arange(t_end_err)

# Compute relative errors (lead vehicle from ref, other vehicles from preceding one)
# The +25 compensates for the safety distance between vehicles
x_err = np.zeros_like(x)
x_err[0:t_end_err, 0, 0] = x[0:t_end_err, 0, 0] - x_ref[0:t_end_err, 0]
x_err[0:t_end_err, 1, 0] = x[0:t_end_err, 1, 0] - x_ref[0:t_end_err, 1]
for i in range(1, 5):
    x_err[0:t_end_err, 0, i] = x[0:t_end_err, 0, i] - x[0:t_end_err, 0, i - 1] + 25
    x_err[0:t_end_err, 1, i] = x[0:t_end_err, 1, i] - x[0:t_end_err, 1, i - 1]

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
fig_abs, ax_abs = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(fig_size_x, fig_size_y),
)
ax_abs = ax_abs.flatten()

# Plot trajectories
linewidth = 0.7
n_agents = x.shape[2]
for i in range(n_agents):
    ax_abs[0].plot(t_abs, x[0:t_end_abs, 0, i], linewidth=linewidth)
    ax_abs[1].plot(t_abs, x[0:t_end_abs, 1, i], linewidth=linewidth)
ax_abs[0].plot(
    t_abs, x_ref[0:t_end_abs, 0], linewidth=linewidth, linestyle="--", color="darkred"
)
ax_abs[1].plot(
    t_abs, x_ref[0:t_end_abs, 1], linewidth=linewidth, linestyle="--", color="darkred"
)

# Add legend
ax_abs[0].legend(
    [f"Vehicle {i+1}" for i in range(n_agents)] + ["Reference"],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.6),
    fontsize=8,
    frameon=False,
    fancybox=True,
    ncol=3,
)

# Add grid, limits, ticks, and labels
for i in [0, 1]:
    ax_abs[i].set_axisbelow(True)
    ax_abs[i].grid(True, which="major", linestyle="-", linewidth=0.6, alpha=1)
    ax_abs[i].grid(
        True, which="minor", axis="both", linestyle=":", linewidth=0.5, alpha=0.7
    )
    ax_abs[i].minorticks_on()
    ax_abs[i].set_zorder(1)

ax_abs[0].set_xticks(list(np.linspace(0, 200, 5, dtype=int)))
ax_abs[1].set_xticks(ax_abs[0].get_xticks())
ax_abs[0].set_ylim([-200, 2700])
ax_abs[0].set_yticks([0, 1000, 2000])
ax_abs[0].set_yticklabels([0, 1, 2])
ax_abs[0].text(
    -0.025,
    1,
    r"$\times 10^3$",
    transform=ax_abs[0].transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
)
ax_abs[1].set_ylim([0, 35])
ax_abs[1].set_yticks([0, 10, 20, 30])
ax_abs[1].set_xlabel("$t$ [s]")
ax_abs[0].set_ylabel("$x^{[1]}_i$ [m]")
ax_abs[1].set_ylabel("$x^{[2]}_i$ [m/s]")
fig_abs.align_ylabels(ax_abs)


# Error plot
fig_err, ax_err = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(fig_size_x, fig_size_y),
)
ax_err = ax_err.flatten()

# Plot trajectories
linewidth = 0.7
n_agents = x.shape[2]
for i in range(n_agents):
    ax_err[0].plot(t_err, x_err[0:t_end_err, 0, i], linewidth=linewidth)
    ax_err[1].plot(t_err, x_err[0:t_end_err, 1, i], linewidth=linewidth)
ax_err[0].plot(
    np.zeros(t_end_err), linewidth=linewidth, linestyle="--", color="darkred"
)
ax_err[1].plot(
    np.zeros(t_end_err), linewidth=linewidth, linestyle="--", color="darkred"
)

# Add legend
ax_err[0].legend(
    [f"Vehicle {i+1}" for i in range(n_agents)] + ["Reference"],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.6),
    fontsize=8,
    frameon=False,
    fancybox=True,
    ncol=3,
)

# Add grid, limits, ticks, and labels
for i in [0, 1]:
    ax_err[i].set_axisbelow(True)
    ax_err[i].grid(True, which="major", linestyle="-", linewidth=0.6, alpha=1)
    ax_err[i].grid(
        True, which="minor", axis="both", linestyle=":", linewidth=0.5, alpha=0.7
    )
    ax_err[i].minorticks_on()
    ax_err[i].set_zorder(1)

if t_end_err == 200:
    ax_err[0].set_xticks(list(np.linspace(0, t_end_err, 5, dtype=int)))
elif t_end_err == 500:
    ax_err[0].set_xticks(list(np.linspace(0, t_end_err, 6, dtype=int)))
ax_err[1].set_xticks(ax_err[0].get_xticks())
ax_err[0].set_ylim([-20, 20])
ax_err[0].set_yticks([-20, 0, 20])
ax_err[1].set_ylim([-7.7, 7])
ax_err[1].set_yticks([-5, 0, 5])
ax_err[1].set_xlabel("$t$ [s]")
ax_err[0].set_ylabel("$\\Delta x^{[1]}_i$ [m]")
ax_err[1].set_ylabel("$\\Delta x^{[2]}_i$ [m/s]")
fig_err.align_ylabels(ax_err)

##### Save Figures #####################################################################
if save_png:
    print("Saving png...")
    fig_abs.savefig(f"plots/{fig_name_abs}.png", dpi=300, bbox_inches="tight")
    fig_err.savefig(f"plots/{fig_name_err}.png", dpi=300, bbox_inches="tight")

if save_tikz:
    from utils.tikz import save2tikz  # import tikzplotlib only if supported

    print("Saving tikz...")
    save2tikz(fig_abs, name=f"plots/{fig_name_abs}.tex")
    save2tikz(fig_err, name=f"plots/{fig_name_err}.tex")

if save_pgf:
    mpl.use("pgf")
    print("Saving pgf...")
    fig_abs.savefig(f"plots/{fig_name_abs}.pgf", bbox_inches="tight")
    fig_err.savefig(f"plots/{fig_name_err}.pgf", bbox_inches="tight")
