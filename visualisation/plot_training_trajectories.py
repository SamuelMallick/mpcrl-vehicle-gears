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

# Select experiment to plot
filename = "results/train_c3/c3_seed1/data_step_5000000.pkl"
fig_name = "train_traj"

# Save settings
save_png = True
save_pgf = True
save_tikz = False

# Plot settings
fig_size_x = 9  # cm
fig_size_y = 7.5  # cm


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

# Extract data and flatten it
x = np.squeeze(np.array(list(chain.from_iterable(data["X"]))))
x_ref = np.squeeze(np.array(list(chain.from_iterable(data["x_ref"]))))
u = np.squeeze(np.array(list(chain.from_iterable(data["U"]))))
r = np.squeeze(np.array(list(chain.from_iterable(data["R"]))))
kappa = np.squeeze(np.array(list(chain.from_iterable(data["infeasible"]))).astype(int))

# Compute data to plot
t = np.arange(500)  # time steps for plotting
begin_p_err = x[0:500, 0] - x_ref[0:500, 0]
begin_v = x[0:500, 1]
begin_v_ref = x_ref[0:500, 1]
begin_j = u[0:500, 2] + 1
begin_k = kappa[0:500]
end_p_err = x[-501:-1, 0] - x_ref[-501:-1, 0]
end_v = x[-501:-1, 1]
end_v_ref = x_ref[-501:-1, 1]
end_j = u[-501:-1, 2] + 1
end_k = kappa[-501:-1]

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

# Initialize plot
print("Generating plots...")
fig_size_x = cm2inch(fig_size_x)
fig_size_y = cm2inch(fig_size_y)
fig, ax = plt.subplots(
    4,
    2,
    sharex=True,
    figsize=(fig_size_x, fig_size_y),
)
ax = ax.flatten()
plt.subplots_adjust(
    wspace=0.1,  # horizontal space between subplots
    hspace=0.15,  # vertical space between subplots
)

# Plot the data
linewidth = 0.8
linewidth_kappa = 0.7
ax[0].plot(t, begin_p_err, linewidth=linewidth)
ax[2].plot(t, begin_v, linewidth=linewidth, label="_nolegend_")
ax[2].plot(t, begin_v_ref, "--", linewidth=linewidth, color="darkred")
ax[4].plot(t, begin_j, linewidth=linewidth)
ax[6].plot(t, begin_k, linewidth=linewidth_kappa)
ax[1].plot(t, end_p_err, linewidth=linewidth)
ax[3].plot(t, end_v, linewidth=linewidth)
ax[3].plot(t, end_v_ref, "--", linewidth=linewidth, color="darkred")
ax[5].plot(t, end_j, linewidth=linewidth)
ax[7].plot(t, end_k, linewidth=linewidth_kappa)

# Add grid
for i in ax:
    i.set_axisbelow(True)
    i.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=1)
    if i in [ax[4], ax[5]]:
        i.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
    else:
        i.grid(
            True, which="minor", axis="both", linestyle=":", linewidth=0.5, alpha=0.7
        )
    i.minorticks_on()
    i.set_zorder(1)

# Set y limits
ax[6].set_xticks([0, 100, 200, 300, 400, 500])
ax[7].set_xticks([0, 100, 200, 300, 400, 500])
ax[0].set_ylim([-70, 70])
ax[1].set_ylim([-70, 70])
ax[2].set_ylim([-5, 40])
ax[3].set_ylim([-5, 40])
ax[4].set_ylim([0.5, 6.5])
ax[5].set_ylim([0.5, 6.5])
ax[6].set_ylim([-0.2, 1.2])
ax[7].set_ylim([-0.2, 1.2])
ax[1].set_yticklabels([])
ax[2].set_yticks([0, 15, 30])
ax[3].set_yticks([0, 15, 30])
ax[3].set_yticklabels([])
ax[4].set_yticks([1, 2, 3, 4, 5, 6])
ax[5].set_yticks([1, 2, 3, 4, 5, 6])
ax[5].set_yticklabels([])
ax[6].set_yticks([0, 1])
ax[7].set_yticks([0, 1])
ax[7].set_yticklabels([])

# Set titles and labels
ax[6].set_xlabel("$k$ (first 500 steps)")
ax[7].set_xlabel("$k$ (last 500 steps)")
ax[0].set_ylabel("$x^{[1]} - x^{[1]}_\\mathrm{ref}$")
ax[2].set_ylabel("$x^{[2]}$")
ax[4].set_ylabel("$j$")
ax[6].set_ylabel("$\\kappa_1$")
fig.align_ylabels(ax)

# Save figures
if save_png:
    print("Saving png...")
    fig.savefig(f"plots/{fig_name}.png", dpi=300, bbox_inches="tight")

if save_tikz:
    from utils.tikz import save2tikz  # import tikzplotlib only if supported

    print("Saving tikz...")
    save2tikz(plt.gcf(), name=f"plots/{fig_name}.tex")

if save_pgf:
    mpl.use("pgf")
    print("Saving pgf...")
    fig.savefig(f"plots/{fig_name}.pgf", bbox_inches="tight")
