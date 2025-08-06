import os
import sys
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from packaging import version

sys.path.append(os.getcwd())
from utils.plot_fcns import cm2inch

##### Plot settings ####################################################################

# Select experiment to plot
filename = "results/eval_single/eval_l_mpc/c4_seed4/l_mpc_N_15_c_eval_seed_1001.pkl"
fig_name = "c4_seed4_eval_seed_1001"

# Save settings
save_png = True
save_pgf = True
save_tikz = False

# Plot settings
fig_size_x = 20.0  # cm
fig_size_y = 10.0  # cm


##### Generate Plot ####################################################################

# Load data
print("Loading data...")
with open(filename, "rb") as f:
    data = pickle.load(f)

# Extract data
X = data["X"]
x_ref = data["x_ref"]
U = data["U"]
R = data["R"]
# TODO: complete data extraction --> how to get kappa?

# Set save settings
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
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pgf.rcfonts": False,
        "pgf.preamble": "\\usepackage[T1]{fontenc}",  # extra preamble for LaTeX
    }
)

# Plot data
print("Generating plots...")
fig_size_x = cm2inch(fig_size_x)
fig_size_y = cm2inch(fig_size_y)
fig, ax = plt.subplots(
    4,
    2,
    sharex=True,
    sharey=True,
    figsize=(fig_size_x, fig_size_y),
)
ax = ax.flatten()


for i in range(X.shape[2]):
    if i == 0:
        ax[0].plot(X[:-1, 0, i] - x_ref[:, 0, 0])
    else:
        ax[0].plot(X[:-1, 0, i] - X[:-1, 0, i - 1])
# ax[0].hlines(-10, 0, X.shape[0], color="red", linestyle="--")
ax[0].set_ylabel("d_e (m)")
# ax[1].plot(X[:, 0], label="_nolegend_")
# ax[1].plot(x_ref[:, 0], "--")
# ax[1].legend(["ref"])
ax[1].plot(X[:, 1], label="_nolegend_")
ax[1].plot(x_ref[:, 1], "--")
ax[1].legend(["ref"])
ax[1].set_ylabel("v (m/s)")
ax[2].plot(U[:, 2])
ax[2].set_ylabel("gear")
# ax[3].set_xticks([i for i in range(len(U))])
ax[2].set_yticks([i for i in range(6)])
# ax[3].plot(np.cumsum(fuel))
# ax[3].set_ylabel("Fuel (L)")
# # ax[4].plot(np.cumsum(R))
# ax[4].plot(R)
# ax[4].set_ylabel("Reward")

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
