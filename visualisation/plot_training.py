import os
import pickle
import re
import sys
from itertools import chain

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from packaging import version
from scipy.ndimage import uniform_filter1d

sys.path.append(os.getcwd())
from utils.plot_fcns import cm2inch

# Save settings
final_version = True  # Set to True for final version, False for faster version
save_png = True
save_pgf = True
save_tikz = True

# Plot settings
train_stage = "c4"  # {c1, c2, c3, c4}
fig_size_x = 9.0  # cm
fig_size_y = 6.0  # cm
show_individual_lines = False
show_legend = False

if train_stage == "c3":
    avg_skip = 10000
    avg_window = 10000
elif train_stage == "c4":
    avg_skip = 1000
    avg_window = 1000

########################################################################################

# Select files
root_folder = f"results/train_{train_stage}/"
file_names = []


# Helper function to extract step number from filename
def get_step(filename):
    match = re.search(r"data_step_(\d+)", filename)
    return int(match.group(1)) if match else -1


# Get all the files in the train folder
print(f"Searching for files in {root_folder}...")

# Get all subfolders in the root folder
list_subfolders = os.listdir(root_folder)
print(f"Found {len(list_subfolders)} files/folders in {root_folder}")

# Search for .pkl files in each subfolder
for name in list_subfolders:
    subfolder = os.path.join(root_folder, name)
    if os.path.isdir(subfolder):
        pkl_files = [f for f in os.listdir(subfolder) if f.endswith(".pkl")]
        if not pkl_files:
            continue  # Skip if no pkl files found
        elif len(pkl_files) == 1:
            pkl_file = pkl_files[0]
        else:
            pkl_file = max(pkl_files, key=get_step)
        file_names.append(os.path.join(subfolder, pkl_file))

L = []
L_t = []
L_f = []
kappa = []

# Load data from files
print("Loading data from files...")
for file_name in file_names:
    print(f"\tLoading {file_name}...")
    with open(file_name, "rb") as f:
        data = pickle.load(f)

    # Flatten the data and compute the necessary arrays
    cost = np.array(list(chain.from_iterable(data["cost"])))
    fuel = np.array(list(chain.from_iterable(data["fuel"])))
    R = np.array(list(chain.from_iterable(data["R"])))
    tracking = R - fuel  # Tracking error (not saved directly but easy to recover)

    # Append the data to the lists
    L.append(cost)
    L_t.append(tracking)
    L_f.append(fuel)

    # Assemble kappa vector
    k = np.array([])  # default empty
    if train_stage in ["c1", "c3"]:
        k = np.array(list(chain.from_iterable(data["infeasible"]))).astype(float)
    elif train_stage in ["c2", "c4"]:
        k = 1 - np.array(list(chain.from_iterable(data["heuristic"]))).astype(float)
        k[0:10000] = 0  # Ensure that kappa starts from 0
    kappa.append(k)

# Assemble data into a list
data = [L, L_t, L_f, kappa]

# Compute moving average
print("Computing moving average...")
if final_version is True:
    data_avg = [
        np.array(
            [
                np.convolve(seq, np.ones(avg_window) / avg_window, mode="valid")
                for seq in d
            ]
        )[:, ::avg_skip]
        for d in data
    ]
else:
    # Use scipy for faster computation of the moving average
    data_avg = [
        np.array(
            [
                uniform_filter1d(seq, size=avg_window, mode="nearest")[
                    avg_window - 1 : -avg_window + 1
                ]
                for seq in d
            ]
        )[:, ::avg_skip]
        for d in data
    ]

# Create DataFrames for plotting
print("Creating DataFrames for plotting...")
data_df = [pd.DataFrame(d.T, columns=file_names) for d in data_avg]
for d in data_df:
    d["x"] = np.arange(len(d))
data_df_long = [d.melt(id_vars="x", var_name="seed", value_name="L") for d in data_df]

# Cut length for c4
if train_stage == "c4":
    data_df_long = [d[d["x"] <= 1000] for d in data_df_long]

# Plot results #########################################################################

print("Generating plots...")

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

fig_size_x = cm2inch(fig_size_x)
fig_size_y = cm2inch(fig_size_y)
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(fig_size_x, fig_size_y))
ax = ax.flatten()

if show_individual_lines:
    p0 = sns.lineplot(
        data=data_df_long[0],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[0],
        hue="seed",
    )

    p1 = sns.lineplot(
        data=data_df_long[1],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[1],
        hue="seed",
    )

    p2 = sns.lineplot(
        data=data_df_long[2],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[2],
        hue="seed",
    )

    p3 = sns.lineplot(
        data=data_df_long[3],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[3],
        hue="seed",
    )

    if show_legend is False:
        ax[0].get_legend().remove()
        ax[1].get_legend().remove()
        ax[2].get_legend().remove()
        ax[3].get_legend().remove()


else:
    p0 = sns.lineplot(
        data=data_df_long[0],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[0],
    )

    p1 = sns.lineplot(
        data=data_df_long[1],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[1],
    )

    p2 = sns.lineplot(
        data=data_df_long[2],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[2],
    )

    p3 = sns.lineplot(
        data=data_df_long[3],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[3],
    )

# Set grid
for i in ax:
    i.set_axisbelow(True)  # Set grid below the plot elements
    i.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=1)
    i.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)
    i.minorticks_on()
    i.set_zorder(1)

# Raise plots zorder
p0.set_zorder(10)
p1.set_zorder(10)
p2.set_zorder(10)
p3.set_zorder(10)

# Set labels and ticks
label_L = "$L$"
label_kappa = "$\\kappa$"
if train_stage in ["c1", "c3"]:
    ax[3].set_xticks(np.array([0, 100, 200, 300, 400, 500]))
    ax[3].set_ylim([-0.1, 0.5])
    ax[3].set_yticks([0, 0.2, 0.4])
    label_L = "$L_1$"
    label_kappa = "$\\kappa_1$"
elif train_stage in ["c2", "c4"]:
    ax[3].set_xticks(np.array([0, 250, 500, 750, 1000]))
    ax[3].set_ylim([-0.1, 1.1])
    ax[3].set_yticks([0, 0.5, 1])
    label_kappa = "$\\kappa_2$"
    label_L = "$L_2$"
formatter: FuncFormatter = None  # initialize
if train_stage == "c3":
    formatter = FuncFormatter(lambda x_val, _: f"{int(x_val * 10)}")
elif train_stage == "c4":
    formatter = FuncFormatter(lambda x_val, _: f"{int(x_val)}")
ax[3].xaxis.set_major_formatter(formatter)
ax[3].set_xlabel("Training step $k$")
ax[3].text(
    1.01,
    -0.75,
    r"$\times 10^3$",
    transform=ax[3].transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
)
ax[0].set_ylabel(label_L)
ax[1].set_ylabel("$J_\\mathrm{t}$")
ax[2].set_ylabel("$J_\\mathrm{f}$")
ax[3].set_ylabel(label_kappa)
fig.align_ylabels(ax)

# Save figures
fig_name = ""
if train_stage in ["c1", "c3"]:
    fig_name = "train_stage1"
elif train_stage in ["c2", "c4"]:
    fig_name = "train_stage2"

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
