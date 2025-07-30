import os
import pickle
import sys
import re
from packaging import version

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from utils.plot_fcns import cm2inch

# Save settings
save_png = True
save_pgf = True
save_tikz = False

# Plot settings
train_stage = "c3"  # {c1, c2, c3, c4}
skip = 10000
average_interval = 10000
show_individual_lines = False
show_legend = False

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
print(f"Found {len(list_subfolders)} subfolders in {root_folder}")

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
    with open(file_name, "rb") as f:
        data = pickle.load(f)

    cost = data["cost"]
    fuel = data["fuel"]
    R = data["R"]
    tracking = [r - f for sub_r, sub_f in zip(R, fuel) for r, f in zip(sub_r, sub_f)]
    if "infeasible" in data:
        infeasible = data["infeasible"]
    if "heuristic" in data:
        heuristic = data["heuristic"]

    L.append([l for sub_l in cost for l in sub_l])
    L_t.append(tracking)
    L_f.append([f for sub_f in fuel for f in sub_f])
    kappa.append([i for sub_i in infeasible for i in sub_i])
    # kappa.append(heuristic)

# Compute average
print("Computing average...")
data = [L, L_t, L_f, kappa]
data_avg = [
    np.array(
        [
            np.convolve(l, np.ones(average_interval) / average_interval, mode="valid")
            for l in d
        ]
    )[:, ::skip]
    for d in data
]
data_df = [pd.DataFrame(d.T, columns=file_names) for d in data_avg]
for d in data_df:
    d["x"] = np.arange(len(d))
data_df_long = [d.melt(id_vars="x", var_name="seed", value_name="L") for d in data_df]

# Plot results #########################################################################

print("Generating plots...")

if version.parse(mpl.__version__) <= version.parse("3.7"):
    if save_pgf is True:
        save_pgf = True
    if save_tikz is True:
        save_tikz = True
        from utils.tikz import save2tikz  # import tikzplotlib only if supported
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
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 8,
        "pgf.rcfonts": False,
        "pgf.preamble": "\\usepackage[T1]{fontenc}",  # extra preamble for LaTeX
    }
)

fig_size_x = cm2inch(8)
fig_size_y = cm2inch(10)
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(fig_size_x, fig_size_y))

if show_individual_lines:
    sns.lineplot(
        data=data_df_long[0],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[0],
        hue="seed",
    )

    sns.lineplot(
        data=data_df_long[1],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[1],
        hue="seed",
    )

    sns.lineplot(
        data=data_df_long[2],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[2],
        hue="seed",
    )

    sns.lineplot(
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
    sns.lineplot(
        data=data_df_long[0],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[0],
    )

    sns.lineplot(
        data=data_df_long[1],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[1],
    )

    sns.lineplot(
        data=data_df_long[2],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[2],
    )

    sns.lineplot(
        data=data_df_long[3],
        x="x",
        y="L",
        errorbar="sd",
        ax=ax[3],
    )

# Set axis labels
ax[0].set_ylabel("L")
ax[1].set_ylabel("L_t")
ax[2].set_ylabel("L_f")
ax[3].set_ylabel("$\\kappa$")

# Save figures
if save_png:
    print("Saving png...")
    fig.savefig(f"results/plots/{train_stage}.png", dpi=300, bbox_inches="tight")

if save_tikz:
    print("Saving tikz...")
    save2tikz(plt.gcf(), name=f"results/plots/{train_stage}.tex")

if save_pgf:
    mpl.use("pgf")
    print("Saving pgf...")
    fig.savefig(f"results/plots/{train_stage}.pgf")
