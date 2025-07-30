import matplotlib as mpl

import tikzplotlib as tikz_lib  # recommended to install tikzplotlib-patched

from matplotlib.figure import Figure

# monkey patching to fix some issues with tikzplotlib
# mpl.lines.Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
# mpl.lines.Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
# mpl.legend.Legend._ncol = property(lambda self: self._ncols)

# TODO: consider exporting the data to a separate .dat file instead and importing it
# in the tikz file to have a more clean code.


def save2tikz(fig: Figure, name: str = "") -> None:
    """Saves the figure to a tikz file (`.tex` extension).
    See https://pypi.org/project/tikzplotlib/ for more details (note: no longer
    mantained). Recommended installation to avoid compatibility issues:
    https://pypi.org/project/tikzplotlib-patched/

    Parameters
    ----------
    figs : matplotlib Figures
        One or more matplotlib figures to be converted to tikz files. These
        files will be named based on the number of the corresponding figure.

    name : str, optional
        The name of the tikz file to be saved. If not provided, the name will
        be generated based on the figure number.
    """

    # generate figure name
    if name == "":
        name = f"figure_{fig.number}.tex"
    elif not name.endswith(".tex"):
        name = f"{name}.tex"
    else:
        pass  # name is already a valid tikz file name

    # save figure
    tikz_lib.save(
        name,
        figure=fig,
        extra_axis_parameters={r"tick scale binop=\times"},
    )
