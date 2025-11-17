import tikzplotlib as tikz_lib  # recommended to install tikzplotlib-patched
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

# Monkey patching to fix some issues with tikzplotlib
# See https://github.com/nschloe/tikzplotlib/issues/567
Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)


# TODO: consider exporting the data to a separate .dat file instead and importing it
# in the tikz file to have a more clean code.
# TODO: consider switching to https://github.com/ErwindeGelder/matplot2tikz


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

    # fix tikzplotlib issues
    tikzplotlib_fix_ncols(fig)

    # save figure
    tikz_lib.save(
        name,
        figure=fig,
        extra_axis_parameters={r"tick scale binop=\times"},
    )


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks
    tikzplotlib. See: https://stackoverflow.com/questions/75900239/attributeerror-occurs-with-tikzplotlib-when-legend-is-plotted  # pylint: disable=line-too-long
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols  # pylint: disable=protected-access
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
