from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

from ..typing import AnyNDArray


def despine(ax: Axes) -> None:
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def create_colorbar(
    im: AxesImage,
    ax: Axes | None = None,
    size: float = 0.05,
    pad: float = 0.1,
    orientation: str = "vertical",
) -> tuple[Axes, Colorbar]:
    r"""Create a colorbar.

    Divides  axis and attaches a colorbar to it.

    Parameters
    ----------
    im : :obj:`AxesImage <matplotlib.image.AxesImage>`
        Image from which the colorbar will be created.
        Commonly the output of :obj:`matplotlib.pyplot.imshow`.
    ax : :obj:`Axes <matplotlib.axes.Axes>`, optional
        Axis which to split. Uses `plt.gca()` if None.
    size : :obj:`float`, optional
        Size of split, by default 0.05. Effectively sets the size of the colorbar.
    pad : :obj:`float`, optional
        Padding between colorbar axis and input axis, by default 0.1.
    orientation : :obj:`str`, optional
        Orientation of the colorbar, by default "vertical".

    Returns
    -------
    Tuple[:obj:`Axes <matplotlib.axes.Axes>`, :obj:`Colorbar <matplotlib.colorbar.Colorbar>`]
        **cax** : Colorbar axis.

        **cb** : Colorbar.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.ticker import MultipleLocator
    >>> from curvelops.plot import create_colorbar
    >>> fig, ax = plt.subplots()
    >>> im = ax.imshow([[0]], vmin=-1, vmax=1, cmap="gray")
    >>> cax, cb = create_colorbar(im, ax)
    >>> cax.yaxis.set_major_locator(MultipleLocator(0.1))
    >>> print(cb.vmin)
    -1.0
    """
    ax_ = plt.gca() if ax is None else ax
    divider = make_axes_locatable(ax_)
    cax: Axes = divider.append_axes("right", size=f"{size:%}", pad=pad)
    fig = ax_.get_figure()
    if fig is None:
        cb = plt.colorbar(im, cax=cax, orientation=orientation)
    else:
        cb = fig.colorbar(im, cax=cax, orientation=orientation)
    return cax, cb


def _create_range(start: float, end: float, n: int) -> AnyNDArray:
    return start + (end - start) * (0.5 + np.arange(n)) / n


def create_axes_grid(
    rows: int,
    cols: int,
    kwargs_figure: dict[str, Any] | None = None,
    kwargs_gridspec: dict[str, Any] | None = None,
    kwargs_subplots: dict[str, Any] | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    r"""Creates a grid of axes.

    Parameters
    ----------
    rows : :obj:`int`
        Number of rows.
    cols : :obj:`int`
        Number of columns.
    kwargs_figure : :obj:`dict[str, Any]`, optional
        Arguments to be passed to :obj:`matplotlib.pyplot.figure`.
    kwargs_gridspec : :obj:`dict[str, Any]`, optional
        Arguments to be passed to :obj:`matplotlib.gridspec.GridSpec`.
    kwargs_subplots : :obj:`dict[str, Any]`, optional
        Arguments to be passed to :obj:`matplotlib.figure.Figure.add_subplot`.


    Returns
    -------
    Tuple[:obj:`Figure <matplotlib.figure.Figure>`, :obj:`NDArray[Axes]`]
        **fig** : Figure.

        **axs** : Array of :obj:`Axes <matplotlib.axes.Axes>` shaped ``(rows, cols)``.

    Examples
    --------
    >>> from curvelops.plot import create_axes_grid
    >>> rows, cols = 2, 3
    >>> fig, axs = create_axes_grid(
    >>>     rows,
    >>>     cols,
    >>>     kwargs_figure=dict(figsize=(8, 8)),
    >>>     kwargs_gridspec=dict(wspace=0.3, hspace=0.3),
    >>> )
    >>> for irow in range(rows):
    >>>     for icol in range(cols):
    >>>         axs[irow][icol].plot(np.cos((2 + irow + icol**2) * np.linspace(0, 1)))
    >>>         axs[irow][icol].set(title=f"Row, Col: ({irow}, {icol})")
    """
    if kwargs_figure is None:
        kwargs_figure = {}
    if kwargs_gridspec is None:
        kwargs_gridspec = {}
    if kwargs_subplots is None:
        kwargs_subplots = {}
    fig = plt.figure(**kwargs_figure)
    grid = fig.add_gridspec(rows, cols, **kwargs_gridspec)
    axs: NDArray[np.object_] = np.empty((rows, cols), dtype=Axes)
    for irow in range(rows):
        for icol in range(cols):
            axs[irow, icol] = fig.add_subplot(grid[irow, icol], **kwargs_subplots)
    return fig, axs


def create_inset_axes_grid(
    ax: Axes,
    rows: int = 1,
    cols: int = 1,
    height: float = 0.5,
    width: float = 0.5,
    squeeze: bool = True,
    kwargs_inset_axes: dict[Any, Any] | None = None,
) -> NDArray[np.object_] | Axes:
    r"""Create a grid of insets.

    The input axis will be overlaid with a grid of insets.
    Numbering of the axes is top to bottom (rows) and
    left to right (cols).

    Parameters
    ----------
    ax : :obj:`Axes <matplotlib.axes.Axes>`
        Input axis.
    rows : :obj:`int`
        Number of rows.
    cols : :obj:`int`
        Number of columns.
    width : :obj:`float`, optional
        Width of each axis, as a percentage of ``cols``, by default 0.5.
    height : :obj:`float`, optional
        Height of each axis, as a percentage of ``rows``, by default 0.5.
    kwargs_inset_axes : :obj:`dict[Any, Any]`, optional
        Arguments to be passed to :obj:`matplotlib.axes.Axes.inset_axes`.

    Returns
    -------
    :obj:`NDArray[Axes]`
        Array of :obj:`Axes <matplotlib.axes.Axes>` shaped ``(rows, cols)``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from curvelops.plot import create_inset_axes_grid
    >>> fig, ax = plt.subplots(figsize=(6, 6))
    >>> ax.imshow([[0]], extent=[-2, 2, 2, -2], vmin=-1, vmax=1, cmap="gray")
    >>> rows, cols = 2, 3
    >>> inset_axes = create_inset_axes_grid(
    >>>     ax,
    >>>     rows,
    >>>     cols,
    >>>     width=0.5,
    >>>     height=0.5,
    >>>     kwargs_inset_axes=dict(projection="polar"),
    >>> )
    >>> nscales = 4
    >>> lw = 0.1
    >>> for irow in range(rows):
    >>>     for icol in range(cols):
    >>>         for iscale in range(1, nscales):
    >>>             inset_axes[irow][icol].bar(
    >>>                 x=0,
    >>>                 height=lw,
    >>>                 width=2 * np.pi,
    >>>                 bottom=((iscale + 1) - 0.5 * lw) / (nscales - 1),
    >>>                 color="r",
    >>>             )
    >>>             inset_axes[irow][icol].set(title=f"Row, Col: ({irow}, {icol})")
    >>>             inset_axes[irow][icol].axis("off")
    """
    if kwargs_inset_axes is None:
        kwargs_inset_axes = {}

    axes: NDArray[np.object_] = np.empty((rows, cols), dtype=Axes)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)

    width *= (xmax - xmin) / cols
    height *= (ymax - ymin) / rows

    for irow, rowpos in enumerate(_create_range(ymin, ymax, rows)):
        for icol, colpos in enumerate(_create_range(xmin, xmax, cols)):
            axes[irow, icol] = ax.inset_axes(
                (colpos - 0.5 * width, rowpos - 0.5 * height, width, height),
                transform=ax.transData,
                **kwargs_inset_axes,
            )
    if squeeze:
        if rows == cols == 1:
            return axes[0, 0]
        return axes.squeeze()
    return axes


def overlay_arrows(
    vectors: AnyNDArray, ax: Axes, arrowprops: dict[str, Any] | None = None
) -> Axes:
    r"""Overlay arrows on an axis.

    Parameters
    ----------
    vectors : :obj:`AnyNDArray`
        Array shaped ``(rows, cols, 2)``, corresponding to a 2D vector field.
    ax : :obj:`Axes <matplotlib.axes.Axes>`
        Axis on which to overlay the arrows.
    arrowprops : :obj:`dict`, optional
        Arrow properties, to be passed to :obj:`matplotlib.pyplot.annotate`.
        By default will be set to ``dict(facecolor="black", shrink=0.05)``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from curvelops.plot import overlay_arrows
    >>> fig, ax = plt.subplots(figsize=(8, 10))
    >>> ax.imshow([[0]], vmin=-1, vmax=1, extent=[0, 1, 1, 0], cmap="gray")
    >>> rows, cols = 3, 4
    >>> kvecs = np.array(
    >>>     [
    >>>         [(1 + x, x * y) for x in (0.5 + np.arange(cols)) / cols]
    >>>         for y in (0.5 + np.arange(rows)) / rows
    >>>     ]
    >>> )
    >>> overlay_arrows(
    >>>     0.05 * kvecs,
    >>>     ax,
    >>>     arrowprops=dict(
    >>>         facecolor="r",
    >>>         shrink=0.05,
    >>>         width=10 / cols,
    >>>         headwidth=10,
    >>>         headlength=10,
    >>>     ),
    >>> )
    """
    rows, cols, _ = vectors.shape

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)

    if arrowprops is None:
        arrowprops = dict(facecolor="black", shrink=0.05)

    for irow, rowpos in enumerate(_create_range(ymin, ymax, rows)):
        for icol, colpos in enumerate(_create_range(xmin, xmax, cols)):
            ax.annotate(
                "",
                xy=(
                    colpos + vectors[irow, icol, 0],
                    rowpos + vectors[irow, icol, 1],
                ),
                xytext=(colpos, rowpos),
                xycoords="data",
                arrowprops=arrowprops,
                annotation_clip=False,
            )
    return ax
