from __future__ import annotations

try:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.image import AxesImage
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    HAS_MATPLOTLIB = True
except ImportError:
    pass


if HAS_MATPLOTLIB:

    def despine(ax: Axes) -> None:
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    def create_colorbar(
        im: AxesImage,
        ax: Axes,
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
        ax : :obj:`Axes <matplotlib.axes.Axes>`
            Axis which to split.
        size : :obj:`float`, optional
            Size of split, by default 0.05. Effectively sets the size of the colorbar.
        pad : :obj:`float`, optional`
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
        divider = make_axes_locatable(ax)
        cax: Axes = divider.append_axes("right", size=f"{size:%}", pad=pad)
        fig = ax.get_figure()
        cb = fig.colorbar(im, cax=cax, orientation=orientation)
        return cax, cb
