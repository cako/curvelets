from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from typing import Any, Callable, TypeVar

import numpy as np
from numpy.fft import fft2, fftfreq, fftshift
from numpy.typing import NDArray

from ..typing import AnyNDArray, ComplexNDArray, RecursiveListAnyNDArray

T = TypeVar("T")


def deepflatten(lst: list[Any]) -> Iterable[Any]:
    """Flatten a list using a deque."""
    q: Any = deque()
    for item in lst:
        if isinstance(item, list):
            q.extendleft(reversed(item))
        else:
            q.appendleft(item)
        while q:
            elem = q.popleft()
            if isinstance(elem, list):
                q.extendleft(reversed(elem))
            else:
                yield elem


def array_split_nd(ary: AnyNDArray, *args: int) -> RecursiveListAnyNDArray:
    r"""Split an array into multiple sub-arrays recursively, possibly unevenly.

    See Also
    --------
    :obj:`numpy.array_split` : Split an array into multiple sub-arrays.

    :obj:`split_nd`: Evenly split an array into multiple sub-arrays recursively.

    Parameters
    ----------
    ary : :obj:`AnyNDArray`
        Input array.

    args : :obj:`int`, optional
        Number of splits for each axis of `ary`.
        Axis 0 will be split into `args[0]` subarrays, axis 1 will be
        into `args[1]` subarrays, etc. An axis of length
        `l = ary.shape[axis]` that should be split into `n = args[axis]`
        sections, will return `l % n` sub-arrays of size `l//n + 1`
        and the rest of size `l//n`.

    Returns
    -------
    :obj:`RecursiveListAnyNDArray <curvelops.typing.RecursiveListAnyNDArray>`
        Recursive lists of lists of :obj:`AnyNDArray`.
        The number of recursions is equivalent to the number arguments in args.

    Examples
    --------
    >>> from curvelops.utils import array_split_nd
    >>> ary = np.outer(1 + np.arange(2), 2 + np.arange(3))
    array([[2, 3, 4],
           [4, 6, 8]])
    >>> array_split_nd(ary, 2, 3)
    [[array([[2]]), array([[3]]), array([[4]])],
     [array([[4]]), array([[6]]), array([[8]])]]

    >>> from curvelops.utils import array_split_nd
    >>> ary = np.outer(np.arange(3), np.arange(5))
    >>> array_split_nd(ary, 2, 3)
    [[array([[0, 0],
             [0, 1]]),
      array([[0, 0],
             [2, 3]]),
      array([[0],
             [4]])],
     [array([[0, 2]]), array([[4, 6]]), array([[8]])]]
    """
    axis = ary.ndim - len(args)
    split = np.array_split(ary, args[0], axis=axis)
    if len(args) == 1:
        return split
    return [array_split_nd(s, *args[1:]) for s in split]


def split_nd(ary: AnyNDArray, *args: int) -> RecursiveListAnyNDArray:
    r"""Evenly split an array into multiple sub-arrays recursively.

    See Also
    --------
    :obj:`numpy.split` : Split an array into multiple sub-arrays.

    :obj:`array_split_nd`: Split an array into multiple sub-arrays recursively, possibly unevenly.


    Parameters
    ----------
    ary : :obj:`AnyNDArray`
        Input array.

    args : :obj:`int`, optional
        Number of splits for each axis of `ary`.
        Axis 0 will be split into `args[0]` subarrays, axis 1 will be
        into `args[1]` subarrays, etc. If the split cannot be made even
        for all dimensions, raises an error.

    Returns
    -------
    :obj:`RecursiveListAnyNDArray <curvelops.typing.RecursiveListAnyNDArray>`
        Recursive lists of lists of :obj:`AnyNDArray`.
        The number of recursions is equivalent to the number arguments in args.

    Examples
    --------
    >>> from curvelops.utils import split_nd
    >>> ary = np.outer(1 + np.arange(2), 2 + np.arange(3))
    array([[2, 3, 4],
           [4, 6, 8]])
    >>> split_nd(ary, 2, 3)
    [[array([[2]]), array([[3]]), array([[4]])],
     [array([[4]]), array([[6]]), array([[8]])]]

    >>> from curvelops.utils import split_nd
    >>> ary = np.outer(np.arange(3), np.arange(5))
    >>> split_nd(ary, 2, 3)
    ValueError: array split does not result in an equal division
    """
    axis = ary.ndim - len(args)
    split = np.split(ary, args[0], axis=axis)
    if len(args) == 1:
        return split
    return [split_nd(s, *args[1:]) for s in split]


def ndargmax(ary: AnyNDArray) -> tuple[np.intp, ...]:
    """N-dimensional argmax of array.

    Parameters
    ----------
    ary : :obj:`AnyNDArray`
        Input array

    Examples
    --------
    >>> import numpy as np
    >>> from curvelops.utils import ndargmax
    >>> x = np.zeros((10, 10, 10))
    >>> x[1, 1, 1] = 1.0
    >>> ndargmax(x)
    (1, 1, 1)

    Returns
    -------
    tuple
        N-dimensional index of the maximum of ``ary``.
    """
    return np.unravel_index(ary.argmax(), ary.shape)


def apply_along_wedges(
    c_struct: list[list[list[AnyNDArray]]],
    fun: Callable[
        [
            AnyNDArray,
            int,  # wedge index
            int,  # direction index
            int,  # scale index
            int,  # nwedges
            int,  # ndirections
            int,  # nscales
        ],
        T,
    ],
) -> list[list[list[T]]]:
    """Applies a function to each individual wedge.

    Parameters
    ----------
    c_struct : :obj:`FDCTStructLike <curvelops.typing.FDCTStructLike>`
        Input curvelet coefficients in struct format.
    fun : Callable[[:obj:`AnyNDArray <numpy.typing.AnyNDArray>`, :obj:`int`, :obj:`int`, :obj:`int`, :obj:`int`], T]
        Function to apply to each individual wedge. The function's arguments
        are respectively: `wedge`, `wedge index in scale`, `scale index`, `number of
        wedges in scale`, `number of scales`.

    Returns
    -------
    List[List[T]]
        Struct containing the result of applying `fun` to each wedge.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelops import FDCT2D
    >>> from curvelops.utils import apply_along_wedges
    >>> x = np.zeros((32, 32))
    >>> C = FDCT2D(x.shape, nbscales=3, nbangles_coarse=8, allcurvelets=False)
    >>> y = C.struct(C @ x)
    >>> apply_along_wedges(y, lambda w, *_: w.shape)
    [[(11, 11)],
     [(23, 11),
      (23, 11),
      (11, 23),
      (11, 23),
      (23, 11),
      (23, 11),
      (11, 23),
      (11, 23)],
     [(32, 32)]]
    """
    mapped_struct: list[list[list[T]]] = []
    for iscale, c_angles in enumerate(c_struct):
        tmp_scale = []
        for idir, c_dir in enumerate(c_angles):
            tmp_dir = []
            for iwedge, c_wedge in enumerate(c_dir):
                out = fun(
                    c_wedge,
                    iwedge,
                    idir,
                    iscale,
                    len(c_dir),
                    len(c_angles),
                    len(c_struct),
                )
                tmp_dir.append(out)
            tmp_scale.append(tmp_dir)
        mapped_struct.append(tmp_scale)
    return mapped_struct


def energy(ary: ComplexNDArray) -> float:
    r"""Computes the energy of an n-dimensional wedge.

    The energy of a vector (flattened n-dimensional array)
    :math:`(a_0,\ldots,a_{N-1})` is defined as

    .. math::

        \sqrt{\frac{1}{N}\sum\limits_{i=0}^{N-1} |a_i|^2}.

    Parameters
    ----------
    ary : :obj:`AnyNDArray <numpy.typing.AnyNDArray>`
        Input wedge.

    Returns
    -------
    :obj:`float`
        Energy.
    """
    return math.sqrt((ary.real**2 + ary.imag**2).mean())


def energy_split(ary: AnyNDArray, rows: int, cols: int) -> AnyNDArray:
    """Splits a wedge into ``(rows, cols)`` wedges and computes the energy
    of each of these subdivisions.

    See Also
    --------

    :obj:`energy` : Computes the energy of a wedge.

    Parameters
    ----------
    ary : :obj:`AnyNDArray <numpy.typing.AnyNDArray>`
        Input wedge.
    rows : :obj:`int`
        Split axis 0 into `rows` subdivisions.
    cols : :obj:`int`
        Split axis 1 into `cols` subdivisions.

    Returns
    -------
    :obj:`AnyNDArray <numpy.typing.AnyNDArray>`
        Matrix of shape ``(rows, cols)`` containing the energy of each
        subdivision of the input wedge.
    """
    norm_local: NDArray[float] = np.empty((rows, cols), dtype=float)
    split = array_split_nd(ary, rows, cols)
    for irow in range(rows):
        for icol in range(cols):
            norm_local[irow, icol] = energy(split[irow][icol])
    return norm_local


def normal_vector_field(
    data: AnyNDArray, rows: int, cols: int, dx: float = 1, dy: float = 1
) -> NDArray[np.float64]:
    kvecs: NDArray[np.float64] = np.empty((rows, cols, 2), dtype=float)
    d_split: list[list[AnyNDArray]] = array_split_nd(data.T, rows, cols)

    for irow in range(kvecs.shape[0]):
        for icol in range(kvecs.shape[1]):
            d_loc = d_split[irow][icol].T
            d_k_loc = fftshift(fft2(d_loc))
            kx_loc = fftshift(fftfreq(d_loc.shape[0], d=dx))
            kz_loc = fftshift(fftfreq(d_loc.shape[1], d=dy))

            # Use top quadrants of f-k spectrum
            top_quadrant = kz_loc > 0
            kx_locmax, kz_locmax = ndargmax(np.abs(d_k_loc[:, top_quadrant]))

            k = np.array([kx_loc[kx_locmax], kz_loc[top_quadrant][kz_locmax]])
            kvecs[irow, icol, :] = k / np.linalg.norm(k)
    return kvecs
