from __future__ import annotations

import sys
from dataclasses import dataclass, field
from math import ceil, prod
from typing import TypeVar

import numpy as np
import numpy.typing as npt

D_T = TypeVar("D_T", bound=np.floating)


@dataclass(**({"kw_only": True} if sys.version_info >= (3, 10) else {}))
class ParamUDCT:
    dim: int
    size: tuple[int, ...]
    angular_wedges_config: npt.NDArray[np.int_]  # last dimension  == dim
    window_overlap: float
    radial_frequency_params: tuple[float, float, float, float]
    window_threshold: float
    len: int = field(init=False)
    res: int = field(init=False)
    decim: npt.NDArray[np.int_] = field(init=False)
    ind: dict[int, dict[int, np.ndarray]] | None = None
    dec: dict[int, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.len = prod(self.size)
        self.res = len(self.angular_wedges_config)
        self.decim = 2 * (np.asarray(self.angular_wedges_config, dtype=int) // 3)


def circshift(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    assert arr.ndim == len(shape)
    return np.roll(arr, shape, axis=tuple(range(len(shape))))


def adapt_grid(S1: np.ndarray, S2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x1, x2 = np.meshgrid(S2, S1)

    t1: npt.NDArray[np.floating] = np.zeros_like(x1, dtype=float)
    ind = (x1 != 0) & (np.abs(x2) <= np.abs(x1))
    t1[ind] = -x2[ind] / x1[ind]

    t2: npt.NDArray[np.floating] = np.zeros_like(x1, dtype=float)
    ind = (x2 != 0) & (np.abs(x1) < np.abs(x2))
    t2[ind] = x1[ind] / x2[ind]

    t3 = t2.copy()
    t3[t2 < 0] = t2[t2 < 0] + 2
    t3[t2 > 0] = t2[t2 > 0] - 2

    M1 = t1 + t3
    M1[x1 >= 0] = -2

    t1 = np.zeros_like(x1, dtype=float)
    ind = (x2 != 0) & (abs(x1) <= abs(x2))
    t1[ind] = -x1[ind] / x2[ind]

    t2 = np.zeros_like(x1, dtype=float)
    ind = (x1 != 0) & (abs(x2) < abs(x1))
    t2[ind] = x2[ind] / x1[ind]

    t3 = t2.copy()
    t3[t2 < 0] = t2[t2 < 0] + 2
    t3[t2 > 0] = t2[t2 > 0] - 2

    M2 = t1 + t3
    M2[x2 >= 0] = -2

    return M2, M1


def angle_fun(
    Mgrid: np.ndarray, direction: int, n: int, window_overlap: float
) -> np.ndarray:
    # % create 2-D grid function-------------------------------------------------

    # angle meyer window
    angd = 2 / n
    ang = angd * np.array(
        [-window_overlap, window_overlap, 1 - window_overlap, 1 + window_overlap]
    )

    Mang = []
    # This is weird, both directions are the same code
    if direction in (1, 2):
        for jn in range(1, ceil(n / 2) + 1):
            ang2 = -1 + (jn - 1) * angd + ang
            fang = fun_meyer(Mgrid, *ang2)
            Mang.append(fang[None, :])
    else:
        msg = "Unrecognized direction"
        raise ValueError(msg)
    return np.concatenate(Mang, axis=0)


def angle_kron(
    angle_arr: np.ndarray, nper: np.ndarray, param_udct: ParamUDCT
) -> np.ndarray:
    # , nper, param_udct
    krsz: npt.NDArray[np.int_] = np.ones(3, dtype=int)
    krsz[0] = np.prod(param_udct.size[: nper[0] - 1])
    krsz[1] = np.prod(param_udct.size[nper[0] : nper[1] - 1])
    krsz[2] = np.prod(param_udct.size[nper[1] : param_udct.dim])

    tmp1 = np.kron(np.ones((krsz[1], 1), dtype=int), angle_arr)
    tmp2 = np.kron(np.ones((krsz[2], 1), dtype=int), tmp1.ravel()).ravel()
    tmp3 = np.kron(tmp2, np.ones((krsz[0], 1), dtype=int)).ravel()
    return tmp3.reshape(*param_udct.size)


def downsamp(F: np.ndarray, decim: np.ndarray) -> np.ndarray:
    assert F.ndim == len(decim)
    return F[tuple(slice(None, None, d) for d in decim)]


def fftflip(F: np.ndarray, axis: int) -> np.ndarray:
    Fc = F
    dim = F.ndim
    shiftvec: npt.NDArray[np.int_] = np.zeros((dim,), dtype=int)
    shiftvec[axis] = 1
    Fc = np.flip(F, axis)
    return circshift(Fc, tuple(shiftvec))


def _fftflip_all_axes(F: np.ndarray) -> np.ndarray:
    """
    Apply fftflip to all axes of an array.

    This produces X(-omega) from X(omega) in FFT representation.
    After flipping, the array is circshifted by 1 in each dimension
    to maintain proper frequency alignment.

    Parameters
    ----------
    F : np.ndarray
        Input array in FFT representation.

    Returns
    -------
    np.ndarray
        Flipped array representing negative frequencies.
    """
    Fc = F.copy()
    for axis in range(F.ndim):
        Fc = np.flip(Fc, axis)
    shiftvec = tuple(1 for _ in range(F.ndim))
    return circshift(Fc, shiftvec)


def fun_meyer(x: np.ndarray, p1: float, p2: float, p3: float, p4: float) -> np.ndarray:
    p = np.array([-20.0, 70.0, -84.0, 35.0, 0.0, 0.0, 0.0, 0.0])
    y = np.zeros_like(x)

    win = (x >= p1) & (x <= p2)
    y[win] = np.polyval(p, (x[win] - p1) / (p2 - p1))

    win = (x > p2) & (x <= p3)
    y[win] = 1.0

    win = (x >= p3) & (x <= p4)
    y[win] = np.polyval(p, (x[win] - p4) / (p3 - p4))
    return y


def to_sparse(
    arr: npt.NDArray[D_T], thresh: float
) -> tuple[npt.NDArray[np.intp], npt.NDArray[D_T]]:
    idx = np.argwhere(arr.ravel() > thresh)
    return (idx, arr.ravel()[idx])


def upsamp(F: np.ndarray, decim: np.ndarray) -> np.ndarray:
    assert F.ndim == len(decim)
    upsamp_shape = tuple(s * d for s, d in zip(F.shape, decim))
    D = np.zeros(upsamp_shape, dtype=F.dtype)
    D[tuple(slice(None, None, d) for d in decim)] = F[...]
    return D
