from __future__ import annotations

import sys
from dataclasses import dataclass, field
from math import ceil, prod

import numpy as np


@dataclass(**({"kw_only": True} if sys.version_info >= (3, 10) else {}))
class ParamUDCT:
    dim: int
    size: tuple[int, ...]
    cfg: tuple | np.ndarray  # last dimension  == dim
    alpha: float
    r: tuple[float, float, float, float]
    winthresh: float
    len: int = field(init=False)
    res: int = field(init=False)
    decim: np.ndarray = field(init=False)
    ind: dict[int, dict[int, np.ndarray]] | None = None
    dec: dict[int, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.len = prod(self.size)
        self.res = len(self.cfg)
        self.decim = 2 * (np.asarray(self.cfg, dtype=int) // 3)


def circshift(arr, shape: tuple[int, ...]):
    assert arr.ndim == len(shape)
    return np.roll(arr, shape, axis=tuple(range(len(shape))))


def adapt_grid(S1, S2):
    x1, x2 = np.meshgrid(S2, S1)

    t1 = np.zeros_like(x1, dtype=float)
    ind = (x1 != 0) & (np.abs(x2) <= np.abs(x1))
    t1[ind] = -x2[ind] / x1[ind]

    t2 = np.zeros_like(x1, dtype=float)
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


def angle_fun(Mgrid, direction, n, alpha):
    # % create 2-D grid function-------------------------------------------------

    # angle meyer window
    angd = 2 / n
    ang = angd * np.array([-alpha, alpha, 1 - alpha, 1 + alpha])

    Mang = []
    if direction == 1:
        for jn in range(1, ceil(n / 2) + 1):
            ang2 = -1 + (jn - 1) * angd + ang
            fang = fun_meyer(Mgrid, *ang2)
            Mang.append(fang[None, :])
    elif direction == 2:  # This is weird, both directions are the same code
        for jn in range(1, ceil(n / 2) + 1):
            ang2 = -1 + (jn - 1) * angd + ang
            fang = fun_meyer(Mgrid, *ang2)
            Mang.append(fang[None, :])
    else:
        msg = "Unrecognized direction"
        raise ValueError(msg)
    return np.concatenate(Mang, axis=0)


def angle_kron(angle_fun, nper, param_udct):
    # , nper, param_udct
    krsz = np.ones(3, dtype=int)
    krsz[0] = np.prod(param_udct.size[: nper[0] - 1])
    krsz[1] = np.prod(param_udct.size[nper[0] : nper[1] - 1])
    krsz[2] = np.prod(param_udct.size[nper[1] : param_udct.dim])

    tmp = angle_fun

    tmp1 = np.kron(np.ones((krsz[1], 1), dtype=int), tmp)
    tmp2 = np.kron(np.ones((krsz[2], 1), dtype=int), travel(tmp1)).ravel()
    tmp3 = travel(np.kron(tmp2, np.ones((krsz[0], 1), dtype=int)))
    ang_md = tmp3.reshape(*param_udct.size[::-1]).T

    return ang_md


def downsamp(F, decim):
    assert F.ndim == len(decim)
    D = F[tuple(slice(None, None, d) for d in decim)].copy()
    return D


def fftflip(F, axis):
    Fc = F
    dim = F.ndim
    shiftvec = np.zeros((dim,), dtype=int)
    shiftvec[axis] = 1
    Fc = np.flip(F, axis)
    Fc = circshift(Fc, shiftvec)
    return Fc


def fun_meyer(x, p1, p2, p3, p4):
    p = np.array([-20.0, 70.0, -84.0, 35.0, 0.0, 0.0, 0.0, 0.0])
    y = np.zeros_like(x)

    win = (x >= p1) & (x <= p2)
    y[win] = np.polyval(p, (x[win] - p1) / (p2 - p1))

    win = (x > p2) & (x <= p3)
    y[win] = 1.0

    win = (x >= p3) & (x <= p4)
    y[win] = np.polyval(p, (x[win] - p4) / (p3 - p4))
    return y


def travel(arr):
    return arr.T.ravel()


def to_sparse(arr, thresh):
    idx = np.argwhere(travel(arr) > thresh)
    return np.c_[idx + 1, travel(arr)[idx]]


def upsamp(F, decim):
    assert F.ndim == len(decim)
    upsamp_shape = tuple(s * d for s, d in zip(F.shape, decim))
    D = np.zeros(upsamp_shape, dtype=F.dtype)
    D[tuple(slice(None, None, d) for d in decim)] = F[...]
    return D
