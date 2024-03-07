from __future__ import annotations

import sys
from dataclasses import dataclass, field
from math import ceil, prod

import numpy as np
import torch
from torch import Tensor


@dataclass(**({"kw_only": True} if sys.version_info >= (3, 10) else {}))
class ParamUDCT:
    dim: int
    size: tuple[int, ...]
    cfg: Tensor  # last dimension  == dim
    alpha: float
    r: tuple[float, float, float, float]
    winthresh: float
    len: int = field(init=False)
    res: int = field(init=False)
    decim: Tensor = field(init=False)
    ind: dict[int, dict[int, Tensor]] | None = None
    dec: dict[int, Tensor] | None = None

    def __post_init__(self) -> None:
        self.len = prod(self.size)
        self.res = len(self.cfg)
        self.decim = 2 * (torch.as_tensor(self.cfg, dtype=int) // 3)


def circshift(arr: Tensor, shape: tuple[int, ...]) -> Tensor:
    assert arr.ndim == len(shape)
    return np.roll(arr, shape, axis=tuple(range(len(shape))))


def circshift_torch(arr: Tensor, shape: tuple[int, ...]) -> Tensor:
    assert arr.ndim == len(shape)
    return torch.roll(arr, shape, dims=tuple(range(len(shape))))


def adapt_grid(S1: Tensor, S2: Tensor) -> tuple[Tensor, Tensor]:
    x1, x2 = np.meshgrid(S2, S1)

    t1: Tensor = np.zeros_like(x1, dtype=float)
    ind = (x1 != 0) & (np.abs(x2) <= np.abs(x1))
    t1[ind] = -x2[ind] / x1[ind]

    t2: Tensor = np.zeros_like(x1, dtype=float)
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


def angle_fun(Mgrid: Tensor, direction: int, n: int, alpha: float) -> Tensor:
    # % create 2-D grid function-------------------------------------------------

    # angle meyer window
    angd = 2 / n
    ang = angd * np.array([-alpha, alpha, 1 - alpha, 1 + alpha])

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


def angle_kron(angle_arr: Tensor, nper: Tensor, param_udct: ParamUDCT) -> Tensor:
    # , nper, param_udct
    krsz: Tensor = np.ones(3, dtype=int)
    krsz[0] = np.prod(param_udct.size[: nper[0] - 1])
    krsz[1] = np.prod(param_udct.size[nper[0] : nper[1] - 1])
    krsz[2] = np.prod(param_udct.size[nper[1] : param_udct.dim])

    tmp1 = np.kron(np.ones((krsz[1], 1), dtype=int), angle_arr)
    tmp2 = np.kron(np.ones((krsz[2], 1), dtype=int), travel(tmp1)).ravel()
    tmp3 = travel(np.kron(tmp2, np.ones((krsz[0], 1), dtype=int)))
    return tmp3.reshape(*param_udct.size[::-1]).T


def downsamp(F: Tensor, decim: Tensor) -> Tensor:
    assert F.ndim == len(decim)
    return F[tuple(slice(None, None, d) for d in decim)]


def fftflip(F: Tensor, axis: int) -> Tensor:
    Fc = F
    dim = F.ndim
    shiftvec: Tensor = np.zeros((dim,), dtype=int)
    shiftvec[axis] = 1
    Fc = np.flip(F, axis)
    return circshift(Fc, tuple(shiftvec))


def fun_meyer(x: Tensor, p1: float, p2: float, p3: float, p4: float) -> Tensor:
    p = np.array([-20.0, 70.0, -84.0, 35.0, 0.0, 0.0, 0.0, 0.0])
    y = np.zeros_like(x)

    win = (x >= p1) & (x <= p2)
    y[win] = np.polyval(p, (x[win] - p1) / (p2 - p1))

    win = (x > p2) & (x <= p3)
    y[win] = 1.0

    win = (x >= p3) & (x <= p4)
    y[win] = np.polyval(p, (x[win] - p4) / (p3 - p4))
    return y


def travel(arr: Tensor) -> Tensor:
    return arr.T.ravel()


def travel_new(arr: Tensor) -> Tensor:
    return arr.ravel()


def to_sparse(arr: Tensor, thresh: float) -> Tensor:
    idx = np.argwhere(travel(arr) > thresh)
    out: Tensor = np.c_[idx + 1, travel(arr)[idx]]
    return out


def to_sparse_new(arr: Tensor, thresh: float) -> list[Tensor]:
    idx = np.argwhere(arr.ravel() > thresh)
    return [idx, arr.ravel()[idx]]


def from_sparse(arr: Tensor) -> tuple[Tensor, Tensor]:
    idx = arr[:, 0].astype(int) - 1
    val = arr[:, 1]
    return idx, val


def from_sparse_new(arr_list: list[Tensor]) -> list[Tensor]:
    return arr_list


def upsamp(F: Tensor, decim: Tensor) -> Tensor:
    assert F.ndim == len(decim)
    upsamp_shape = tuple(s * d for s, d in zip(F.shape, decim))
    D = torch.zeros(upsamp_shape, dtype=F.dtype)
    D[tuple(slice(None, None, d) for d in decim)] = F[...]
    return D
