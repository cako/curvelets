from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from ._downsamp import downsamp
from ._upsamp import upsamp


@dataclass(**(dict(kw_only=True) if sys.version_info >= (3, 10) else {}))
class ParamUDCT:
    dim: int
    size: tuple[int, int] | tuple[int, int, int]
    cfg: tuple | np.ndarray  # last dimension  == dim
    alpha: float
    r: tuple[float, float, float, float]
    winthresh: float
    len: int = field(init=False)
    res: int = field(init=False)
    decim: np.ndarray = field(init=False)
    ind: dict[int, dict[int, np.ndarray]] | None = None
    dec: dict[int, np.ndarray] | None = None

    def __post_init__(self):
        self.len = np.prod(self.size)
        self.res = len(self.cfg)
        self.decim = 2 * (np.asarray(self.cfg, dtype=int) // 3)


def udctmddec(im, param_udct, udctwin):
    imf = np.fft.fftn(im)

    fband = np.zeros_like(imf)
    idx = udctwin[1][1][:, 0].astype(int) - 1
    val = udctwin[1][1][:, 1]
    fband.T.flat[idx] = imf.T.flat[idx] * val
    cband = np.fft.ifftn(fband)

    coeff = {}
    coeff[1] = {}
    decim = np.full((param_udct.dim,), fill_value=2 ** (param_udct.res - 1), dtype=int)
    coeff[1][1] = downsamp(cband, decim)
    norm = np.sqrt(
        np.prod(np.full((param_udct.dim,), fill_value=2 ** (param_udct.res - 1)))
    )
    coeff[1][1] *= norm

    for res in range(1, 1 + param_udct.res):
        coeff[res + 1] = {}
        for dir in range(1, 1 + param_udct.dim):
            coeff[res + 1][dir] = {}
            for ang in range(1, 1 + len(udctwin[res + 1][dir])):
                fband = np.zeros_like(imf)
                idx = udctwin[res + 1][dir][ang][:, 0].astype(int) - 1
                val = udctwin[res + 1][dir][ang][:, 1]
                fband.T.flat[idx] = imf.T.flat[idx] * val

                cband = np.fft.ifftn(fband)
                decim = param_udct.dec[res][dir - 1, :].astype(int)
                coeff[res + 1][dir][ang] = downsamp(cband, decim)
                coeff[res + 1][dir][ang] *= np.sqrt(
                    2 * np.prod(param_udct.dec[res][dir - 1, :])
                )
    return coeff


def udctmdrec(coeff, param_udct, udctwin):
    imf = np.zeros(param_udct.size, dtype=np.complex128)

    for res in range(1, 1 + param_udct.res):
        for dir in range(1, 1 + param_udct.dim):
            for ang in range(1, 1 + len(udctwin[res + 1][dir])):
                decim = param_udct.dec[res][dir - 1, :].astype(int)
                cband = upsamp(coeff[res + 1][dir][ang], decim)
                cband /= np.sqrt(2 * np.prod(param_udct.dec[res][dir - 1, :]))
                cband = np.prod(param_udct.dec[res][dir - 1, :]) * np.fft.fftn(cband)
                idx = udctwin[res + 1][dir][ang][:, 0].astype(int) - 1
                val = udctwin[res + 1][dir][ang][:, 1]
                imf.T.flat[idx] += cband.T.flat[idx] * val

    imfl = np.zeros(param_udct.size, dtype=np.complex128)
    decimlow = np.full(
        (param_udct.dim,), fill_value=2 ** (param_udct.res - 1), dtype=int
    )
    cband = upsamp(coeff[1][1], decimlow)
    cband = np.sqrt(np.prod(decimlow)) * np.fft.fftn(cband)
    idx = udctwin[1][1][:, 0].astype(int) - 1
    val = udctwin[1][1][:, 1]
    imfl.T.flat[idx] += cband.T.flat[idx] * val
    imf = 2 * imf + imfl
    im2 = np.fft.ifftn(imf).real
    return im2
