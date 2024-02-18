from __future__ import annotations

import numpy as np

from .udctmdwin import udctmdwin
from .utils import ParamUDCT, downsamp, upsamp


def udctmddec(
    im: np.ndarray, param_udct: ParamUDCT, udctwin: dict[dict[np.ndarray | dict]]
) -> dict[dict[np.ndarray | dict]]:
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


def udctmdrec(
    coeff: dict[dict[np.ndarray | dict]],
    param_udct: ParamUDCT,
    udctwin: dict[dict[np.ndarray | dict]],
) -> np.ndarray:
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


class UDCT:
    def __init__(
        self,
        size: tuple[int, ...],
        cfg: np.ndarray | None = None,
        alpha: float = 0.15,
        r: tuple[float, float, float, float] | None = None,
        winthresh: float = 1e-5,
    ) -> None:
        dim = len(size)
        cfg1 = np.c_[np.ones((dim,)) * 3, np.ones((dim,)) * 6].T if cfg is None else cfg
        one = np.pi / 3
        r1 = (one, 2 * one, 2 * one, 4 * one) if r is None else r
        self.params = ParamUDCT(
            dim=dim, size=size, cfg=cfg1, alpha=alpha, r=r1, winthresh=winthresh
        )

        self.windows = udctmdwin(self.params)

    def forward(self, x: np.ndarray) -> dict[dict[np.ndarray | dict]]:
        return udctmddec(x, self.params, self.windows)

    def backward(self, c: dict[dict[np.ndarray | dict]]) -> np.ndarray:
        return udctmdrec(c, self.params, self.windows)
