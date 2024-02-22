from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .udctmdwin import udctmdwin
from .utils import ParamUDCT, downsamp, from_sparse_new, upsamp


def udctmddec(
    im: np.ndarray,
    param_udct: ParamUDCT,
    udctwin: dict[int, dict[int, dict[int, list[np.ndarray]]]],
    decimation_ratio: dict[int, npt.NDArray[np.int_]],
) -> dict[int, dict[int, dict[int, np.ndarray]]]:
    imf = np.fft.fftn(im)

    fband = np.zeros_like(imf)
    idx, val = from_sparse_new(udctwin[0][0][0])
    fband.flat[idx] = imf.flat[idx] * val
    cband = np.fft.ifftn(fband)

    coeff: dict[int, dict[int, dict[int, np.ndarray]]] = {}
    coeff[0] = {}
    coeff[0][0] = {}
    decim: npt.NDArray[np.int_] = np.full(
        (param_udct.dim,), fill_value=2 ** (param_udct.res - 1), dtype=int
    )
    coeff[0][0][0] = downsamp(cband, decim)
    norm = np.sqrt(
        np.prod(np.full((param_udct.dim,), fill_value=2 ** (param_udct.res - 1)))
    )
    coeff[0][0][0] *= norm

    for res in range(1, 1 + param_udct.res):
        coeff[res] = {}
        for dir in range(param_udct.dim):
            coeff[res][dir] = {}
            for ang in range(len(udctwin[res][dir])):
                fband = np.zeros_like(imf)
                idx, val = from_sparse_new(udctwin[res][dir][ang])
                fband.flat[idx] = imf.flat[idx] * val

                cband = np.fft.ifftn(fband)
                decim = decimation_ratio[res][dir, :]
                coeff[res][dir][ang] = downsamp(cband, decim)
                coeff[res][dir][ang] *= np.sqrt(
                    2 * np.prod(decimation_ratio[res][dir, :])
                )
    return coeff


def udctmdrec(
    coeff: dict[int, dict[int, dict[int, np.ndarray]]],
    param_udct: ParamUDCT,
    udctwin: dict[int, dict[int, dict[int, list[np.ndarray]]]],
    decimation_ratio: dict[int, npt.NDArray[np.int_]],
) -> np.ndarray:
    rdtype = udctwin[0][0][0][1].real.dtype
    cdtype = (np.ones(1, dtype=rdtype) + 1j * np.ones(1, dtype=rdtype)).dtype
    imf = np.zeros(param_udct.size, dtype=cdtype)

    for res in range(1, 1 + param_udct.res):
        for dir in range(param_udct.dim):
            for ang in range(len(udctwin[res][dir])):
                decim = decimation_ratio[res][dir, :]
                cband = upsamp(coeff[res][dir][ang], decim)
                cband /= np.sqrt(2 * np.prod(decimation_ratio[res][dir, :]))
                cband = np.prod(decimation_ratio[res][dir, :]) * np.fft.fftn(cband)
                idx, val = from_sparse_new(udctwin[res][dir][ang])
                imf.flat[idx] += cband.flat[idx] * val

    imfl = np.zeros(param_udct.size, dtype=cdtype)
    decimlow: npt.NDArray[np.int_] = np.full(
        (param_udct.dim,), fill_value=2 ** (param_udct.res - 1), dtype=int
    )
    cband = upsamp(coeff[0][0][0], decimlow)
    cband = np.sqrt(np.prod(decimlow)) * np.fft.fftn(cband)
    idx, val = from_sparse_new(udctwin[0][0][0])
    imfl.flat[idx] += cband.flat[idx] * val
    imf = 2 * imf + imfl
    return np.fft.ifftn(imf).real


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

        self.windows, self.decimation_ratio, self.indices = udctmdwin(self.params)

    def forward(self, x: np.ndarray) -> dict[int, dict[int, dict[int, np.ndarray]]]:
        return udctmddec(x, self.params, self.windows, self.decimation_ratio)

    def backward(self, c: dict[int, dict[int, dict[int, np.ndarray]]]) -> np.ndarray:
        return udctmdrec(c, self.params, self.windows, self.decimation_ratio)
