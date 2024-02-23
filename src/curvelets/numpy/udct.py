from __future__ import annotations

import logging

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

    for ires in range(1, 1 + param_udct.res):
        coeff[ires] = {}
        for idir in range(param_udct.dim):
            coeff[ires][idir] = {}
            for iang in range(len(udctwin[ires][idir])):
                fband = np.zeros_like(imf)
                idx, val = from_sparse_new(udctwin[ires][idir][iang])
                fband.flat[idx] = imf.flat[idx] * val

                cband = np.fft.ifftn(fband)
                decim = decimation_ratio[ires][idir, :]
                coeff[ires][idir][iang] = downsamp(cband, decim)
                coeff[ires][idir][iang] *= np.sqrt(
                    2 * np.prod(decimation_ratio[ires][idir, :])
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

    for ires in range(1, 1 + param_udct.res):
        for idir in range(param_udct.dim):
            for iang in range(len(udctwin[ires][idir])):
                decim = decimation_ratio[ires][idir, :]
                cband = upsamp(coeff[ires][idir][iang], decim)
                cband /= np.sqrt(2 * np.prod(decimation_ratio[ires][idir, :]))
                cband = np.prod(decimation_ratio[ires][idir, :]) * np.fft.fftn(cband)
                idx, val = from_sparse_new(udctwin[ires][idir][iang])
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
        shape: tuple[int, ...],
        cfg: np.ndarray | None = None,
        alpha: float = 0.15,
        r: tuple[float, float, float, float] | None = None,
        winthresh: float = 1e-5,
    ) -> None:
        dim = len(shape)
        cfg1 = np.c_[np.ones((dim,)) * 3, np.ones((dim,)) * 6].T if cfg is None else cfg
        r1: tuple[float, float, float, float] = (
            tuple(np.array([1.0, 2.0, 2.0, 4.0]) * np.pi / 3) if r is None else r
        )
        self.params = ParamUDCT(
            dim=dim, size=shape, cfg=cfg1, alpha=alpha, r=r1, winthresh=winthresh
        )

        self.windows, self.decimation_ratio, self.indices = udctmdwin(self.params)

    def forward(self, x: np.ndarray) -> dict[int, dict[int, dict[int, np.ndarray]]]:
        return udctmddec(x, self.params, self.windows, self.decimation_ratio)

    def backward(self, c: dict[int, dict[int, dict[int, np.ndarray]]]) -> np.ndarray:
        return udctmdrec(c, self.params, self.windows, self.decimation_ratio)


class SimpleUDCT:
    def __init__(
        self,
        shape: tuple[int, ...],
        nscales: int = 3,
        nbands_per_direction: int = 3,
        alpha: float | None = None,
        winthresh: float = 1e-5,
    ) -> None:
        assert nscales > 1
        assert nbands_per_direction >= 3

        dim = len(shape)
        nbands: npt.NDArray[np.int_] = (
            nbands_per_direction * 2 ** np.arange(nscales - 1)
        ).astype(int)

        if alpha is None:
            if nbands_per_direction == 3:
                alpha = 0.15
            elif nbands_per_direction == 4:
                alpha = 0.3
            elif nbands_per_direction == 5:
                alpha = 0.5
            else:
                alpha = 0.5
        for i, nb in enumerate(nbands, start=1):
            if 2**i * (1 + 2 * alpha) * (1 + alpha) >= nb:
                msg = f"alpha={alpha:.3f} does not respect respect the relationship (2^{i}/{nb})(1+2α)(1+α) < 1 for scale {i+1}"  # noqa: RUF001
                logging.warning(msg)
        cfg = np.tile(nbands[:, None], dim)
        r: tuple[float, float, float, float] = tuple(
            np.array([1.0, 2.0, 2.0, 4.0]) * np.pi / 3
        )
        self.params = ParamUDCT(
            dim=dim, size=shape, cfg=cfg, alpha=alpha, r=r, winthresh=winthresh
        )

        self.windows, self.decimation_ratio, self.indices = udctmdwin(self.params)

    def forward(self, x: np.ndarray) -> dict[int, dict[int, dict[int, np.ndarray]]]:
        return udctmddec(x, self.params, self.windows, self.decimation_ratio)

    def backward(self, c: dict[int, dict[int, dict[int, np.ndarray]]]) -> np.ndarray:
        return udctmdrec(c, self.params, self.windows, self.decimation_ratio)
