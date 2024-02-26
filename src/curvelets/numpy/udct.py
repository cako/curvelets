from __future__ import annotations

import logging
from math import prod

import numpy as np
import numpy.typing as npt

from .typing import UDCTCoefficients, UDCTWindows
from .udctmdwin import udctmdwin
from .utils import ParamUDCT, downsamp, from_sparse_new, upsamp


def udctmddec(
    im: np.ndarray,
    param_udct: ParamUDCT,
    udctwin: UDCTWindows,
    decimation_ratio: list[npt.NDArray[np.int_]],
) -> UDCTCoefficients:
    imf = np.fft.fftn(im)
    cdtype = imf.dtype

    fband = np.zeros_like(imf)
    idx, val = from_sparse_new(udctwin[0][0][0])
    fband.flat[idx] = imf.flat[idx] * val.astype(cdtype)
    cband = np.fft.ifftn(fband)

    coeff: UDCTCoefficients = [[[downsamp(cband, decimation_ratio[0][0])]]]
    norm = np.sqrt(
        np.prod(np.full((param_udct.dim,), fill_value=2 ** (param_udct.res - 1)))
    )
    coeff[0][0][0] *= norm

    for ires in range(1, 1 + param_udct.res):
        coeff.append([])
        for idir in range(param_udct.dim):
            coeff[ires].append([])
            for iang in range(len(udctwin[ires][idir])):
                fband = np.zeros_like(imf)
                idx, val = from_sparse_new(udctwin[ires][idir][iang])
                fband.flat[idx] = imf.flat[idx] * val.astype(cdtype)

                cband = np.fft.ifftn(fband)
                decim = decimation_ratio[ires][idir, :]
                coeff[ires][idir].append(downsamp(cband, decim))
                coeff[ires][idir][iang] *= np.sqrt(2 * np.prod(decim))
    return coeff


def udctmdrec(
    coeff: UDCTCoefficients,
    param_udct: ParamUDCT,
    udctwin: UDCTWindows,
    decimation_ratio: list[npt.NDArray[np.int_]],
) -> np.ndarray:
    rdtype = coeff[0][0][0].real.dtype
    cdtype = (np.ones(1, dtype=rdtype) + 1j * np.ones(1, dtype=rdtype)).dtype
    imf = np.zeros(param_udct.size, dtype=cdtype)

    for ires in range(1, 1 + param_udct.res):
        for idir in range(param_udct.dim):
            for iang in range(len(udctwin[ires][idir])):
                decim = decimation_ratio[ires][idir, :]
                cband = upsamp(coeff[ires][idir][iang], decim)
                cband /= np.sqrt(2 * np.prod(decim))
                cband = np.prod(decim) * np.fft.fftn(cband)
                idx, val = from_sparse_new(udctwin[ires][idir][iang])
                imf.flat[idx] += cband.flat[idx] * val.astype(cdtype)

    imfl = np.zeros(param_udct.size, dtype=cdtype)
    decim = decimation_ratio[0][0]
    cband = upsamp(coeff[0][0][0], decim)
    cband = np.sqrt(np.prod(decim)) * np.fft.fftn(cband)
    idx, val = from_sparse_new(udctwin[0][0][0])
    imfl.flat[idx] += cband.flat[idx] * val.astype(cdtype)
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
        self.shape = shape
        dim = len(shape)
        cfg1 = np.c_[np.ones((dim,)) * 3, np.ones((dim,)) * 6].T if cfg is None else cfg
        r1: tuple[float, float, float, float] = (
            tuple(np.array([1.0, 2.0, 2.0, 4.0]) * np.pi / 3) if r is None else r
        )
        self.params = ParamUDCT(
            dim=dim, size=shape, cfg=cfg1, alpha=alpha, r=r1, winthresh=winthresh
        )

        self.windows, self.decimation, self.indices = udctmdwin(self.params)

    def vect(self, coeffs: UDCTCoefficients) -> npt.NDArray[np.complexfloating]:
        coeffs_vec = []
        for c in coeffs:
            for d in c:
                for a in d:
                    coeffs_vec.append(a.ravel())
        return np.concatenate(coeffs_vec)

    def struct(self, coeffs_vec: npt.NDArray[np.complexfloating]) -> UDCTCoefficients:
        ibeg = 0
        coeffs: UDCTCoefficients = []
        for ires, decres in enumerate(self.decimation):
            coeffs.append([])
            for idir, decdir in enumerate(decres):
                coeffs[ires].append([])
                for _ in self.windows[ires][idir]:
                    shape_decim = self.shape // decdir
                    iend = ibeg + prod(shape_decim)
                    coeffs[ires][idir].append(
                        coeffs_vec[ibeg:iend].reshape(shape_decim)
                    )
                    ibeg = iend
        return coeffs

    def forward(self, x: np.ndarray) -> UDCTCoefficients:
        return udctmddec(x, self.params, self.windows, self.decimation)

    def backward(self, c: UDCTCoefficients) -> np.ndarray:
        return udctmdrec(c, self.params, self.windows, self.decimation)


class SimpleUDCT(UDCT):
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

        super().__init__(shape=shape, cfg=cfg, alpha=alpha, r=r, winthresh=winthresh)
