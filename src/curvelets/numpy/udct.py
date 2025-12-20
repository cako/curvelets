from __future__ import annotations

import logging
from math import prod
from typing import Literal

import numpy as np
import numpy.typing as npt

from curvelets.ucurv.meyerwavelet import meyerfwdmd, meyerinvmd

from .typing import UDCTCoefficients, UDCTWindows
from .udctmdwin import udctmdwin
from .utils import ParamUDCT, circshift, downsamp, from_sparse_new, upsamp


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


def udctmddec(
    im: np.ndarray,
    param_udct: ParamUDCT,
    udctwin: UDCTWindows,
    decimation_ratio: list[npt.NDArray[np.int_]],
    complex: bool = False,
) -> UDCTCoefficients:
    """
    Apply UDCT decomposition (forward transform).

    Parameters
    ----------
    im : np.ndarray
        Input image/volume.
    param_udct : ParamUDCT
        UDCT parameters.
    udctwin : UDCTWindows
        Curvelet windows in sparse format.
    decimation_ratio : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale/direction.
    complex : bool, optional
        If True, use complex transform (separate +/- frequency bands).
        If False, use real transform (combined +/- frequencies). Default is False.

    Returns
    -------
    UDCTCoefficients
        Curvelet coefficients. When complex=True, directions are doubled
        (first dim directions for positive frequencies, next dim for negative).
    """
    imf = np.fft.fftn(im)
    cdtype = imf.dtype

    # Low frequency band processing
    fband = np.zeros_like(imf)
    idx, val = from_sparse_new(udctwin[0][0][0])
    fband.flat[idx] = imf.flat[idx] * val.astype(cdtype)

    if complex:
        # Complex transform: keep complex low frequency
        cband = np.fft.ifftn(fband)
    else:
        # Real transform: take real part
        cband = np.fft.ifftn(fband)

    coeff: UDCTCoefficients = [[[downsamp(cband, decimation_ratio[0][0])]]]
    norm = np.sqrt(
        np.prod(np.full((param_udct.dim,), fill_value=2 ** (param_udct.res - 1)))
    )
    coeff[0][0][0] *= norm

    if complex:
        # Complex transform: separate +/- frequency bands
        # Structure: [scale][direction][wedge]
        # Directions 0..dim-1 are positive frequencies
        # Directions dim..2*dim-1 are negative frequencies
        for ires in range(1, 1 + param_udct.res):
            coeff.append([])
            # Positive frequency bands (directions 0..dim-1)
            for idir in range(param_udct.dim):
                coeff[ires].append([])
                for iang in range(len(udctwin[ires][idir])):
                    # Convert sparse window to dense for manipulation
                    idx, val = from_sparse_new(udctwin[ires][idir][iang])
                    subwin = np.zeros(param_udct.size, dtype=val.dtype)
                    subwin.flat[idx] = val

                    # Apply window to frequency domain
                    bandfilt = np.sqrt(0.5) * np.fft.ifftn(imf * subwin.astype(cdtype))

                    decim = decimation_ratio[ires][idir, :]
                    coeff[ires][idir].append(downsamp(bandfilt, decim))
                    coeff[ires][idir][iang] *= np.sqrt(2 * np.prod(decim))

            # Negative frequency bands (directions dim..2*dim-1)
            for idir in range(param_udct.dim):
                coeff[ires].append([])
                for iang in range(len(udctwin[ires][idir])):
                    # Convert sparse window to dense for manipulation
                    idx, val = from_sparse_new(udctwin[ires][idir][iang])
                    subwin = np.zeros(param_udct.size, dtype=val.dtype)
                    subwin.flat[idx] = val

                    # Apply fftflip to get negative frequency window
                    subwin_flip = _fftflip_all_axes(subwin)

                    # Apply flipped window to frequency domain
                    bandfilt = np.sqrt(0.5) * np.fft.ifftn(
                        imf * subwin_flip.astype(cdtype)
                    )

                    decim = decimation_ratio[ires][idir, :]
                    coeff[ires][idir + param_udct.dim].append(downsamp(bandfilt, decim))
                    coeff[ires][idir + param_udct.dim][iang] *= np.sqrt(
                        2 * np.prod(decim)
                    )
    else:
        # Real transform: combined +/- frequencies
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
    complex: bool = False,
) -> np.ndarray:
    """
    Apply UDCT reconstruction (backward transform).

    Parameters
    ----------
    coeff : UDCTCoefficients
        Curvelet coefficients.
    param_udct : ParamUDCT
        UDCT parameters.
    udctwin : UDCTWindows
        Curvelet windows in sparse format.
    decimation_ratio : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale/direction.
    complex : bool, optional
        If True, use complex transform (separate +/- frequency bands).
        If False, use real transform (combined +/- frequencies). Default is False.

    Returns
    -------
    np.ndarray
        Reconstructed image/volume.
    """
    rdtype = coeff[0][0][0].real.dtype
    cdtype = (np.ones(1, dtype=rdtype) + 1j * np.ones(1, dtype=rdtype)).dtype
    imf = np.zeros(param_udct.size, dtype=cdtype)

    if complex:
        # Complex transform: reconstruct from separate +/- frequency bands
        for ires in range(1, 1 + param_udct.res):
            # Process positive frequency bands (directions 0..dim-1)
            for idir in range(param_udct.dim):
                for iang in range(len(udctwin[ires][idir])):
                    # Convert sparse window to dense
                    idx, val = from_sparse_new(udctwin[ires][idir][iang])
                    subwin = np.zeros(param_udct.size, dtype=val.dtype)
                    subwin.flat[idx] = val

                    decim = decimation_ratio[ires][idir, :]
                    cband = upsamp(coeff[ires][idir][iang], decim)
                    cband /= np.sqrt(2 * np.prod(decim))
                    cband = np.prod(decim) * np.fft.fftn(cband)

                    # Apply window
                    imf += np.sqrt(0.5) * cband * subwin.astype(cdtype)

            # Process negative frequency bands (directions dim..2*dim-1)
            for idir in range(param_udct.dim):
                for iang in range(len(udctwin[ires][idir])):
                    # Convert sparse window to dense
                    idx, val = from_sparse_new(udctwin[ires][idir][iang])
                    subwin = np.zeros(param_udct.size, dtype=val.dtype)
                    subwin.flat[idx] = val

                    # Apply fftflip to get negative frequency window
                    subwin_flip = _fftflip_all_axes(subwin)

                    decim = decimation_ratio[ires][idir, :]
                    cband = upsamp(coeff[ires][idir + param_udct.dim][iang], decim)
                    cband /= np.sqrt(2 * np.prod(decim))
                    cband = np.prod(decim) * np.fft.fftn(cband)

                    # Apply flipped window
                    imf += np.sqrt(0.5) * cband * subwin_flip.astype(cdtype)

        # Low frequency band
        imfl = np.zeros(param_udct.size, dtype=cdtype)
        decim = decimation_ratio[0][0]
        cband = upsamp(coeff[0][0][0], decim)
        cband = np.sqrt(np.prod(decim)) * np.fft.fftn(cband)
        idx, val = from_sparse_new(udctwin[0][0][0])
        imfl.flat[idx] += cband.flat[idx] * val.astype(cdtype)

        # Combine: low frequency + high frequency contributions
        imf = 2 * imf + imfl
        return np.fft.ifftn(imf).real
    else:
        # Real transform: combined +/- frequencies
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
    """
    Uniform Discrete Curvelet Transform (UDCT) implementation.

    This class provides forward and backward curvelet transforms with support
    for both real and complex transforms, as well as optional Meyer wavelet
    decomposition at the highest scale.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the input data.
    cfg : np.ndarray, optional
        Configuration array specifying the number of angular wedges per scale
        and dimension. Shape is (nscales, dim). Default creates a 2-scale
        configuration with 3 and 6 wedges.
    alpha : float, optional
        Window overlap parameter. Default is 0.15.
    r : tuple[float, float, float, float], optional
        Radial frequency parameters. Default is (pi/3, 2*pi/3, 2*pi/3, 4*pi/3).
    winthresh : float, optional
        Threshold for sparse window storage. Default is 1e-5.
    high : {"curvelet", "wavelet"}, optional
        High frequency mode. "curvelet" uses curvelets at all scales,
        "wavelet" applies Meyer wavelet decomposition at the highest scale.
        Default is "curvelet".
    complex : bool, optional
        If True, use complex transform which separates positive and negative
        frequency components into different bands. Each band is scaled by
        sqrt(0.5). If False (default), use real transform where each band
        captures both +/- frequencies combined.

    Attributes
    ----------
    shape : tuple[int, ...]
        Shape of the input data.
    high : str
        High frequency mode.
    complex : bool
        Whether complex transform is enabled.
    params : ParamUDCT
        Internal UDCT parameters.
    windows : UDCTWindows
        Curvelet windows in sparse format.
    decimation : list
        Decimation ratios for each scale/direction.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> # Create a 2D transform
    >>> transform = UDCT(shape=(64, 64))
    >>> data = np.random.randn(64, 64)
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon, atol=1e-4)
    True
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        cfg: np.ndarray | None = None,
        alpha: float = 0.15,
        r: tuple[float, float, float, float] | None = None,
        winthresh: float = 1e-5,
        high: Literal["curvelet", "wavelet"] = "curvelet",
        complex: bool = False,
    ) -> None:
        self.shape = shape
        self.high = high
        self.complex = complex
        dim = len(self.shape)
        cfg1 = np.c_[np.ones((dim,)) * 3, np.ones((dim,)) * 6].T if cfg is None else cfg

        # Validate wavelet mode requirements
        nscales = len(cfg1)
        if high == "wavelet" and nscales < 2:
            msg = "Wavelet mode requires at least 2 scales (nscales >= 2)"
            raise ValueError(msg)

        # In wavelet mode, the working size is halved (Meyer wavelet decomposition)
        if high == "wavelet":
            self._internal_shape = tuple(s // 2 for s in self.shape)
        else:
            self._internal_shape = self.shape

        r1: tuple[float, float, float, float] = (
            tuple(np.array([1.0, 2.0, 2.0, 4.0]) * np.pi / 3) if r is None else r
        )
        self.params = ParamUDCT(
            dim=dim,
            size=self._internal_shape,
            cfg=cfg1,
            alpha=alpha,
            r=r1,
            winthresh=winthresh,
        )

        self.windows, self.decimation, self.indices = udctmdwin(self.params)

        # Store wavelet coefficients between forward and backward (for reference)
        self._wavelet_bands: list[np.ndarray] = []

    def from_sparse(
        self, arr_sparse: tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        idx, val = from_sparse_new(arr_sparse)
        arr_full = np.zeros(self.params.size, dtype=val.dtype)
        arr_full.flat[idx] += val
        return arr_full

    def vect(self, coeffs: UDCTCoefficients) -> npt.NDArray[np.complexfloating]:
        coeffs_vec = []
        for c in coeffs:
            for d in c:
                for w in d:
                    coeffs_vec.append(w.ravel())
        return np.concatenate(coeffs_vec)

    def struct(self, coeffs_vec: npt.NDArray[np.complexfloating]) -> UDCTCoefficients:
        ibeg = 0
        coeffs: UDCTCoefficients = []
        internal_shape = np.array(self._internal_shape)
        for ires, decres in enumerate(self.decimation):
            coeffs.append([])
            for idir, decdir in enumerate(decres):
                coeffs[ires].append([])
                for _ in self.windows[ires][idir]:
                    shape_decim = internal_shape // decdir
                    iend = ibeg + prod(shape_decim)
                    wedge = coeffs_vec[ibeg:iend].reshape(shape_decim)
                    coeffs[ires][idir].append(wedge)
                    ibeg = iend
        return coeffs

    def forward(self, x: np.ndarray) -> UDCTCoefficients:
        """
        Apply forward curvelet transform.

        Parameters
        ----------
        x : np.ndarray
            Input data with shape matching self.shape.

        Returns
        -------
        UDCTCoefficients
            Curvelet coefficients as nested list structure.
            When complex=True, directions are doubled (first dim directions
            for positive frequencies, next dim for negative).
        """
        np.testing.assert_equal(self.shape, x.shape)

        if self.high == "wavelet":
            # Apply Meyer wavelet decomposition
            # meyerfwdmd returns 2^dim bands: first is lowpass, rest are highpass
            bands = meyerfwdmd(x)
            lowpass = bands[0]
            self._wavelet_bands = bands[1:]  # Store highpass bands for backward

            # Apply curvelet transform to lowpass only
            return udctmddec(
                lowpass,
                self.params,
                self.windows,
                self.decimation,
                complex=self.complex,
            )
        else:
            return udctmddec(
                x, self.params, self.windows, self.decimation, complex=self.complex
            )

    def backward(self, c: UDCTCoefficients) -> np.ndarray:
        """
        Apply backward curvelet transform (reconstruction).

        Parameters
        ----------
        c : UDCTCoefficients
            Curvelet coefficients from forward transform.

        Returns
        -------
        np.ndarray
            Reconstructed data with shape matching self.shape.
        """
        if self.high == "wavelet":
            # Reconstruct lowpass from curvelet coefficients
            lowpass_recon = udctmdrec(
                c, self.params, self.windows, self.decimation, complex=self.complex
            )

            # Combine with wavelet highpass bands and apply Meyer inverse
            all_bands = [lowpass_recon] + self._wavelet_bands
            return meyerinvmd(all_bands)
        else:
            return udctmdrec(
                c, self.params, self.windows, self.decimation, complex=self.complex
            )


class SimpleUDCT(UDCT):
    """
    Simplified UDCT with automatic configuration.

    This class provides a simplified interface to the UDCT, automatically
    generating the configuration based on the number of scales and bands
    per direction.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the input data.
    nscales : int, optional
        Number of scales. Must be > 1. Default is 3.
    nbands_per_direction : int, optional
        Number of angular wedges per direction at the coarsest scale.
        The number of wedges doubles at each finer scale. Must be >= 3.
        Default is 3.
    alpha : float, optional
        Window overlap parameter. If None, automatically chosen based on
        nbands_per_direction. Default is None.
    winthresh : float, optional
        Threshold for sparse window storage. Default is 1e-5.
    high : {"curvelet", "wavelet"}, optional
        High frequency mode. Default is "curvelet".
    complex : bool, optional
        If True, use complex transform (separate +/- frequency bands).
        If False, use real transform. Default is False.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import SimpleUDCT
    >>> transform = SimpleUDCT(shape=(64, 64), nscales=3, nbands_per_direction=3)
    >>> data = np.random.randn(64, 64)
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon, atol=1e-4)
    True
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        nscales: int = 3,
        nbands_per_direction: int = 3,
        alpha: float | None = None,
        winthresh: float = 1e-5,
        high: Literal["curvelet", "wavelet"] = "curvelet",
        complex: bool = False,
    ) -> None:
        assert nscales > 1
        assert nbands_per_direction >= 3

        # Validate wavelet mode requirements
        if high == "wavelet" and nscales < 2:
            msg = "Wavelet mode requires at least 2 scales (nscales >= 2)"
            raise ValueError(msg)

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
            if (const := 2 ** (i / nb) * (1 + 2 * alpha) * (1 + alpha)) >= nb:
                msg = (
                    f"alpha={alpha:.3f} does not respect the relationship "
                    f"(2^{i}/{nb})(1+2α)(1+α) = {const:.3f} = < 1 for scale {i + 1}"  # noqa: RUF001
                )
                logging.warning(msg)
        cfg = np.tile(nbands[:, None], dim)
        r: tuple[float, float, float, float] = tuple(
            np.array([1.0, 2.0, 2.0, 4.0]) * np.pi / 3
        )

        super().__init__(
            shape=shape,
            cfg=cfg,
            alpha=alpha,
            r=r,
            winthresh=winthresh,
            high=high,
            complex=complex,
        )
