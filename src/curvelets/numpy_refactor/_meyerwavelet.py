from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .utils import fun_meyer


def meyer_wavelet(N: int) -> tuple[npt.NDArray, npt.NDArray]:
    step = 2 * np.pi / N
    x = np.linspace(0, 2 * np.pi - step, N) - np.pi / 2
    prm = np.pi * np.array([-1 / 3, 1 / 3, 2 / 3, 4 / 3])
    f1 = np.sqrt(np.fft.fftshift(fun_meyer(x, prm[0], prm[1], prm[2], prm[3])))
    f2 = np.sqrt(fun_meyer(x, prm[0], prm[1], prm[2], prm[3]))
    return f1, f2


def meyerfwd1d(img: npt.NDArray, dim: int) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Apply 1D Meyer wavelet forward transform along specified dimension.

    Parameters
    ----------
    img : npt.NDArray
        Input array (real or complex).
    dim : int
        Dimension along which to apply the transform.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
        Lowpass (h1) and highpass (h2) subbands. Output dtype matches input:
        real input produces real output, complex input produces complex output.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy.meyerwavelet import meyerfwd1d
    >>> img = np.random.randn(64, 64)
    >>> h1, h2 = meyerfwd1d(img, 0)
    >>> h1.shape
    (32, 64)
    >>> h2.shape
    (32, 64)
    """
    ldim = img.ndim - 1
    img = np.swapaxes(img, dim, ldim)
    sp = img.shape
    N = sp[-1]
    f1, f2 = meyer_wavelet(N)
    f1 = np.reshape(f1, (1, N))
    f2 = np.reshape(f2, (1, N))

    imgf = np.fft.fft(img, axis=ldim)
    h1_full = np.fft.ifft(f1 * imgf, axis=ldim)
    h2_full = np.fft.ifft(f2 * imgf, axis=ldim)

    # Preserve complex values for complex input, take real for real input
    if not np.iscomplexobj(img):
        h1_full = h1_full.real
        h2_full = h2_full.real

    h1 = h1_full[..., ::2]
    h2 = h2_full[..., 1::2]
    h1 = np.swapaxes(h1, dim, ldim)
    h2 = np.swapaxes(h2, dim, ldim)

    return h1, h2


def meyerinv1d(h1: npt.NDArray, h2: npt.NDArray, dim: int) -> npt.NDArray:
    """
    Apply 1D Meyer wavelet inverse transform along specified dimension.

    Parameters
    ----------
    h1 : npt.NDArray
        Lowpass subband (real or complex).
    h2 : npt.NDArray
        Highpass subband (real or complex).
    dim : int
        Dimension along which to apply the transform.

    Returns
    -------
    npt.NDArray
        Reconstructed array. Output dtype matches input: real input produces
        real output, complex input produces complex output.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy.meyerwavelet import meyerfwd1d, meyerinv1d
    >>> img = np.random.randn(64, 64)
    >>> h1, h2 = meyerfwd1d(img, 0)
    >>> recon = meyerinv1d(h1, h2, 0)
    >>> np.allclose(img, recon, atol=1e-10)
    True
    """
    ldim = h1.ndim - 1
    h1 = np.swapaxes(h1, dim, ldim)
    h2 = np.swapaxes(h2, dim, ldim)

    sp = list(h1.shape)
    sp[-1] = 2 * sp[-1]

    # Use appropriate dtype for complex or real input
    is_complex = np.iscomplexobj(h1) or np.iscomplexobj(h2)
    dtype = h1.dtype if is_complex else float

    g1 = np.zeros(sp, dtype=dtype)
    g2 = np.zeros(sp, dtype=dtype)
    g1[..., ::2] = h1
    g2[..., 1::2] = h2
    N = sp[-1]
    f1, f2 = meyer_wavelet(N)
    f1 = np.reshape(f1, (1, N))
    f2 = np.reshape(f2, (1, N))
    imfsum = f1 * np.fft.fft(g1, axis=ldim) + f2 * np.fft.fft(g2, axis=ldim)
    imrecon_full = np.fft.ifft(imfsum, axis=ldim)

    # Preserve complex values for complex input, take real for real input
    if is_complex:
        imrecon = 2 * imrecon_full
    else:
        imrecon = 2 * imrecon_full.real

    return np.swapaxes(imrecon, dim, ldim)


def meyerfwdmd(img: npt.NDArray) -> list[npt.NDArray]:
    """
    Apply multi-dimensional Meyer wavelet forward transform.

    Parameters
    ----------
    img : npt.NDArray
        Input array (real or complex).

    Returns
    -------
    list[npt.NDArray]
        List of 2^dim subbands. First is lowpass, rest are highpass.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy.meyerwavelet import meyerfwdmd
    >>> img = np.random.randn(64, 64)
    >>> bands = meyerfwdmd(img)
    >>> len(bands)
    4
    >>> bands[0].shape
    (32, 32)
    """
    band = [img]
    dim = len(img.shape)
    for i in range(dim):
        cband = []
        for j in range(len(band)):
            h1, h2 = meyerfwd1d(band[j], i)
            cband.append(h1)
            cband.append(h2)
        band = cband
    return cband


def meyerinvmd(band: list[npt.NDArray]) -> npt.NDArray:
    """
    Apply multi-dimensional Meyer wavelet inverse transform.

    Parameters
    ----------
    band : list[npt.NDArray]
        List of 2^dim subbands from meyerfwdmd.

    Returns
    -------
    npt.NDArray
        Reconstructed array.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy.meyerwavelet import meyerfwdmd, meyerinvmd
    >>> img = np.random.randn(64, 64)
    >>> bands = meyerfwdmd(img)
    >>> recon = meyerinvmd(bands)
    >>> np.allclose(img, recon, atol=1e-10)
    True
    """
    dim = len(band[0].shape)
    for i in range(dim - 1, -1, -1):
        cband = []
        for j in range(len(band) // 2):
            imrecon = meyerinv1d(band[2 * j], band[2 * j + 1], i)
            cband.append(imrecon)
        band = cband
    return band[0]

