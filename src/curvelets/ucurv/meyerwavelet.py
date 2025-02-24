from __future__ import annotations

import numpy as np

from .util import fun_meyer


def meyer_wavelet(N):
    step = 2 * np.pi / N
    x = np.linspace(0, 2 * np.pi - step, N) - np.pi / 2
    prm = np.pi * np.array([-1 / 3, 1 / 3, 2 / 3, 4 / 3])
    f1 = np.sqrt(np.fft.fftshift(fun_meyer(x, prm)))
    f2 = np.sqrt(fun_meyer(x, prm))
    return f1, f2


def meyerfwd1d(img, dim):
    ldim = img.ndim - 1
    img = np.swapaxes(img, dim, ldim)
    sp = img.shape
    N = sp[-1]
    f1, f2 = meyer_wavelet(N)
    f1 = np.reshape(f1, (1, N))
    f2 = np.reshape(f2, (1, N))

    imgf = np.fft.fft(img, axis=ldim)
    h1 = np.real(np.fft.ifft(f1 * imgf, axis=ldim))[..., ::2]
    h2 = np.real(np.fft.ifft(f2 * imgf, axis=ldim))[..., 1::2]
    h1 = np.swapaxes(h1, dim, ldim)
    h2 = np.swapaxes(h2, dim, ldim)

    return h1, h2


def meyerinv1d(h1, h2, dim):
    ldim = h1.ndim - 1
    h1 = np.swapaxes(h1, dim, ldim)
    h2 = np.swapaxes(h2, dim, ldim)

    sp = list(h1.shape)
    sp[-1] = 2 * sp[-1]
    g1 = np.zeros(sp)
    g2 = np.zeros(sp)
    g1[..., ::2] = h1
    g2[..., 1::2] = h2
    N = sp[-1]
    f1, f2 = meyer_wavelet(N)
    f1 = np.reshape(f1, (1, N))
    f2 = np.reshape(f2, (1, N))
    imfsum = f1 * np.fft.fft(g1, axis=ldim) + f2 * np.fft.fft(g2, axis=ldim)
    imrecon = 2 * np.real(np.fft.ifft(imfsum, axis=ldim))
    imrecon = np.swapaxes(imrecon, dim, ldim)
    return imrecon


def meyerfwdmd(img):
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


def meyerinvmd(band):
    dim = len(band[0].shape)
    for i in range(dim - 1, -1, -1):
        cband = []
        for j in range(len(band) // 2):
            imrecon = meyerinv1d(band[2 * j], band[2 * j + 1], i)
            cband.append(imrecon)
        band = cband
    return band[0]
