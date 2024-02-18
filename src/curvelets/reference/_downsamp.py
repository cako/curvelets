from __future__ import annotations

from math import prod

import numpy as np


def downsamp(F, decim):
    assert F.ndim == len(decim)
    D = F[tuple(slice(None, None, d) for d in decim)].copy()
    return D


def downsamp_old_not_working(F, decim):
    # Downsampling F in decim
    # downsamp can work in any dimension
    # decim must be vector equal to dimension of F
    sz = F.shape
    # max value
    ms = prod(sz)
    indw = 1 + np.arange(ms)

    dim = len(sz)
    mask = np.zeros((ms, dim))

    for in1 in range(1, dim + 1):
        if in1 == 1:
            step = decim[in1 - 1]
            template = np.zeros((step, 1))
            template.flat[0] = 1
        else:
            step = decim[in1 - 1]
            tmp = np.zeros((step, 1))
            tmp.flat[0] = 1
            tmp2 = np.ones((prod(sz[: in1 - 1]), 1))
            template = np.kron(tmp, tmp2)

        kronsz = np.ones((ms // len(template), 1))
        maskcol = np.kron(kronsz, template)
        mask[:, in1 - 1 : in1] = maskcol

    maskp = np.prod(mask, axis=1)
    idx = np.nonzero(maskp.ravel())[0]
    indwf = indw.flat[idx]

    D = np.reshape(F.flat[indwf - 1], sz // decim)
    return D


def downsamp_save(F, decim):
    assert F.ndim == len(decim)
    # Downsampling F in decim
    D = F[tuple(slice(None, None, d) for d in decim)].copy()
    return D
