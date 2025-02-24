from __future__ import annotations

import math

import numpy as np


def fun_meyer(x, param):
    """
    Return a smooth window similar to Meyer wavelet
    x is the grid generated by linespace
    param is a vector of four incrasing values. The window is zero outsize
    param[0] and param[3] and one between param[1] and param[2]. Its changing
    smoothly from 0 to 1 and 1 to 0 between param[0] and param[1], and
    param[2] and param[3].
    """

    p = np.array([-20, 70, -84, 35, 0, 0, 0, 0])
    # x = np.linspace(0,5)
    y = np.ones_like(x)

    y[x <= param[0]] = 0.0
    y[x >= param[3]] = 0.0
    xx = (x[(x >= param[0]) & (x <= param[1])] - param[0]) / (param[1] - param[0])
    y[(x >= param[0]) & (x <= param[1])] = np.polyval(p, xx)
    xx = (x[(x >= param[2]) & (x <= param[3])] - param[3]) / (param[2] - param[3])
    y[(x >= param[2]) & (x <= param[3])] = np.polyval(p, xx)
    return y.reshape(x.shape)


def bands2vec(imband):
    """
    Convert dictionary of subband to a vector
    """
    compressed = np.real(imband[(0,)].flatten())
    # ucurv.imSz[0] = imband[0].shape
    for id, subwin in imband.items():
        if id == (0,):
            continue
        # ucurv.imSz[id] = subwin.shape
        a = np.real(subwin.flatten())
        b = np.imag(subwin.flatten())
        c = [item for pair in zip(a, b) for item in pair]
        compressed = np.concatenate((compressed, np.array(c)))
    return compressed


def vec2bands(imband, udct):
    """
    Convert vector banck to a dictionary of subband
    """
    imSz = np.array(udct.sz) // 2 ** (udct.res - 1)
    # first is the low band
    uncompressed = {(0,): np.reshape(imband[: np.prod(imSz)], imSz)}
    p = np.prod(imSz)
    for id in udct.Msubwin.keys():
        # if id == 0: continue
        imSz = udct.sz // udct.Sampling[(id[0], id[1])]
        c = imband[p : p + 2 * np.prod(imSz)]
        c = [complex(c[i], c[i + 1]) for i in range(0, len(c), 2)]
        uncompressed[id] = np.reshape(c, imSz)
        p += 2 * np.prod(imSz)
    return uncompressed


def ucurv2d_show(imband, udct):
    if udct.dim != 2:
        raise Exception(" ucurv2d_show only work with 2D transform")
    cfg = udct.cfg
    imlist = []
    res = udct.res
    sz = udct.sz
    for rs in range(res):
        dirim = []
        for dir in [0, 1]:
            bandlist = [imband[(rs, dir, i)] for i in range(cfg[rs][dir])]
            dirim.append(np.concatenate(bandlist, axis=1 - dir))

        sp = dirim[1].shape
        sp0 = sp[0] // 3
        d1 = np.concatenate(
            [dirim[1][:sp0, :], dirim[1][sp0 : 2 * sp0, :], dirim[1][2 * sp0 :, :]],
            axis=1,
        )
        dimg = np.concatenate([dirim[0], d1], axis=0)
        dshape = dimg.shape
        dimg2 = np.zeros((sz[0], np.max(dshape)), dtype=complex)
        dimg2[: dshape[0], : dshape[1]] = dimg
        imlist.append(dimg2)

    dimg2 = np.concatenate(imlist, axis=1)
    lbshape = imband[(0,)].shape
    iml = np.zeros((sz[0], lbshape[1]), dtype=complex)
    iml[: lbshape[0], :] = imband[(0,)]
    dimg3 = np.concatenate([iml, dimg2], axis=1)
    return dimg3
