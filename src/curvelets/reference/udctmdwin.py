from __future__ import annotations

from copy import deepcopy
from itertools import combinations

import numpy as np

from ._adapt_grid import adapt_grid
from ._angle_fun import angle_fun
from ._angle_kron import angle_kron
from ._circshift import circshift
from ._fftflip import fftflip
from ._fun_meyer import fun_meyer
from ._travel import travel
from .udct import ParamUDCT


def _to_sparse(arr, thresh):
    idx = np.argwhere(travel(arr) > thresh)
    return np.c_[idx + 1, travel(arr)[idx]]


def udctmdwin(param_udct: ParamUDCT) -> dict:
    Sgrid = {}
    f1d = {}
    for ind in range(1, param_udct.dim + 1):
        start = -1.5 * np.pi
        stop = 0.5 * np.pi  # - np.pi / (param_udct.size[ind - 1] / 2)
        # step = np.pi / (param_udct.size[ind - 1] / 2)
        # np.arange(start, stop + step, step)
        Sgrid[ind] = np.linspace(start, stop, param_udct.size[ind - 1], endpoint=False)

        params = np.array([-2, -1, *param_udct.r[:2]])
        f1d[(param_udct.res, ind)] = fun_meyer(np.abs(Sgrid[ind]), *params)
        if param_udct.res == 1:
            f1d[(param_udct.res, ind)] += fun_meyer(
                np.abs(Sgrid[ind] + 2 * np.pi), *params
            )
        params[2:] = param_udct.r[2:]
        f1d[(param_udct.res + 1, ind)] = fun_meyer(abs(Sgrid[ind]), *params)

        for jn in range(param_udct.res - 1, 0, -1):
            params[2:] = param_udct.r[:2]
            params[2:] /= 2 ** (param_udct.res - jn)
            f1d[(jn, ind)] = fun_meyer(abs(Sgrid[ind]), *params)

    F2d = {}
    for jn in range(param_udct.res, 0, -1):
        fltmp = np.array([1.0])
        fhtmp = np.array([1.0])
        for ind in range(param_udct.dim, 0, -1):
            fltmp = np.kron(fltmp.ravel(), f1d[(jn, ind)].ravel())
            fhtmp = np.kron(fhtmp.ravel(), f1d[(jn + 1, ind)].ravel())
        FL = fltmp.reshape(*param_udct.size[::-1]).T.astype(float).copy()
        FH = fhtmp.reshape(*param_udct.size[::-1]).T.astype(float).copy()
        FP = FH - FL
        FP[FP < 0] = 0
        F2d[jn + 1] = FP
    F2d[1] = FL.copy()

    Winlow = circshift(np.sqrt(F2d[1]), tuple(s // 4 for s in param_udct.size))
    # convert to sparse format
    udctwin = {}
    udctwin[1] = {}
    udctwin[1][1] = _to_sparse(Winlow, param_udct.winthresh)

    param_udct.ind = {}
    param_udct.ind[1] = {}
    param_udct.ind[1][1] = np.zeros((1, 1), dtype=int)
    # every combination of 2 dimension out of 1:dim
    mperms = np.asarray(list(combinations(np.arange(1, param_udct.dim + 1), 2)))
    M = {}
    for ind in range(1, 1 + len(mperms)):
        M[(ind, 1)], M[ind, 2] = adapt_grid(
            Sgrid[mperms[ind - 1, 0]], Sgrid[mperms[ind - 1, 1]]
        )

    # gather angle function for each pyramid

    Mdir = {}
    Mang = {}
    Mang_in = {}
    for res in range(1, param_udct.res + 1):
        Mang[res] = {}
        Mang_in[res] = {}

        Mdir[res] = np.zeros((param_udct.dim, param_udct.dim - 1), dtype=int)
        # for each resolution
        for ind in range(1, param_udct.dim + 1):
            # for each pyramid in resolution res
            cnt = 1
            # cnt is number of angle function required for each pyramid
            # now loop through mperms
            Mdir[res][ind - 1, :] = np.array(
                list(range(1, ind)) + list(range(ind + 1, param_udct.dim + 1)),
                dtype=int,
            )
            # Mdir is dimension of need to calculate angle function on each
            # hyperpyramid
            for hp in range(1, mperms.shape[0] + 1):
                for ndir in range(1, 3):
                    # Fill with zeros, will be replaced
                    # Mang[res][(ind, cnt)] = np.zeros(param_udct.size)
                    if mperms[hp - 1, ndir - 1] == ind:
                        tmp = angle_fun(
                            M[(hp, ndir)],
                            ndir,
                            param_udct.cfg[res - 1, mperms[hp - 1, 3 - ndir - 1] - 1],
                            param_udct.alpha,
                        )
                        Mang[res][(ind, cnt)] = tmp
                        Mang_in[res][(ind, cnt)] = mperms[hp - 1, 1 - 1 : 2]
                        cnt += 1

    # Mang is 1-d angle function for each hyper pyramid (row) and each angle
    # dimension (column)
    for res in range(1, param_udct.res + 1):
        # for each resolution
        subband = {}
        udctwin[res + 1] = {}
        param_udct.ind[res + 1] = {}
        for in1 in range(1, param_udct.dim + 1):
            udctwin[res + 1][in1] = {}
            # for each hyperpyramid
            ang_in = 1
            for in2 in range(1, param_udct.dim - 1 + 1):
                ln = len(Mang[res][(in1, in2)])
                tmp2 = np.arange(ln, dtype=int)[:, None] + 1
                if in2 == 1:
                    ang_in = tmp2
                else:
                    ln2 = ln * len(ang_in)
                    tmp3 = np.kron(ang_in, np.ones((ln, 1), dtype=int))
                    tmp4 = np.kron(np.ones((ang_in.shape[0], 1), dtype=int), tmp2)
                    ang_in = np.c_[tmp3, tmp4]
            lent = ang_in.shape[0]
            ang_inmax = param_udct.cfg[res - 1, Mdir[res][in1 - 1, :] - 1]
            # lent is the smallest number of windows need to calculated on each
            # pyramid
            # ang_inmax is M-1 vector contain number of angle function per each
            # dimension of the hyperpyramid
            ang_ind = 0
            ind = 1
            for in3 in range(1, lent + 1):
                # for each calculated windows function, estimated all the other
                # flipped window functions
                afun = np.ones(param_udct.size, dtype=float)
                afunin = 1
                for in4 in range(1, param_udct.dim - 1 + 1):
                    idx = ang_in.reshape(len(ang_in), -1)[in3 - 1, in4 - 1]
                    tmp = Mang[res][(in1, in4)][idx - 1]
                    # print(f"{tmp.shape}")
                    tmp2 = Mang_in[res][(in1, in4)]
                    afun2 = angle_kron(tmp, tmp2, param_udct)
                    afun *= afun2
                aafun = {}
                ang_in2 = None
                afun = afun * F2d[res + 1]
                afun = np.sqrt(circshift(afun, tuple(s // 4 for s in param_udct.size)))

                # first windows function
                aafun[afunin] = afun

                # index of current angle
                ang_in2 = ang_in[in3 - 1 : in3, :].copy()
                # print(f"{ang_in2.shape=}")

                # all possible flip along different dimension
                for in5 in range(param_udct.dim - 1, 0, -1):
                    lentmp = ang_in2.shape[0]
                    for in6 in range(1, lentmp + 1):
                        if 2 * ang_in2[in6 - 1, in5 - 1] <= ang_inmax[in5 - 1]:
                            ang_in2tmp = ang_in2[in6 - 1 : in6, :].copy()
                            ang_in2tmp[0, in5 - 1] = (
                                ang_inmax[in5 - 1] + 1 - ang_in2[in6 - 1, in5 - 1]
                            )
                            # print(f"{ang_in2.shape=}")
                            # print(f"{ang_in2tmp.shape=}")
                            ang_in2 = np.concatenate((ang_in2, ang_in2tmp), axis=0)
                            # print(f"{ang_in2.shape=}")
                            a = aafun[in6]
                            b = Mdir[res][in1 - 1, in5 - 1]
                            end = max(aafun.keys())
                            aafun[end + 1] = fftflip(a, b - 1)
                #    Mwin[tmp[in3-1,:]] = afun
                # subband[ang]
                aafun = np.concatenate(
                    [aafun[k][None, ...] for k in sorted(aafun.keys())], axis=0
                )
                if isinstance(ang_ind, int) and ang_ind == 0:
                    ang_ind = ang_in2
                    # subband[in1] = aafun.copy()
                    for in7 in range(1, ang_ind.shape[0] + 1):
                        # convert to sparse format
                        udctwin[res + 1][in1][in7] = _to_sparse(
                            aafun[in7 - 1], param_udct.winthresh
                        )
                else:
                    inold = ang_ind.shape[0]
                    ang_ind = np.concatenate((ang_ind, ang_in2), axis=0)
                    innew = ang_ind.shape[0]
                    # note:
                    # subband has to be expanded to accommodate new indices
                    # matlab does that automatically, but numpy does not
                    # the following line DOES NOT WORK
                    # subband[in1][inold + 1 - 1 : innew] = aafun
                    for in7 in range(inold + 1, innew + 1):
                        in8 = in7 - inold
                        udctwin[res + 1][in1][in7] = _to_sparse(
                            aafun[in8 - 1], param_udct.winthresh
                        )
                    param_udct.ind[res + 1][in1] = ang_ind.copy()

    sumw2 = np.zeros(param_udct.size)
    idx = udctwin[1][1][:, 0].astype(int) - 1
    val = udctwin[1][1][:, 1]
    sumw2.T.flat[idx] += val.T.ravel() ** 2
    for res in range(1, param_udct.res + 1):
        for dir in range(1, param_udct.dim + 1):
            for ang in range(1, len(udctwin[res + 1][dir]) + 1):
                tmpw = np.zeros(param_udct.size)
                idx = udctwin[res + 1][dir][ang][:, 0].astype(int) - 1
                val = udctwin[res + 1][dir][ang][:, 1]
                tmpw.T.flat[idx] += val.T.ravel() ** 2
                sumw2 += tmpw
                tmpw = fftflip(tmpw, dir - 1)
                sumw2 += tmpw

    sumw2 = np.sqrt(sumw2)
    idx = udctwin[1][1][:, 0].astype(int) - 1
    udctwin[1][1][:, 1] /= sumw2.T.ravel()[idx]
    for res in range(1, param_udct.res + 1):
        for dir in range(1, param_udct.dim + 1):
            for ang in range(1, len(udctwin[res + 1][dir]) + 1):
                idx = udctwin[res + 1][dir][ang][:, 0].astype(int) - 1
                val = udctwin[res + 1][dir][ang][:, 1]
                udctwin[res + 1][dir][ang][:, 1] /= sumw2.T.ravel()[idx]

    # decimation ratio for each band
    param_udct.dec = {}
    for res in range(1, param_udct.res + 1):
        tmp = np.ones((param_udct.dim, param_udct.dim))
        param_udct.dec[res] = 2.0 ** (param_udct.res - res + 1) * tmp
        for ind in range(1, param_udct.dim + 1):
            ind2 = Mdir[res][ind - 1, :] - 1
            ind3 = Mdir[res][ind - 1, :] - 1
            param_udct.dec[res][ind - 1, ind2] = (
                2.0 ** (param_udct.res - res) * 2 * param_udct.cfg[res - 1, ind3] / 3
            )

    param_udct.len_r = param_udct.len / 2.0 ** ((param_udct.res - 1) * param_udct.dim)
    for res in range(1, param_udct.res + 1):
        for ind in range(1, param_udct.dim + 1):
            a = udctwin[res + 1][ind]
            b = param_udct.dec[res][ind - 1, :]
            param_udct.len_r += len(a) * 2 * param_udct.len / np.prod(b)

    # sort the window
    newwin = {}
    for res in range(2, param_udct.res + 1 + 1):
        for pyr in range(1, param_udct.dim + 1):
            # take out the angle index list
            mlist = param_udct.ind[res][pyr].copy()

            # map it to a number
            mult = 1
            nlist = np.zeros((mlist.shape[0], 1))
            for d in range(mlist.shape[1], 0, -1):
                for b in range(1, mlist.shape[0] + 1):
                    nlist[b - 1] += mult * mlist[b - 1, d - 1]
                mult *= 100
            ix = np.argsort(nlist, axis=0) + 1
            # b = nlist[ix]

            newind = mlist.copy()
            for b in range(1, mlist.shape[0] + 1):
                newind[b - 1, :] = mlist[ix[b - 1] - 1, :].copy()
                newwin[b] = udctwin[res][pyr][ix[b - 1].item()].copy()

            param_udct.ind[res][pyr] = newind.copy()
            udctwin[res][pyr] = newwin.copy()

    return udctwin
