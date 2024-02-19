from __future__ import annotations

__all__ = ["udctmdwin"]

from itertools import combinations

# import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    ParamUDCT,
    adapt_grid,
    angle_fun,
    angle_kron,
    circshift,
    fftflip,
    fun_meyer,
    to_sparse,
)


def _create_bandpass_windows(
    nscales: int, shape: tuple[int, ...], r: tuple[float, float, float, float]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    dim = len(shape)
    shape_grid: dict[int, np.ndarray] = {}
    meyers: dict[tuple[int, int], np.ndarray] = {}
    for ind in range(dim):
        # Don't take the np.pi out of the linspace
        shape_grid[ind] = np.linspace(
            -1.5 * np.pi, 0.5 * np.pi, shape[ind], endpoint=False
        )

        params = np.array([-2, -1, *r[:2]])
        abs_shape_grid = np.abs(shape_grid[ind])
        meyers[(nscales, ind)] = fun_meyer(abs_shape_grid, *params)
        if nscales == 1:
            meyers[(nscales, ind)] += fun_meyer(
                np.abs(shape_grid[ind] + 2 * np.pi), *params
            )
        params[2:] = r[2:]
        meyers[(nscales + 1, ind)] = fun_meyer(abs_shape_grid, *params)

        for jn in range(nscales - 1, 0, -1):
            params[2:] = r[:2]
            params[2:] /= 2 ** (nscales - jn)
            meyers[(jn, ind)] = fun_meyer(abs_shape_grid, *params)

    bandpasses: dict[int, np.ndarray] = {}
    for jn in range(nscales, 0, -1):
        lo = np.array([1.0])
        hi = np.array([1.0])
        for ind in range(dim - 1, -1, -1):
            lo = np.kron(meyers[(jn, ind)], lo)
            hi = np.kron(meyers[(jn + 1, ind)], hi)
        lo_nd = lo.reshape(*shape)
        hi_nd = hi.reshape(*shape)
        bp_nd = hi_nd - lo_nd
        bp_nd[bp_nd < 0] = 0
        bandpasses[jn] = bp_nd
    bandpasses[0] = lo_nd
    return shape_grid, bandpasses


def udctmdwin(
    param_udct: ParamUDCT,
) -> dict[int, dict[int, dict[int, np.ndarray]]]:
    Sgrid, F2d = _create_bandpass_windows(
        nscales=param_udct.res, shape=param_udct.size, r=param_udct.r
    )
    Winlow = circshift(np.sqrt(F2d[0]), tuple(s // 4 for s in param_udct.size))

    # convert to sparse format
    udctwin = {}
    udctwin[1] = {}
    udctwin[1][1] = {}
    udctwin[1][1][1] = to_sparse(Winlow, param_udct.winthresh)

    # `indices` gets stored as `param_udct.ind` in the original.
    indices = {}
    indices[1] = {}
    indices[1][1] = {}
    indices[1][1][1] = np.zeros((1, 1), dtype=int)
    # every combination of 2 dimension out of 1:dim
    mperms = np.asarray(list(combinations(np.arange(1, param_udct.dim + 1), 2)))
    M = {}
    for ind in range(len(mperms)):
        M[(ind + 1, 1)], M[ind + 1, 2] = adapt_grid(
            Sgrid[mperms[ind, 0] - 1], Sgrid[mperms[ind, 1] - 1]
        )

    # gather angle function for each pyramid
    Mdir = {}
    Mang = {}
    Mang_in = {}
    for res in range(param_udct.res):
        Mang[res + 1] = {}
        Mang_in[res + 1] = {}

        Mdir[res + 1] = np.zeros((param_udct.dim, param_udct.dim - 1), dtype=int)
        # for each resolution
        for ind in range(param_udct.dim):
            # for each pyramid in resolution res
            cnt = 1
            # cnt is number of angle function required for each pyramid
            # now loop through mperms
            Mdir[res + 1][ind, :] = (
                1 + np.r_[range(ind), range(ind + 1, param_udct.dim)]
            )
            # Mdir is dimension of need to calculate angle function on each
            # hyperpyramid
            for hp in range(mperms.shape[0]):
                for ndir in range(2):
                    # Fill with zeros, will be replaced
                    # Mang[res+1][(ind+1, cnt)] = np.zeros(param_udct.size)
                    if mperms[hp, ndir] == ind + 1:
                        tmp = angle_fun(
                            M[(hp + 1, ndir + 1)],
                            ndir + 1,
                            param_udct.cfg[res, mperms[hp, 1 - ndir] - 1],
                            param_udct.alpha,
                        )
                        Mang[res + 1][(ind + 1, cnt)] = tmp
                        Mang_in[res + 1][(ind + 1, cnt)] = mperms[hp, :2]
                        cnt += 1

    # Mang is 1-d angle function for each hyper pyramid (row) and each angle
    # dimension (column)
    for res in range(1, param_udct.res + 1):
        # for each resolution
        udctwin[res + 1] = {}
        indices[res + 1] = {}
        for in1 in range(param_udct.dim):
            udctwin[res + 1][in1 + 1] = {}
            # for each hyperpyramid
            ang_in: int | np.ndarray = 1
            for in2 in range(1, param_udct.dim - 1 + 1):
                ln = len(Mang[res][(in1 + 1, in2)])
                tmp2 = np.arange(ln, dtype=int)[:, None] + 1
                if in2 == 1:
                    ang_in = tmp2
                else:
                    tmp3 = np.kron(ang_in, np.ones((ln, 1), dtype=int))
                    tmp4 = np.kron(np.ones((ang_in.shape[0], 1), dtype=int), tmp2)
                    ang_in = np.c_[tmp3, tmp4]
            lent = ang_in.shape[0]
            ang_inmax = param_udct.cfg[res - 1, Mdir[res][in1, :] - 1]
            # lent is the smallest number of windows need to calculated on each
            # pyramid
            # ang_inmax is M-1 vector contain number of angle function per each
            # dimension of the hyperpyramid
            ang_ind = 0
            ind = 1
            for in3 in range(lent):
                # for each calculated windows function, estimated all the other
                # flipped window functions
                afun = np.ones(param_udct.size, dtype=float)
                afunin = 1
                for in4 in range(param_udct.dim - 1):
                    idx = ang_in.reshape(len(ang_in), -1)[in3, in4]
                    tmp = Mang[res][(in1 + 1, in4 + 1)][idx - 1]
                    # print(f"{tmp.shape}")
                    tmp2 = Mang_in[res][(in1 + 1, in4 + 1)]
                    afun2 = angle_kron(tmp, tmp2, param_udct)
                    afun *= afun2
                aafun = {}
                ang_in2 = None
                afun = afun * F2d[res]
                afun = np.sqrt(circshift(afun, tuple(s // 4 for s in param_udct.size)))

                # first windows function
                aafun[afunin] = afun

                # index of current angle
                ang_in2 = ang_in[in3 : in3 + 1, :]
                # print(f"{ang_in2.shape=}")

                # all possible flip along different dimension
                for in5 in range(param_udct.dim - 2, -1, -1):
                    for in6 in range(ang_in2.shape[0]):
                        if 2 * ang_in2[in6, in5] <= ang_inmax[in5]:
                            ang_in2tmp = ang_in2[in6 : in6 + 1, :].copy()
                            ang_in2tmp[0, in5] = ang_inmax[in5] + 1 - ang_in2[in6, in5]
                            ang_in2 = np.concatenate((ang_in2, ang_in2tmp), axis=0)
                            a = aafun[in6 + 1]
                            b = Mdir[res][in1, in5]
                            end = max(aafun.keys())
                            aafun[end + 1] = fftflip(a, b - 1)
                aafun = np.concatenate(
                    [aafun[k][None, ...] for k in sorted(aafun.keys())], axis=0
                )
                if isinstance(ang_ind, int) and ang_ind == 0:
                    ang_ind = ang_in2
                    for in7 in range(ang_ind.shape[0]):
                        # convert to sparse format
                        udctwin[res + 1][in1 + 1][in7 + 1] = to_sparse(
                            aafun[in7], param_udct.winthresh
                        )
                else:
                    inold = ang_ind.shape[0]
                    ang_ind = np.concatenate((ang_ind, ang_in2), axis=0)
                    innew = ang_ind.shape[0]
                    for in7 in range(inold, innew):
                        in8 = in7 - inold
                        udctwin[res + 1][in1 + 1][in7 + 1] = to_sparse(
                            aafun[in8], param_udct.winthresh
                        )
                    indices[res + 1][in1 + 1] = ang_ind.copy()

    sumw2 = np.zeros(param_udct.size)
    idx = udctwin[1][1][1][:, 0].astype(int) - 1
    val = udctwin[1][1][1][:, 1]
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
    idx = udctwin[1][1][1][:, 0].astype(int) - 1
    udctwin[1][1][1][:, 1] /= sumw2.T.ravel()[idx]
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

    # sort the window
    newwin = {}
    for res in range(2, param_udct.res + 1 + 1):
        for pyr in range(1, param_udct.dim + 1):
            # take out the angle index list
            mlist = indices[res][pyr].copy()

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

            indices[res][pyr] = newind.copy()
            udctwin[res][pyr] = newwin.copy()

    return udctwin
