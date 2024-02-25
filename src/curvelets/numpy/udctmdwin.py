from __future__ import annotations

__all__ = ["udctmdwin"]
from itertools import combinations
from typing import Any

# import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .typing import UDCTWindows
from .utils import (
    ParamUDCT,
    adapt_grid,
    angle_fun,
    angle_kron,
    circshift,
    fftflip,
    from_sparse_new,
    fun_meyer,
    to_sparse_new,
)


def _create_bandpass_windows(
    nscales: int, shape: tuple[int, ...], r: tuple[float, float, float, float]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    dim = len(shape)
    shape_grid: dict[int, np.ndarray] = {}
    meyers: dict[tuple[int, int], np.ndarray] = {}
    for idim in range(dim):
        # Don't take the np.pi out of the linspace
        shape_grid[idim] = np.linspace(
            -1.5 * np.pi, 0.5 * np.pi, shape[idim], endpoint=False
        )

        params = np.array([-2, -1, *r[:2]])
        abs_shape_grid = np.abs(shape_grid[idim])
        meyers[(nscales, idim)] = fun_meyer(abs_shape_grid, *params)
        if nscales == 1:
            meyers[(nscales, idim)] += fun_meyer(
                np.abs(shape_grid[idim] + 2 * np.pi), *params
            )
        params[2:] = r[2:]
        meyers[(nscales + 1, idim)] = fun_meyer(abs_shape_grid, *params)

        for jn in range(nscales - 1, 0, -1):
            params[2:] = r[:2]
            params[2:] /= 2 ** (nscales - jn)
            meyers[(jn, idim)] = fun_meyer(abs_shape_grid, *params)

    bandpasses: dict[int, np.ndarray] = {}
    for jn in range(nscales, 0, -1):
        lo = np.array([1.0])
        hi = np.array([1.0])
        for idim in range(dim - 1, -1, -1):
            lo = np.kron(meyers[(jn, idim)], lo)
            hi = np.kron(meyers[(jn + 1, idim)], hi)
        lo_nd = lo.reshape(*shape)
        hi_nd = hi.reshape(*shape)
        bp_nd = hi_nd - lo_nd
        bp_nd[bp_nd < 0] = 0
        bandpasses[jn] = bp_nd
    bandpasses[0] = lo_nd
    return shape_grid, bandpasses


def _nchoosek(n: Any, k: Any) -> np.ndarray:
    return np.asarray(list(combinations(n, k)))


def _create_mdirs(dim: int, res: int) -> list[np.ndarray]:
    # Mdir is dimension of need to calculate angle function on each
    # hyperpyramid
    return [
        np.c_[[np.r_[np.arange(idim), np.arange(idim + 1, dim)] for idim in range(dim)]]
        for ires in range(res)
    ]
    # Mdirs: dict[int, np.ndarray] = {}
    # for ires in range(res):
    #     Mdirs[ires] = np.zeros((dim, dim - 1), dtype=int)
    #     for idim in range(dim):
    #         Mdirs[ires][idim, :] = np.r_[range(idim), range(idim + 1, dim)]

    # return Mdirs


def _create_angle_info(
    Sgrid: dict[int, np.ndarray], dim: int, res: int, cfg: np.ndarray, alpha: float
) -> tuple[
    dict[int, dict[tuple[int, int], np.ndarray]],
    dict[int, dict[tuple[int, int], np.ndarray]],
]:
    # every combination of 2 dimension out of 1:dim
    mperms = _nchoosek(np.arange(dim), 2)
    Mgrid: dict[tuple[int, int], np.ndarray] = {}
    for ind, perm in enumerate(mperms):
        out = adapt_grid(Sgrid[perm[0]], Sgrid[perm[1]])
        Mgrid[(ind, 0)] = out[0]
        Mgrid[(ind, 1)] = out[1]

    # gather angle function for each pyramid
    Mangs: dict[int, dict[tuple[int, int], np.ndarray]] = {}
    Minds: dict[int, dict[tuple[int, int], np.ndarray]] = {}
    for ires in range(res):
        Mangs[ires] = {}
        Minds[ires] = {}
        # for each resolution
        for idim in range(dim):
            # for each pyramid in resolution res
            cnt = 0
            # cnt is number of angle function required for each pyramid
            # now loop through mperms
            for ihyp in range(mperms.shape[0]):
                for idir in range(mperms.shape[1]):
                    if mperms[ihyp, idir] == idim:
                        Mangs[ires][(idim, cnt)] = angle_fun(
                            Mgrid[(ihyp, idir)],
                            idir + 1,
                            cfg[ires, mperms[ihyp, 1 - idir]],
                            alpha,
                        )
                        Minds[ires][(idim, cnt)] = mperms[ihyp, :] + 1
                        cnt += 1
    return Mangs, Minds


def _inplace_normalize_windows(
    udctwin: UDCTWindows, size: tuple[int, ...], dim: int, res: int
) -> None:
    sumw2 = np.zeros(size)
    idx, val = from_sparse_new(udctwin[0][0][0])
    sumw2.flat[idx] += val**2
    for ires in range(1, res + 1):
        for idir in range(dim):
            for iang in range(len(udctwin[ires][idir])):
                tmpw = np.zeros(size)
                idx, val = from_sparse_new(udctwin[ires][idir][iang])
                tmpw.flat[idx] += val**2
                sumw2 += tmpw
                tmpw = fftflip(tmpw, idir)
                sumw2 += tmpw

    sumw2 = np.sqrt(sumw2)
    idx, val = from_sparse_new(udctwin[0][0][0])
    val /= sumw2.ravel()[idx]
    for ires in range(1, res + 1):
        for idir in range(dim):
            for iang in range(len(udctwin[ires][idir])):
                idx, val = from_sparse_new(udctwin[ires][idir][iang])
                val /= sumw2.ravel()[idx]


def _calculate_decimation_ratios_with_lowest(
    res: int, dim: int, cfg: np.ndarray, Mdirs: list[np.ndarray]
) -> list[npt.NDArray[np.int_]]:
    decimation_ratio: list[npt.NDArray[np.int_]] = [
        np.full((1, dim), fill_value=2 ** (res - 1), dtype=int)
    ]
    for ires in range(1, res + 1):
        decimation_ratio.append(
            np.full((dim, dim), fill_value=2.0 ** (res - ires + 1), dtype=int)
        )
        for i0 in range(dim):
            i1s = Mdirs[ires - 1][i0, :]
            decimation_ratio[ires][i0, i1s] = (
                2 * cfg[ires - 1, i1s] * 2 ** (res - ires) // 3
            )
    return decimation_ratio


def _inplace_sort_windows(
    udctwin: UDCTWindows, indices: dict[int, dict[int, np.ndarray]], res: int, dim: int
) -> None:
    for ires in range(1, res + 1):
        for idim in range(dim):
            mlist = indices[ires][idim]

            # # Approach 1: Create a structured array and then sort by fields
            # struct = [(f"x{i}", "<i8") for i in range(mlist.shape[1])]
            # ix = np.argsort(
            #     np.array([tuple(m) for m in mlist], dtype=struct),
            #     order=tuple(t[0] for t in struct),
            # )
            #
            # Approach 2: Create a 1D array then sort that array
            m = mlist.max() + 1
            ix = np.argsort(
                sum(
                    m**i2 * mlist[:, i1]
                    for i1, i2 in enumerate(range(mlist.shape[1] - 1, -1, -1))
                )
            )

            indices[ires][idim] = mlist[ix]
            udctwin[ires][idim] = [udctwin[ires][idim][idx] for idx in ix]


def udctmdwin(
    param_udct: ParamUDCT,
) -> tuple[UDCTWindows, list[npt.NDArray[np.int_]], dict[int, dict[int, np.ndarray]]]:
    Sgrid, F2d = _create_bandpass_windows(
        nscales=param_udct.res, shape=param_udct.size, r=param_udct.r
    )
    Winlow = circshift(np.sqrt(F2d[0]), tuple(s // 4 for s in param_udct.size))

    # convert to sparse format
    udctwin: UDCTWindows = []
    udctwin.append([])
    udctwin[0].append([])
    udctwin[0][0] = [to_sparse_new(Winlow, param_udct.winthresh)]

    # `indices` gets stored as `param_udct.ind` in the original.
    indices: dict[int, dict[int, np.ndarray]] = {}
    indices[0] = {}
    indices[0][0] = np.zeros((1, 1), dtype=int)
    Mdirs = _create_mdirs(dim=param_udct.dim, res=param_udct.res)
    Mangs, Minds = _create_angle_info(
        Sgrid,
        dim=param_udct.dim,
        res=param_udct.res,
        cfg=param_udct.cfg,
        alpha=param_udct.alpha,
    )

    # decimation ratio for each band
    decimation_ratio = _calculate_decimation_ratios_with_lowest(
        res=param_udct.res, dim=param_udct.dim, cfg=param_udct.cfg, Mdirs=Mdirs
    )

    # Mang is 1-d angle function for each hyper pyramid (row) and each angle
    # dimension (column)
    for ires in range(1, param_udct.res + 1):  # pylint: disable=too-many-nested-blocks
        # for each resolution
        udctwin.append([])
        indices[ires] = {}
        for idim in range(param_udct.dim):
            udctwin[ires].append([])
            # for each hyperpyramid
            i1_ang = np.arange(len(Mangs[ires - 1][(idim, 0)]))[:, None]
            for i2 in range(1, param_udct.dim - 1):
                ln = len(Mangs[ires - 1][(idim, i2)])
                tmp2 = np.arange(len(Mangs[ires - 1][(idim, i2)]))[:, None]
                tmp3 = np.kron(i1_ang, np.ones((ln, 1), dtype=int))
                tmp4 = np.kron(np.ones((i1_ang.shape[0], 1), dtype=int), tmp2)
                i1_ang = np.c_[tmp3, tmp4]
            lent = i1_ang.shape[0]
            ang_inmax = param_udct.cfg[ires - 1, Mdirs[ires - 1][idim, :]]
            # lent is the smallest number of windows need to calculated on each
            # pyramid
            # ang_inmax is M-1 vector contain number of angle function per each
            # dimension of the hyperpyramid
            for i3 in range(lent):
                # for each calculated windows function, estimated all the other
                # flipped window functions
                win: npt.NDArray[np.floating] = np.ones(param_udct.size, dtype=float)
                for i4 in range(param_udct.dim - 1):
                    idx = i1_ang.reshape(len(i1_ang), -1)[i3, i4]
                    tmp1 = Mangs[ires - 1][(idim, i4)][idx]
                    tmp2 = Minds[ires - 1][(idim, i4)]
                    afun2 = angle_kron(tmp1, tmp2, param_udct)
                    win *= afun2
                win *= F2d[ires]
                win = np.sqrt(circshift(win, tuple(s // 4 for s in param_udct.size)))

                # first windows function
                angle_functions = []
                angle_functions.append(win)

                # index of current angle
                i2_ang = i1_ang[i3 : i3 + 1, :] + 1

                # all possible flip along different dimension
                for i5 in range(param_udct.dim - 2, -1, -1):
                    for i6 in range(i2_ang.shape[0]):
                        if 2 * i2_ang[i6, i5] <= ang_inmax[i5]:
                            i2_ang_tmp = i2_ang[i6 : i6 + 1, :].copy()
                            i2_ang_tmp[0, i5] = ang_inmax[i5] + 1 - i2_ang[i6, i5]
                            i2_ang = np.r_[i2_ang, i2_ang_tmp]
                            win = fftflip(
                                angle_functions[i6], Mdirs[ires - 1][idim, i5]
                            )
                            angle_functions.append(win)
                i2_ang -= 1  # Adjust so that `indices` is 0-based
                angle_functions = np.c_[angle_functions]

                if i3 == 0:
                    ang_ind = i2_ang
                    for i7 in range(ang_ind.shape[0]):
                        udctwin[ires][idim].append(
                            to_sparse_new(angle_functions[i7], param_udct.winthresh)
                        )
                else:
                    inold = ang_ind.shape[0]
                    ang_ind = np.concatenate((ang_ind, i2_ang), axis=0)
                    innew = ang_ind.shape[0]
                    for i7 in range(inold, innew):
                        udctwin[ires][idim].append(
                            to_sparse_new(
                                angle_functions[i7 - inold], param_udct.winthresh
                            )
                        )
                    indices[ires][idim] = ang_ind.copy()

    # Normalization
    _inplace_normalize_windows(
        udctwin, size=param_udct.size, dim=param_udct.dim, res=param_udct.res
    )

    # sort the window
    _inplace_sort_windows(
        udctwin=udctwin, indices=indices, res=param_udct.res, dim=param_udct.dim
    )

    return udctwin, decimation_ratio, indices
