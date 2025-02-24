from __future__ import annotations

import math

import numpy as np

from .meyerwavelet import meyerfwdmd, meyerinvmd
from .util import fun_meyer


def combinations(lst, r):
    if r == 0:
        return [()]
    elif len(lst) < r:
        return []
    elif len(lst) == r:
        return [tuple(lst)]

    result = []
    for i in range(len(lst)):
        for tail in combinations(lst[i + 1 :], r - 1):
            result.append((lst[i],) + tail)
    return result


def generate_combinations(dim):
    lst = list(range(dim))
    return combinations(lst, dim - 1)


def tan_theta_grid(S1, S2):
    """
    Create a grid approximate the tan theta function as described in (28) of the paper
    """
    # first grid
    x1, x2 = np.meshgrid(S2, S1)

    t1 = np.zeros_like(x1)
    ind = np.logical_and(x2 != 0, np.abs(x1) <= np.abs(x2))
    t1[ind] = -x1[ind] / x2[ind]

    t2 = np.zeros_like(x1)
    ind = np.logical_and(x1 != 0, np.abs(x2) < np.abs(x1))
    t2[ind] = x2[ind] / x1[ind]
    t3 = t2.copy()
    t3[t2 < 0] = t2[t2 < 0] + 2
    t3[t2 > 0] = t2[t2 > 0] - 2

    M2 = t1 + t3
    M2[x2 >= 0] = -2

    return M2


def fftflip(F, dirlist=None):
    """
    Return a fftflip of array F, either on a list of dimension, or a single dimension.
    A fftflip use to produce a X(-omega) representation of X(omega) in a FFT representation
    When a FFT X(omega) is flipped (or reverse), the 0 frequency will be the last element.
    The representation need to be rolled by 1 to make the 0 frequency in the first location.
    """
    Fc = F.copy()
    dim = Fc.ndim
    if dirlist is None:
        dirlist = list(range(dim))
    shiftvec = np.zeros(dim)
    if type(dirlist) is list:
        for dir in dirlist:
            shiftvec[dir] = 1
            Fc = np.flip(Fc, dir)
            Fc = np.roll(Fc, 1, axis=dir)
    if type(dirlist) is int:
        dir = dirlist
        shiftvec[dir] = 1
        Fc = np.flip(Fc, dir)
        Fc = np.roll(Fc, 1, axis=dir)
    return Fc


def angle_fun(Mgrid, n, alpha, dir, bandpass=None):
    """
    Return the angle meyer window as in Figure 8 of the paper
    """
    angd = 2 / n
    ang = angd * np.array([-alpha, alpha, 1 - alpha, 1 + alpha])
    Mang = [[] for i in range(n)]
    tmp = []
    sp = np.array(Mgrid.shape)
    for id in range(math.ceil(n / 2)):
        ang2 = -1 + id * angd + ang
        x = fun_meyer(Mgrid, ang2)
        if bandpass is not None:
            x = x * bandpass
        x = np.roll(x, 3 * sp // 4, (0, 1))
        Mang[id] = x.copy()
        Mang[n - 1 - id] = fftflip(x, dir)

    return Mang


def angle_kron(F, dr, sz):
    """
    This function replicate the 2D array F at dimension dr to match the size sz
    """
    sp = list(F.shape)
    sp2 = int(np.prod(sz) / np.prod(F.shape))
    # remove dr element from sz
    sz2 = list(sz)
    sz2 = [i for j, i in enumerate(sz) if j not in dr]
    # now append sz2 to sp
    sp = sp + sz2
    # Fk is the replicated version of F, along the dimension other than dr[0,1]
    Fk = np.reshape(np.kron(F.flatten(), np.ones(sp2)), sp)

    # Move the dimensions 0,1 to dr
    Fk = np.moveaxis(Fk, [0, 1], dr)
    return Fk


def downsamp(band, samp, shift=None):
    """
    Downsample a N-D array by length N of power-2 integers
    """
    if shift is None:
        shift = np.zeros(len(band.shape), dtype=int)
    if len(samp) == 2:
        return band[shift[0] :: samp[0], shift[1] :: samp[1]]
    if len(samp) == 3:
        return band[shift[0] :: samp[0], shift[1] :: samp[1], shift[2] :: samp[2]]
    if len(samp) == 4:
        return band[
            shift[0] :: samp[0],
            shift[1] :: samp[1],
            shift[2] :: samp[2],
            shift[3] :: samp[3],
        ]
    if len(samp) == 5:
        return band[
            shift[0] :: samp[0],
            shift[1] :: samp[1],
            shift[2] :: samp[2],
            shift[3] :: samp[3],
            shift[4] :: samp[4],
        ]


def upsamp(band, samp, shift=None):
    """
    Upsample a N-D array by length N of power-2 integers
    """
    if shift is None:
        shift = np.zeros(len(band.shape), dtype=int)
    sp = np.array(band.shape) * samp
    bandup = np.zeros(sp.astype(int)).astype(complex)
    if len(samp) == 2:
        bandup[shift[0] :: samp[0], shift[1] :: samp[1]] = band
    if len(samp) == 3:
        bandup[shift[0] :: samp[0], shift[1] :: samp[1], shift[2] :: samp[2]] = band
    if len(samp) == 4:
        bandup[
            shift[0] :: samp[0],
            shift[1] :: samp[1],
            shift[2] :: samp[2],
            shift[3] :: samp[3],
        ] = band
    if len(samp) == 5:
        bandup[
            shift[0] :: samp[0],
            shift[1] :: samp[1],
            shift[2] :: samp[2],
            shift[3] :: samp[3],
            shift[4] :: samp[4],
        ] = band

    return bandup


r = np.pi * np.array([1 / 3, 2 / 3, 2 / 3, 4 / 3])
alpha = 0.1


####  class to hold all curvelet windows and other based on transform configuration
class udct:
    def __init__(self, sz, cfg, complex=False, sparse=False, high="curvelet"):
        self.name = "ucurv"
        #
        if high != "curvelet":
            self.sz = tuple(np.array(sz) // 2)
        else:
            self.sz = tuple(sz)

        self.cfg = tuple(cfg)
        self.complex = complex
        self.sparse = sparse
        self.high = high
        self.dim = len(sz)
        self.res = len(cfg)

        dim = len(sz)
        res = len(cfg)

        self.Sampling = {}
        # calculate output len
        clen = np.prod(np.array(self.sz)) // ((2**self.dim) ** (self.res - 1))
        self.len = clen
        for i in range(self.res):
            clen = clen * ((2**self.dim) ** i)
            self.len = self.len + clen * self.dim * 3 ** (self.dim - 1) // 2 ** (
                self.dim - 1
            )

        # create the subsampling vectors
        self.Sampling[(0)] = 2 ** (res - 1) * np.ones(dim, dtype=int)
        for rs in range(res):
            for ipyr in range(dim):
                dmat = []
                for idir in range(dim):
                    if idir == ipyr:
                        dmat.append(2 ** (res - rs))
                    else:
                        dmat.append(2 * (cfg[rs][idir] // 3) * 2 ** (res - rs - 1))
                self.Sampling[(rs, ipyr)] = np.array(dmat, dtype=int)

        Sgrid = [[] for i in range(dim)]

        for ind in range(dim):
            Sgrid[ind] = np.linspace(
                -1.5 * np.pi, 0.5 * np.pi - np.pi / (self.sz[ind] / 2), self.sz[ind]
            )

        f1d = {}
        # print(f1d)
        for ind in range(dim):
            for rs in range(res):
                f1d[(rs, ind)] = fun_meyer(
                    np.abs(Sgrid[ind]),
                    [-2, -1, r[0] / 2 ** (res - 1 - rs), r[1] / 2 ** (res - 1 - rs)],
                )

            f1d[(res, ind)] = fun_meyer(np.abs(Sgrid[ind]), [-2, -1, r[2], r[3]])

        SLgrid = [[] for i in range(dim)]
        for ind in range(dim):
            SLgrid[ind] = np.linspace(
                -np.pi, np.pi - np.pi / (self.sz[ind] / 2), self.sz[ind]
            )

        # fl1d = []
        FL = np.ones([1])
        for ind in range(dim):
            fl1d = fun_meyer(
                np.abs(SLgrid[ind]),
                [-2, -1, r[0] / 2 ** (res - 1), r[1] / 2 ** (res - 1)],
            )
            FL = np.kron(FL, fl1d.flatten())
            # print(FL.shape)
        FL = FL.reshape(self.sz)

        # Mang2 will contain all the 2D angle functions needed to create dim-dimension
        # angle pyramid. As such it is a 4D dictionary 2D angle funtions. The dimension are
        # Resolution - Dimension (number of hyper pyramid) - Dimension-1 (number of angle
        # function in each pyramid ) - Number of angle function in that particular resolution-direction
        #
        Mang2 = {}
        for rs in range(res):
            # For each resolution we loop through each pyramid
            for ind in range(dim):
                # For each pyramid we try to collect all the 2D angle function so that we can build the dim
                # dim-dimension angle functions
                for idir in range(dim):
                    if (
                        idir == ind
                    ):  # skip the dimension that is the same as the pyramid
                        continue

                    ndir = np.array(
                        [ind, idir]
                    )  # ndir are the dimension in the pyramid
                    # print(ndir, cfg[rs][ idir] )
                    Mg0 = tan_theta_grid(Sgrid[ndir[0]], Sgrid[ndir[1]])

                    # create the bandpass function
                    BP1 = np.outer(f1d[(rs, ind)], f1d[(rs, idir)])
                    BP2 = np.outer(f1d[(rs + 1, ind)], f1d[(rs + 1, idir)])
                    bandpass = (BP2 - BP1) ** (1.0 / (dim - 1.0))

                    # create the 2D angle function, in the vertical 2D pyramid
                    Mang2[(rs, ind, idir)] = angle_fun(
                        Mg0, cfg[rs][idir], alpha, 1, bandpass
                    )

        #################################
        Msubwin = {}
        cnt = 0

        for rs in range(res):
            dlists = generate_combinations(dim)[::-1]
            # print(dlists)
            id_angle_lists = []
            for x in dlists:
                new_list = [[i] for i in range(cfg[rs][x[0]])]
                for i in range(1, len(x)):
                    new_list = [z + [j] for z in new_list for j in range(cfg[rs][x[i]])]
                id_angle_lists.append(new_list)
            # print(dlists)
            # print(id_angle_lists)
            for ipyr in range(dim):
                # for each resolution-pyramid, id_angle_list is the angle combinaion within that pyramid
                # for instance, (5,5) would be the last angle of a (6,6) 3D pyramid
                # and dlist is the list of the dimension of that pyramid,
                # for instance (0,2) would be the list of pyramid of dimension 1 in 3D case
                id_angle_list = id_angle_lists[ipyr]
                dlist = list(dlists[ipyr])

                for alist in id_angle_list:
                    subband = np.ones(self.sz)
                    for idir, aid in enumerate(alist):
                        angkron = angle_kron(
                            Mang2[(rs, ipyr, dlist[idir])][aid],
                            [ipyr, dlist[idir]],
                            self.sz,
                        )
                        subband = subband * angkron
                        cnt += 1

                    Msubwin[tuple([rs, ipyr] + alist)] = subband.copy()

        #################################
        sumall = np.zeros(self.sz)
        for id, subwin in Msubwin.items():
            sumall = sumall + subwin
            # print(id, np.max(subwin), np.max(sumall))

        sumall = sumall + fftflip(sumall)
        sumall = sumall + FL

        self.Msubwin = {}
        for id, subwin in Msubwin.items():
            win = np.fft.fftshift(
                np.sqrt(2 * np.prod(self.Sampling[(id[0], id[1])]) * subwin / sumall)
            )
            if sparse:
                self.Msubwin[id] = (np.nonzero(win), win[np.nonzero(win)])
            else:
                self.Msubwin[id] = win
        win = np.sqrt(np.prod(self.Sampling[(0)])) * np.fft.fftshift(
            np.sqrt(FL / sumall)
        )
        if sparse:
            self.FL = (np.nonzero(win), win[np.nonzero(win)])
        else:
            self.FL = win


def ucurvfwd(img, udct):
    if udct.high == "curvelet":
        assert img.shape == udct.sz
    Msubwin = udct.Msubwin
    # FL = udct.FL
    Sampling = udct.Sampling
    if udct.sparse:
        FL = np.zeros(udct.sz)
        FL[udct.FL[0]] = udct.FL[1]
    else:
        FL = udct.FL
    imband = {}
    if udct.high == "wavelet":
        band = meyerfwdmd(img)
        for i, band in enumerate(band):
            if i == 0:
                imf = np.fft.fftn(band)
            else:
                imband[(udct.res, i)] = band
    else:
        imf = np.fft.fftn(img)

    if udct.complex:
        bandfilt = np.fft.ifftn(imf * FL)
        imband[(0,)] = downsamp(bandfilt, Sampling[(0)])
        for id, subwin in Msubwin.items():
            if udct.sparse:
                sbwin = np.zeros(udct.sz)
                sbwin[subwin[0]] = subwin[1]
                subwin = sbwin
            bandfilt = np.sqrt(0.5) * np.fft.ifftn(imf * subwin)
            imband[id] = downsamp(bandfilt, Sampling[(id[0], id[1])])
            id2 = list(id)
            id2[1] = id2[1] + udct.dim
            bandfilt = np.sqrt(0.5) * np.fft.ifftn(imf * fftflip(subwin))
            imband[tuple(id2)] = downsamp(bandfilt, Sampling[(id[0], id[1])])

    else:
        bandfilt = np.real(np.fft.ifftn(imf * FL))
        imband[(0,)] = downsamp(
            bandfilt, Sampling[(0)]
        )  # np.real(np.fft.ifftn(imf*FL))
        for id, subwin in Msubwin.items():
            if udct.sparse:
                sbwin = np.zeros(udct.sz)
                sbwin[subwin[0]] = subwin[1]
                subwin = sbwin

            bandfilt = np.fft.ifftn(imf * subwin)
            # samp = Sampling[(id[0], id[1])]
            # imband[id] = bandfilt[::samp[0], ::samp[1]]
            imband[id] = downsamp(bandfilt, Sampling[(id[0], id[1])])
            # print(bandfilt.shape, Sampling[(id[0], id[1])], imband[id].shape)

    return imband


##############
def ucurvinv(imband, udct):
    Msubwin = udct.Msubwin
    Sampling = udct.Sampling
    # imlow = imband[0]
    imlow = upsamp(imband[(0,)], Sampling[(0)])

    if udct.sparse:
        FL = np.zeros(udct.sz)
        FL[udct.FL[0]] = udct.FL[1]
    else:
        FL = udct.FL

    if udct.complex:
        recon = np.fft.ifftn(np.fft.fftn(imlow) * FL)
    else:
        recon = np.real(np.fft.ifftn(np.fft.fftn(imlow) * FL))
    for id, subwin in Msubwin.items():
        #
        if udct.high != "curvelet" and id[0] == udct.res:
            continue

        if udct.sparse:
            sbwin = np.zeros(udct.sz)
            sbwin[subwin[0]] = subwin[1]
            subwin = sbwin

        if udct.complex:
            bandup = upsamp(imband[id], Sampling[(id[0], id[1])])
            recon = recon + np.sqrt(0.5) * np.fft.ifftn(np.fft.fftn(bandup) * subwin)
            id2 = list(id)
            id2[1] = id2[1] + udct.dim
            bandup = upsamp(imband[tuple(id2)], Sampling[(id[0], id[1])])
            recon = recon + np.sqrt(0.5) * np.fft.ifftn(
                np.fft.fftn(bandup) * fftflip(subwin)
            )
        else:
            bandup = upsamp(imband[id], Sampling[(id[0], id[1])])
            recon = recon + np.real(np.fft.ifftn(np.fft.fftn(bandup) * subwin))

    if udct.high == "wavelet":
        band = [recon]
        for id, suband in imband.items():
            if id[0] == udct.res:
                band.append(suband)

        recon = meyerinvmd(band)

    return recon
