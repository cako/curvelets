from __future__ import annotations

import numpy as np
import numpy.typing as npt

from curvelets.ucurv.meyerwavelet import meyerfwdmd, meyerinvmd
from curvelets.ucurv.ucurv import (
    alpha,
    angle_fun,
    angle_kron,
    downsamp,
    fftflip,
    generate_combinations,
    r,
    tan_theta_grid,
    upsamp,
)
from curvelets.ucurv.util import fun_meyer


class UDCT:
    def __init__(
        self,
        shape: tuple[int, ...],
        cfg: npt.NDArray[np.integer],
        complex: bool = False,
        sparse: bool = False,
        high: str = "curvelet",
    ) -> None:
        if high == "wavelet":
            self.sz = tuple(np.array(shape) // 2)
        else:
            self.sz = tuple(shape)
        self.cfg = tuple(cfg)
        self.complex = complex
        self.sparse = sparse
        self.high = high
        self.dim = len(shape)
        self.res = len(cfg)
        self.create_windows()

    def create_windows(self) -> None:
        cfg = self.cfg
        sz = self.sz
        dim = len(sz)
        res = len(cfg)
        sparse = self.sparse

        self.Sampling: dict[tuple[int, ...], npt.NDArray[np.integer]] = {}
        # calculate output len
        clen = np.prod(np.array(self.sz)) // ((2**self.dim) ** (self.res - 1))
        self.len = clen
        for i in range(self.res):
            clen = clen * ((2**self.dim) ** i)
            self.len = self.len + clen * self.dim * 3 ** (self.dim - 1) // 2 ** (
                self.dim - 1
            )

        # create the subsampling vectors
        self.Sampling[(0,)] = 2 ** (res - 1) * np.ones(dim, dtype=int)
        for rs in range(res):
            for ipyr in range(dim):
                dmat: list[int] = []
                for idir in range(dim):
                    if idir == ipyr:
                        dmat.append(2 ** (res - rs))
                    else:
                        dmat.append(2 * (cfg[rs][idir] // 3) * 2 ** (res - rs - 1))
                self.Sampling[(rs, ipyr)] = np.array(dmat, dtype=int)

        Sgrid = [np.empty(0) for i in range(dim)]

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

        SLgrid = [np.empty(0) for i in range(dim)]
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
                    new_list = [[*z, j] for z in new_list for j in range(cfg[rs][x[i]])]
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
                            (ipyr, dlist[idir]),
                            self.sz,
                        )
                        subband = subband * angkron
                        cnt += 1

                    Msubwin[(rs, ipyr, *alist)] = subband.copy()

        #################################
        sumall = np.zeros(self.sz)
        for subwin in Msubwin.values():
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
        win = np.sqrt(np.prod(self.Sampling[(0,)])) * np.fft.fftshift(
            np.sqrt(FL / sumall)
        )
        if sparse:
            self.FL = (np.nonzero(win), win[np.nonzero(win)])
        else:
            self.FL = win

    def forward(
        self, img: npt.NDArray
    ) -> dict[tuple[int, ...], npt.NDArray[np.integer]]:
        if self.high == "curvelet":
            assert img.shape == self.sz
        Msubwin = self.Msubwin
        # FL = self.FL
        Sampling = self.Sampling
        if self.sparse:
            FL = np.zeros(self.sz)
            FL[self.FL[0]] = self.FL[1]
        else:
            FL = self.FL
        imband: dict[tuple[int, ...], npt.NDArray] = {}
        if self.high == "wavelet":
            band = meyerfwdmd(img)
            for i, band in enumerate(band):
                if i == 0:
                    imf = np.fft.fftn(band)
                else:
                    imband[(self.res, i)] = band
        else:
            imf = np.fft.fftn(img)

        if self.complex:
            bandfilt = np.fft.ifftn(imf * FL)
            imband[(0,)] = downsamp(bandfilt, Sampling[(0,)])
            for id, subwin in Msubwin.items():
                if self.sparse:
                    sbwin = np.zeros(self.sz)
                    sbwin[subwin[0]] = subwin[1]
                    subwin = sbwin
                bandfilt = np.sqrt(0.5) * np.fft.ifftn(imf * subwin)
                imband[id] = downsamp(bandfilt, Sampling[(id[0], id[1])])
                id2 = list(id)
                id2[1] = id2[1] + self.dim
                bandfilt = np.sqrt(0.5) * np.fft.ifftn(imf * fftflip(subwin))
                imband[tuple(id2)] = downsamp(bandfilt, Sampling[(id[0], id[1])])

        else:
            bandfilt = np.real(np.fft.ifftn(imf * FL))
            imband[(0,)] = downsamp(
                bandfilt, Sampling[(0,)]
            )  # np.real(np.fft.ifftn(imf*FL))
            for id, subwin in Msubwin.items():
                if self.sparse:
                    sbwin = np.zeros(self.sz)
                    sbwin[subwin[0]] = subwin[1]
                    subwin = sbwin

                bandfilt = np.fft.ifftn(imf * subwin)
                # samp = Sampling[(id[0], id[1])]
                # imband[id] = bandfilt[::samp[0], ::samp[1]]
                imband[id] = downsamp(bandfilt, Sampling[(id[0], id[1])])
                # print(bandfilt.shape, Sampling[(id[0], id[1])], imband[id].shape)

        return imband

    def backward(
        self, imband: dict[tuple[int, ...], npt.NDArray[np.integer]]
    ) -> npt.NDArray:
        Msubwin = self.Msubwin
        Sampling = self.Sampling
        # imlow = imband[0]
        imlow = upsamp(imband[(0,)], Sampling[(0,)])

        if self.sparse:
            FL = np.zeros(self.sz)
            FL[self.FL[0]] = self.FL[1]
        else:
            FL = self.FL

        if self.complex:
            recon = np.fft.ifftn(np.fft.fftn(imlow) * FL)
        else:
            recon = np.real(np.fft.ifftn(np.fft.fftn(imlow) * FL))
        for id, subwin in Msubwin.items():
            if self.high != "curvelet" and id[0] == self.res:
                continue

            if self.sparse:
                sbwin = np.zeros(self.sz)
                sbwin[subwin[0]] = subwin[1]
                subwin = sbwin

            if self.complex:
                bandup = upsamp(imband[id], Sampling[(id[0], id[1])])
                recon = recon + np.sqrt(0.5) * np.fft.ifftn(
                    np.fft.fftn(bandup) * subwin
                )
                id2 = list(id)
                id2[1] = id2[1] + self.dim
                bandup = upsamp(imband[tuple(id2)], Sampling[(id[0], id[1])])
                recon = recon + np.sqrt(0.5) * np.fft.ifftn(
                    np.fft.fftn(bandup) * fftflip(subwin)
                )
            else:
                bandup = upsamp(imband[id], Sampling[(id[0], id[1])])
                recon = recon + np.real(np.fft.ifftn(np.fft.fftn(bandup) * subwin))

        if self.high == "wavelet":
            band = [recon]
            for id, suband in imband.items():
                if id[0] == self.res:
                    band.append(suband)

            recon = meyerinvmd(band)

        return recon
