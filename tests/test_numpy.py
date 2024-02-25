from __future__ import annotations

import numpy as np
import pytest

import curvelets.numpy as udct
import curvelets.reference as udct_ref
from curvelets.numpy.utils import from_sparse, from_sparse_new


@pytest.mark.parametrize("dim", list(range(2, 5)))
def test_compare_with_reference(dim):
    rng = np.random.default_rng()

    opts = [16, 32, 64, 128, 256]
    if dim == 3:
        opts = opts[:2]
    elif dim >= 4:
        opts = opts[:1]
    size = rng.choice(opts, size=dim, replace=True)
    cfg = (
        np.array([[3, 3], [6, 6], [12, 6]])
        if dim == 2
        else np.c_[np.ones((dim,)) * 3, np.ones((dim,)) * 6].T
    )
    alpha = 0.3 * rng.uniform(size=1).item()
    r: tuple[float, float, float, float] = tuple(
        np.pi * np.array([1.0, 2.0, 2.0, 4.0]) / 3
    )
    winthresh = 10.0 ** (-rng.integers(low=4, high=6, size=1).item())

    my_udct = udct.UDCT(shape=size, cfg=cfg, alpha=alpha, r=r, winthresh=winthresh)
    param_ref = udct_ref.ParamUDCT(
        dim=dim, size=size, cfg=cfg, alpha=alpha, r=r, winthresh=winthresh
    )

    udctwin = my_udct.windows
    udctwin_ref = udct_ref.udctmdwin(param_ref)
    assert param_ref.ind is not None

    rdtype = udctwin[0][0][0][1].real.dtype
    cdtype = (np.ones(1, dtype=rdtype) + 1j * np.ones(1, dtype=rdtype)).dtype
    win = np.zeros(param_ref.size, dtype=cdtype)
    win_ref = np.zeros(param_ref.size, dtype=cdtype)

    np.testing.assert_array_equal(
        1 + np.arange(len(udctwin)), np.asarray(list(udctwin_ref.keys()))
    )
    for res in udctwin_ref:
        np.testing.assert_array_equal(
            1 + np.arange(len(udctwin[res - 1])),
            np.asarray(list(udctwin_ref[res].keys())),
        )
        for dir in udctwin_ref[res]:
            if res == 1 and dir == 1:
                np.testing.assert_array_equal(
                    my_udct.indices[0][0], param_ref.ind[1][1]
                )
                idx, val = from_sparse_new(udctwin[0][0][0])
                win[...] = 0  # Reset
                win.flat[idx] = val  # Fill

                idx, val = from_sparse(udctwin_ref[1][1])
                win_ref[...] = 0  # Reset
                win_ref.T.flat[idx] = val  # Fill
                np.testing.assert_allclose(win, win_ref, rtol=1e-14)

            else:
                np.testing.assert_array_equal(
                    1 + my_udct.indices[res - 1][dir - 1], param_ref.ind[res][dir]
                )
                np.testing.assert_array_equal(
                    1 + np.arange(len(udctwin[res - 1][dir - 1])),
                    np.asarray(list(udctwin_ref[res][dir].keys())),
                )
                for ang in udctwin_ref[res][dir]:
                    idx, val = from_sparse_new(udctwin[res - 1][dir - 1][ang - 1])
                    win[...] = 0  # Reset
                    win.flat[idx] = val  # Fill

                    idx, val = from_sparse(udctwin_ref[res][dir][ang])
                    win_ref[...] = 0  # Reset
                    win_ref.T.flat[idx] = val  # Fill
                    np.testing.assert_allclose(win, win_ref, rtol=1e-14)

    im = rng.normal(size=size)
    coeffs = my_udct.forward(im)
    coeffs_ref = udct_ref.udctmddec(im, param_udct=param_ref, udctwin=udctwin_ref)

    np.testing.assert_array_equal(
        1 + np.arange(len(coeffs)), np.asarray(list(coeffs_ref.keys()))
    )
    for res in coeffs_ref:
        np.testing.assert_array_equal(
            1 + np.arange(len(coeffs[res - 1])),
            np.asarray(list(coeffs_ref[res].keys())),
        )
        for dir in coeffs_ref[res]:
            if res == 1 and dir == 1:
                np.testing.assert_allclose(
                    coeffs[res - 1][dir - 1][0], coeffs_ref[res][dir], rtol=1e-14
                )
            else:
                np.testing.assert_array_equal(
                    1 + np.arange(len(coeffs[res - 1][dir - 1])),
                    np.asarray(list(coeffs_ref[res][dir].keys())),
                )
                for ang in coeffs_ref[res][dir]:
                    np.testing.assert_allclose(
                        coeffs[res - 1][dir - 1][ang - 1],
                        coeffs_ref[res][dir][ang],
                        rtol=1e-14,
                    )
    im2 = my_udct.backward(coeffs)
    im2_ref = udct_ref.udctmdrec(coeffs_ref, param_udct=param_ref, udctwin=udctwin_ref)
    np.testing.assert_allclose(im2, im2_ref, rtol=1e-14)


@pytest.mark.parametrize("dim", list(range(2, 4)))
def test_round_trip_absolute(dim):
    rng = np.random.default_rng()

    # For these specific parameters, we can guarantee an absolute precision of
    # approximately 1e-4
    size: tuple[int, ...]
    if dim == 2:
        size = (256, 256)
    elif dim == 3:
        size = tuple(4 * np.array([32, 32, 32]))
    elif dim == 3:
        size = (16, 16, 16, 16)
    cfg = (
        np.array([[3, 3], [6, 6], [12, 6]])
        if dim == 2
        else np.c_[np.ones((dim,)) * 3, np.ones((dim,)) * 6].T
    )
    alpha = 0.15
    r = tuple(np.pi * np.array([1.0, 2.0, 2.0, 4.0]) / 3)
    winthresh = 1e-5

    my_udct = udct.UDCT(shape=size, cfg=cfg, alpha=alpha, r=r, winthresh=winthresh)
    im = rng.normal(size=size)
    coeffs = my_udct.forward(im)
    im2 = my_udct.backward(coeffs)

    # try:
    #     import matplotlib.pyplot as plt

    #     idx = [np.random.choice(s) for s in im.shape]
    #     for i in np.random.choice(im.ndim, 2, replace=False):
    #         idx[i] = slice(None)
    #     idx = tuple(idx)

    #     fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    #     img = axs[0].imshow(im[idx])
    #     fig.colorbar(img)
    #     img = axs[1].imshow(im2[idx])
    #     fig.colorbar(img)
    #     img = axs[2].imshow((im - im2)[idx])
    #     fig.colorbar(img)
    #     fig.suptitle(f"Error: {np.abs(im - im2).max()}")
    #     fig.tight_layout()
    #     plt.show()
    # except ImportError:
    #     pass

    np.testing.assert_allclose(im, im2, atol=1e-4)


@pytest.mark.parametrize("dim", list(range(2, 5)))
def test_round_trip_rel(dim):
    rng = np.random.default_rng()

    # For random parameters in the range below, we can guarantee an relative precision of
    # approximately 0.5% of the maximum amplitude in the original image.
    opts = [16, 32, 64, 128, 256]
    if dim == 3:
        opts = opts[:3]
    elif dim >= 4:
        opts = opts[:1]
    size = rng.choice(opts, size=dim, replace=True)
    cfg = (
        np.array([[3, 3], [6, 6], [12, 6]])
        if dim == 2
        else np.c_[np.ones((dim,)) * 3, np.ones((dim,)) * 6].T
    )
    alpha = 0.3 * rng.uniform(size=1)
    r = np.pi * np.array([1.0, 2.0, 2.0, 4.0]) / 3
    winthresh = 10.0 ** (-rng.integers(low=4, high=6, size=1))

    my_udct = udct.UDCT(shape=size, cfg=cfg, alpha=alpha, r=r, winthresh=winthresh)
    im = rng.normal(size=size)
    coeffs = my_udct.forward(im)
    im2 = my_udct.backward(coeffs)

    # try:
    #     import matplotlib.pyplot as plt

    #     idx = [np.random.choice(s) for s in im.shape]
    #     for i in np.random.choice(im.ndim, 2, replace=False):
    #         idx[i] = slice(None)
    #     idx = tuple(idx)

    #     fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    #     img = axs[0].imshow(im[idx])
    #     fig.colorbar(img)
    #     img = axs[1].imshow(im2[idx])
    #     fig.colorbar(img)
    #     img = axs[2].imshow((im - im2)[idx])
    #     fig.colorbar(img)
    #     fig.suptitle(f"Error: {np.abs(im - im2).max()}")
    #     fig.tight_layout()
    #     plt.show()
    # except ImportError:
    #     pass

    np.testing.assert_allclose(im, im2, atol=0.005 * im.max())


@pytest.mark.parametrize("dim", list(range(2, 5)))
def test_vect_struct(dim):
    rng = np.random.default_rng()

    opts = [16, 32, 64, 128, 256]
    if dim == 3:
        opts = opts[:3]
    elif dim >= 4:
        opts = opts[:1]
    shape: tuple[int, ...] = tuple(rng.choice(opts, size=dim, replace=True))
    cfg = (
        np.array([[3, 3], [6, 6], [12, 6]])
        if dim == 2
        else np.c_[np.ones((dim,)) * 3, np.ones((dim,)) * 6].T
    )
    alpha = 0.3 * rng.uniform(size=1).item()
    r: tuple[float, float, float, float] = tuple(
        np.pi * np.array([1.0, 2.0, 2.0, 4.0]) / 3
    )
    winthresh = 10.0 ** (-rng.integers(low=4, high=6, size=1).item())

    C = udct.UDCT(shape=shape, cfg=cfg, alpha=alpha, r=r, winthresh=winthresh)
    im = rng.normal(size=shape)
    c1 = C.forward(im)

    c2 = C.struct(C.vect(c1))
    for ires, _ in enumerate(c1):
        for idir, _ in enumerate(c1[ires]):
            for iang, _ in enumerate(c1[ires][idir]):
                np.testing.assert_allclose(c1[ires][idir][iang], c2[ires][idir][iang])
