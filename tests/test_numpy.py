from __future__ import annotations

import numpy as np
import pytest

import curvelets.numpy as udct
import curvelets.reference as udct_ref


@pytest.mark.parametrize("dim", list(range(2, 5)))
def test_uniform(dim):
    rng = np.random.default_rng()

    opts = [32, 64, 128, 256]
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
    alpha = 0.3 * rng.uniform(size=1)
    r = np.pi * np.array([1.0, 2.0, 2.0, 4.0]) / 3
    winthresh = 10.0 ** (-rng.integers(low=4, high=6, size=1))

    my_udct = udct.UDCT(size=size, cfg=cfg, alpha=alpha, r=r, winthresh=winthresh)
    param_ref = udct_ref.ParamUDCT(
        dim=dim, size=size, cfg=cfg, alpha=alpha, r=r, winthresh=winthresh
    )

    udctwin = my_udct.windows
    udctwin_ref, _ = udct_ref.udctmdwin(param_ref)

    assert udctwin.keys() == udctwin_ref.keys()
    for res in udctwin:
        assert udctwin[res].keys() == udctwin_ref[res].keys()
        for dir in udctwin[res]:
            if res == 1 and dir == 1:
                np.testing.assert_allclose(
                    udctwin[res][dir], udctwin_ref[res][dir], rtol=1e-14
                )
            else:
                assert udctwin[res][dir].keys() == udctwin_ref[res][dir].keys()
                for ang in udctwin[res][dir]:
                    np.testing.assert_allclose(
                        udctwin[res][dir][ang], udctwin_ref[res][dir][ang], rtol=1e-14
                    )

    im = rng.normal(size=size)
    coeffs = my_udct.forward(im)
    coeffs_ref, _ = udct_ref.udctmddec(im, param_udct=param_ref, udctwin=udctwin_ref)

    for res in coeffs:
        assert coeffs[res].keys() == coeffs_ref[res].keys()
        for dir in coeffs[res]:
            if res == 1 and dir == 1:
                np.testing.assert_allclose(
                    coeffs[res][dir], coeffs_ref[res][dir], rtol=1e-14
                )
            else:
                assert coeffs[res][dir].keys() == coeffs_ref[res][dir].keys()
                for ang in coeffs[res][dir]:
                    np.testing.assert_allclose(
                        coeffs[res][dir][ang], coeffs_ref[res][dir][ang], rtol=1e-14
                    )
    im2 = my_udct.backward(coeffs)
    im2_ref, _ = udct_ref.udctmdrec(
        coeffs_ref, param_udct=param_ref, udctwin=udctwin_ref
    )
    np.testing.assert_allclose(im2, im2_ref, rtol=1e-14)
