from __future__ import annotations

import numpy as np

import curvelets.numpy as udct
import curvelets.reference as udct_ref


def test_uniform_2d():
    rng = np.random.default_rng()

    dim = 2
    size = rng.choice([32, 64, 128, 256], size=dim, replace=True)
    cfg = np.array([[3, 3], [6, 6], [12, 6]])
    alpha = 0.3 * rng.uniform(size=1)
    r = np.pi * np.array([1.0, 2.0, 2.0, 4.0]) / 3
    winthresh = 10.0 ** (-rng.integers(low=4, high=6, size=1))
    param = udct.ParamUDCT(
        dim=dim, size=size, cfg=cfg, alpha=alpha, r=r, winthresh=winthresh
    )
    param_ref = udct_ref.ParamUDCT(
        dim=dim, size=size, cfg=cfg, alpha=alpha, r=r, winthresh=winthresh
    )

    udctwin = udct.udctmdwin(param)
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
    coeffs = udct.udctmddec(im, param_udct=param, udctwin=udctwin)
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
