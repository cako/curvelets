"""Adjoint and Parseval tight-frame tests for NumPy UDCT."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest

from curvelets.numpy import UDCT

Mode = Literal["curvelet", "wavelet"]
Kind = Literal["real", "complex"]


@pytest.mark.parametrize("shape", [(32, 32), (16, 16, 16)])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("num_scales", [2, 3])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("high_frequency_mode", ["curvelet", "wavelet"])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("transform_kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_numpy_adjoint_and_parseval(
    shape: tuple[int, ...],
    num_scales: int,
    high_frequency_mode: Mode,
    transform_kind: Kind,
) -> None:
    """Forward and backward form an exact tight frame and adjoint pair."""
    rng = np.random.default_rng(0)
    udct = UDCT(
        shape=shape,
        num_scales=num_scales,
        wedges_per_direction=3,
        high_frequency_mode=high_frequency_mode,
        transform_kind=transform_kind,
    )
    x: npt.NDArray[np.float64] | npt.NDArray[np.complex128] = rng.standard_normal(shape)
    if transform_kind == "complex":
        x = x + 1j * rng.standard_normal(shape)

    u_x = udct.vect(udct.forward(x))
    recon = udct.backward(udct.struct(u_x))
    assert isinstance(recon, np.ndarray)
    assert np.linalg.norm(recon - x) / np.linalg.norm(x) < 1e-12

    energy_x = float(np.vdot(x, x).real)
    energy_u_x = float(np.vdot(u_x, u_x).real)
    assert abs(energy_u_x - energy_x) / energy_x < 1e-12

    y = rng.standard_normal(u_x.shape) + 1j * rng.standard_normal(u_x.shape)
    uh_y = udct.backward(udct.struct(y))
    assert isinstance(uh_y, np.ndarray)
    lhs = np.vdot(y, u_x)
    rhs = np.vdot(uh_y, x)
    if transform_kind == "real":
        lhs_val = float(lhs.real)
        rhs_val = float(rhs.real)
        assert abs(lhs_val - rhs_val) / abs(lhs_val) < 1e-12
    else:
        assert abs(lhs - rhs) / abs(lhs) < 1e-12


@pytest.mark.parametrize("num_scales", [2, 3])  # type: ignore[untyped-decorator]
def test_numpy_monogenic_wavelet_scalar_matches_real(num_scales: int) -> None:
    """Scalar channels of monogenic wavelet mode match the real transform."""
    rng = np.random.default_rng(1)
    shape = (32, 32)
    x = rng.standard_normal(shape)
    udct_real = UDCT(
        shape=shape,
        num_scales=num_scales,
        wedges_per_direction=3,
        high_frequency_mode="wavelet",
        transform_kind="real",
    )
    udct_mono = UDCT(
        shape=shape,
        num_scales=num_scales,
        wedges_per_direction=3,
        high_frequency_mode="wavelet",
        transform_kind="monogenic",
    )
    coeffs_real = udct_real.forward(x)
    coeffs_mono = udct_mono.forward(x)
    for scale_r, scale_m in zip(coeffs_real, coeffs_mono, strict=True):
        for dir_r, dir_m in zip(scale_r, scale_m, strict=True):
            for wedge_r, wedge_m in zip(dir_r, dir_m, strict=True):
                scalar = (
                    np
                    .ascontiguousarray(wedge_m[..., :2])
                    .view(np.complex128)
                    .squeeze(-1)
                )
                np.testing.assert_allclose(scalar, wedge_r, atol=1e-12, rtol=1e-12)

    recon_real = udct_real.backward(coeffs_real)
    recon_mono = udct_mono.backward(coeffs_mono)
    assert isinstance(recon_real, np.ndarray)
    assert isinstance(recon_mono, tuple)
    np.testing.assert_allclose(recon_mono[0], recon_real, atol=1e-12, rtol=1e-12)
