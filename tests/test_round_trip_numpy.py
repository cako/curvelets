"""Round-trip tests for NumPy UDCT implementation only."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import get_test_configs, get_test_shapes, setup_numpy_transform


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_numpy_round_trip_absolute(dim, rng):
    """
    Test NumPy implementation round-trip with absolute tolerance.

    For specific parameters, we can guarantee an absolute precision of approximately 1e-4.
    """
    transform = setup_numpy_transform(dim)
    shapes = get_test_shapes(dim)
    size = shapes[0]

    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4 if dim in {2, 3} else 1e-3
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_numpy_round_trip_relative(dim, rng):
    """
    Test NumPy implementation round-trip with relative tolerance.

    For random parameters in the range below, we can guarantee a relative precision of
    approximately 0.5% of the maximum amplitude in the original image.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)
    shape_idx = rng.integers(0, len(shapes))
    cfg_idx = rng.integers(0, len(configs))
    transform = setup_numpy_transform(dim, shape_idx=shape_idx, cfg_idx=cfg_idx)

    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4
    np.testing.assert_allclose(data, recon, atol=atol * data.max())


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("shape_idx", [0, 1])
def test_numpy_round_trip_parametrized(dim, shape_idx, rng):
    """Test NumPy implementation round-trip with parametrized shapes and configs."""
    shapes = get_test_shapes(dim)
    if shape_idx >= len(shapes):
        pytest.skip(f"Shape index {shape_idx} out of range for dimension {dim}")

    transform = setup_numpy_transform(dim, shape_idx=shape_idx)
    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4 if dim in {2, 3} else 1e-3
    np.testing.assert_allclose(data, recon, atol=atol)


# ============================================================================
# Wavelet mode tests
# ============================================================================


def _get_wavelet_config_idx(dim: int) -> int | None:
    """
    Get the config index for wavelet mode tests.

    Wavelet mode requires at least 2 scales. Returns None if no suitable config exists.

    Parameters
    ----------
    dim : int
        Dimension.

    Returns
    -------
    int | None
        Config index with at least 2 scales, or None if not available.
    """
    configs = get_test_configs(dim)
    for idx, cfg in enumerate(configs):
        if len(cfg) >= 2:  # nscales >= 2
            return idx
    return None


@pytest.mark.round_trip
@pytest.mark.parametrize(
    "dim,high",
    [
        (2, "wavelet"),
        (3, "wavelet"),
        (4, "wavelet"),
    ],
)
def test_numpy_round_trip_wavelet_absolute(dim, high, rng):
    """
    Test NumPy implementation round-trip with wavelet mode using absolute tolerance.

    Wavelet mode applies Meyer wavelet decomposition at the highest scale,
    with curvelet transform on the lowpass component only.
    """
    cfg_idx = _get_wavelet_config_idx(dim)
    if cfg_idx is None:
        pytest.skip(f"No config with nscales >= 2 available for dimension {dim}")

    transform = setup_numpy_transform(dim, cfg_idx=cfg_idx, high=high)
    shapes = get_test_shapes(dim)
    size = shapes[0]

    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Wavelet mode has slightly higher reconstruction error due to Meyer wavelet
    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize(
    "dim,high",
    [
        (2, "wavelet"),
        (3, "wavelet"),
        (4, "wavelet"),
    ],
)
def test_numpy_round_trip_wavelet_relative(dim, high, rng):
    """
    Test NumPy implementation round-trip with wavelet mode using relative tolerance.

    Tests with random shapes and configs (that have at least 2 scales for wavelet mode).
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)

    # Filter configs to only those with nscales >= 2 for wavelet mode
    valid_cfg_indices = [idx for idx, cfg in enumerate(configs) if len(cfg) >= 2]
    if not valid_cfg_indices:
        pytest.skip(f"No config with nscales >= 2 available for dimension {dim}")

    shape_idx = rng.integers(0, len(shapes))
    cfg_idx = valid_cfg_indices[rng.integers(0, len(valid_cfg_indices))]
    transform = setup_numpy_transform(
        dim, shape_idx=shape_idx, cfg_idx=cfg_idx, high=high
    )

    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4
    np.testing.assert_allclose(data, recon, atol=atol * data.max())
