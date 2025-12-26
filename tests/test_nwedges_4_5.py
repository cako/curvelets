"""Tests for nwedges=4 and nwedges=5 cases.

These tests specifically target cases where wedges_per_direction=4 and 5,
which have been reported to fail. This test suite verifies round-trip
reconstruction accuracy for these configurations.
"""

from __future__ import annotations

import numpy as np
import pytest

from curvelets.numpy import UDCT


@pytest.mark.parametrize("wedges_per_direction", [4, 5])
@pytest.mark.parametrize("num_scales", [2, 3, 4])
@pytest.mark.parametrize("dim", [2, 3])
def test_round_trip_nwedges_4_5_curvelet(
    wedges_per_direction: int, num_scales: int, dim: int, rng: np.random.Generator
) -> None:
    """Test round-trip reconstruction for nwedges=4,5 in curvelet mode.

    Parameters
    ----------
    wedges_per_direction : int
        Number of wedges per direction (4 or 5).
    num_scales : int
        Number of scales (2, 3, or 4).
    dim : int
        Dimension (2 or 3).
    rng : numpy.random.Generator
        Random number generator fixture.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> rng = np.random.default_rng(42)
    >>> transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=4)
    >>> data = rng.normal(size=(64, 64))
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon, atol=1e-4)
    True
    """
    # Select appropriate shape based on dimension
    if dim == 2:
        shape = (64, 64)
    elif dim == 3:
        shape = (32, 32, 32)
    else:
        pytest.skip(f"Dimension {dim} not supported in this test")

    # Create transform
    transform = UDCT(
        shape=shape,
        num_scales=num_scales,
        wedges_per_direction=wedges_per_direction,
        high_frequency_mode="curvelet",
    )

    # Generate test data
    data = rng.normal(size=shape).astype(np.float64)

    # Apply forward and backward transforms
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify reconstruction accuracy
    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, recon, atol=atol, rtol=1e-5)


@pytest.mark.parametrize("wedges_per_direction", [4, 5])
@pytest.mark.parametrize("num_scales", [2, 3, 4])
@pytest.mark.parametrize("dim", [2, 3])
def test_round_trip_nwedges_4_5_wavelet(
    wedges_per_direction: int, num_scales: int, dim: int, rng: np.random.Generator
) -> None:
    """Test round-trip reconstruction for nwedges=4,5 in wavelet mode.

    Parameters
    ----------
    wedges_per_direction : int
        Number of wedges per direction (4 or 5).
    num_scales : int
        Number of scales (2, 3, or 4).
    dim : int
        Dimension (2 or 3).
    rng : numpy.random.Generator
        Random number generator fixture.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> rng = np.random.default_rng(42)
    >>> transform = UDCT(
    ...     shape=(64, 64),
    ...     num_scales=3,
    ...     wedges_per_direction=4,
    ...     high_frequency_mode="wavelet"
    ... )
    >>> data = rng.normal(size=(64, 64))
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon, atol=1e-4)
    True
    """
    # Select appropriate shape based on dimension
    if dim == 2:
        shape = (64, 64)
    elif dim == 3:
        shape = (32, 32, 32)
    else:
        pytest.skip(f"Dimension {dim} not supported in this test")

    # Create transform
    transform = UDCT(
        shape=shape,
        num_scales=num_scales,
        wedges_per_direction=wedges_per_direction,
        high_frequency_mode="wavelet",
    )

    # Generate test data
    data = rng.normal(size=shape).astype(np.float64)

    # Apply forward and backward transforms
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify reconstruction accuracy
    # Wavelet mode may have slightly higher reconstruction error
    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, recon, atol=atol, rtol=1e-5)


@pytest.mark.parametrize("wedges_per_direction", [4, 5])
@pytest.mark.parametrize("num_scales", [2, 3])
@pytest.mark.parametrize("dim", [2, 3])
def test_round_trip_nwedges_4_5_complex(
    wedges_per_direction: int, num_scales: int, dim: int, rng: np.random.Generator
) -> None:
    """Test round-trip reconstruction for nwedges=4,5 with complex transform.

    Parameters
    ----------
    wedges_per_direction : int
        Number of wedges per direction (4 or 5).
    num_scales : int
        Number of scales (2 or 3).
    dim : int
        Dimension (2 or 3).
    rng : numpy.random.Generator
        Random number generator fixture.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> rng = np.random.default_rng(42)
    >>> transform = UDCT(
    ...     shape=(64, 64),
    ...     num_scales=3,
    ...     wedges_per_direction=4,
    ...     use_complex_transform=True
    ... )
    >>> data = rng.normal(size=(64, 64))
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon, atol=1e-4)
    True
    """
    # Select appropriate shape based on dimension
    if dim == 2:
        shape = (64, 64)
    elif dim == 3:
        shape = (32, 32, 32)
    else:
        pytest.skip(f"Dimension {dim} not supported in this test")

    # Create transform with complex transform enabled
    transform = UDCT(
        shape=shape,
        num_scales=num_scales,
        wedges_per_direction=wedges_per_direction,
        use_complex_transform=True,
    )

    # Generate test data
    data = rng.normal(size=shape).astype(np.float64)

    # Apply forward and backward transforms
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify reconstruction accuracy
    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, recon, atol=atol, rtol=1e-5)


@pytest.mark.parametrize("wedges_per_direction", [4, 5])
@pytest.mark.parametrize("num_scales", [2, 3])
def test_round_trip_nwedges_4_5_complex_input(
    wedges_per_direction: int, num_scales: int, rng: np.random.Generator
) -> None:
    """Test round-trip reconstruction for nwedges=4,5 with complex-valued input.

    Parameters
    ----------
    wedges_per_direction : int
        Number of wedges per direction (4 or 5).
    num_scales : int
        Number of scales (2 or 3).
    rng : numpy.random.Generator
        Random number generator fixture.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> rng = np.random.default_rng(42)
    >>> transform = UDCT(
    ...     shape=(64, 64),
    ...     num_scales=3,
    ...     wedges_per_direction=4,
    ...     use_complex_transform=True
    ... )
    >>> data = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon, atol=1e-4)
    True
    """
    shape = (64, 64)

    # Create transform with complex transform enabled
    transform = UDCT(
        shape=shape,
        num_scales=num_scales,
        wedges_per_direction=wedges_per_direction,
        use_complex_transform=True,
    )

    # Generate complex-valued test data
    data = (
        rng.normal(size=shape).astype(np.float64)
        + 1j * rng.normal(size=shape).astype(np.float64)
    )

    # Apply forward and backward transforms
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify output is complex
    assert np.iscomplexobj(recon), "Output should be complex for complex=True"

    # Verify reconstruction accuracy
    atol = 1e-4
    np.testing.assert_allclose(data, recon, atol=atol, rtol=1e-5)


@pytest.mark.parametrize("wedges_per_direction", [4, 5])
def test_round_trip_nwedges_4_5_angular_config(
    wedges_per_direction: int, rng: np.random.Generator
) -> None:
    """Test round-trip reconstruction for nwedges=4,5 using angular_wedges_config.

    Parameters
    ----------
    wedges_per_direction : int
        Number of wedges per direction (4 or 5).
    rng : numpy.random.Generator
        Random number generator fixture.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> rng = np.random.default_rng(42)
    >>> cfg = np.array([[4, 4], [8, 8]])
    >>> transform = UDCT(shape=(64, 64), angular_wedges_config=cfg)
    >>> data = rng.normal(size=(64, 64))
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon, atol=1e-4)
    True
    """
    shape = (64, 64)

    # Create angular_wedges_config with nwedges=4 or 5
    # For 3 scales: [wedges_per_direction, wedges_per_direction*2]
    angular_wedges_config = np.array(
        [
            [wedges_per_direction, wedges_per_direction],
            [wedges_per_direction * 2, wedges_per_direction * 2],
        ]
    )

    # Create transform using angular_wedges_config
    transform = UDCT(shape=shape, angular_wedges_config=angular_wedges_config)

    # Generate test data
    data = rng.normal(size=shape).astype(np.float64)

    # Apply forward and backward transforms
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify reconstruction accuracy
    atol = 1e-4
    np.testing.assert_allclose(data, recon, atol=atol, rtol=1e-5)

