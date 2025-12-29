"""Tests that nwedges=4,5 raise ValueError.

These tests verify that nwedges=4,5 raise appropriate errors, as they are not
supported according to the Nguyen & Chauris (2010) paper specification. The
decimation ratio formula requires integer division by 3, so wedges must be
divisible by 3.
"""

from __future__ import annotations

import numpy as np
import pytest

from curvelets.numpy import UDCT


@pytest.mark.parametrize("num_scales", [2, 3, 4])  # type: ignore[misc]
@pytest.mark.parametrize("dim", [2, 3])  # type: ignore[misc]
def test_non_multiple_of_3_raises_error(
    num_scales: int, dim: int, rng: np.random.Generator
) -> None:
    """Test that wedges_per_direction that is not a multiple of 3 raises ValueError.

    Parameters
    ----------
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
    >>> import pytest
    >>> with pytest.raises(ValueError, match="divisible by 3"):
    ...     UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=4)
    """
    # Select appropriate shape based on dimension
    if dim == 2:
        shape: tuple[int, ...] = (64, 64)
    elif dim == 3:
        shape = (32, 32, 32)
    else:
        pytest.skip(f"Dimension {dim} not supported in this test")

    # Generate a random non-multiple of 3
    # Start with a random number >= 1, multiply by 3, then add 1 or 2
    # This ensures the result is >= 4 and not a multiple of 3
    base = rng.integers(1, 7)  # Random number between 1 and 6
    offset = rng.choice([1, 2])  # Randomly add 1 or 2
    wedges_per_direction = base * 3 + offset

    # Verify that ValueError is raised
    with pytest.raises(ValueError, match="divisible by 3"):
        UDCT(
            shape=shape,
            num_scales=num_scales,
            wedges_per_direction=wedges_per_direction,
        )


@pytest.mark.parametrize("wedges_per_direction", [4, 5])  # type: ignore[misc]
@pytest.mark.parametrize("num_scales", [2, 3])  # type: ignore[misc]
def test_nwedges_4_5_raises_error_complex_transform(
    wedges_per_direction: int, num_scales: int
) -> None:
    """Test that wedges_per_direction=4,5 raises ValueError with complex transform.

    Parameters
    ----------
    wedges_per_direction : int
        Number of wedges per direction (4 or 5).
    num_scales : int
        Number of scales (2 or 3).

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> import pytest
    >>> with pytest.raises(ValueError, match="divisible by 3"):
    ...     UDCT(
    ...         shape=(64, 64),
    ...         num_scales=3,
    ...         wedges_per_direction=4,
    ...         transform_kind="complex"
    ...     )
    """
    shape = (64, 64)

    # Verify that ValueError is raised
    with pytest.raises(ValueError, match="divisible by 3"):
        UDCT(
            shape=shape,
            num_scales=num_scales,
            wedges_per_direction=wedges_per_direction,
            transform_kind="complex",
        )


@pytest.mark.parametrize("wedges_per_direction", [4, 5])  # type: ignore[misc]
def test_nwedges_4_5_raises_error_angular_config(
    wedges_per_direction: int,
) -> None:
    """Test that angular_wedges_config with nwedges=4,5 raises ValueError.

    Parameters
    ----------
    wedges_per_direction : int
        Number of wedges per direction (4 or 5).

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> import pytest
    >>> cfg = np.array([[4, 4], [8, 8]])
    >>> with pytest.raises(ValueError, match="divisible by 3"):
    ...     UDCT(shape=(64, 64), angular_wedges_config=cfg)
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

    # Verify that ValueError is raised
    with pytest.raises(ValueError, match="divisible by 3"):
        UDCT(shape=shape, angular_wedges_config=angular_wedges_config)


def test_angular_config_mixed_invalid_values() -> None:
    """Test that angular_wedges_config with mixed valid/invalid values raises error.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> import pytest
    >>> # Config with some invalid values should raise error
    >>> cfg = np.array([[3, 4], [6, 6]])  # 4 is not divisible by 3
    >>> with pytest.raises(ValueError, match="divisible by 3"):
    ...     UDCT(shape=(64, 64), angular_wedges_config=cfg)
    """
    shape = (64, 64)

    # Test with mixed valid/invalid values
    angular_wedges_config = np.array([[3, 4], [6, 6]])  # 4 is not divisible by 3

    # Verify that ValueError is raised
    with pytest.raises(ValueError, match="divisible by 3"):
        UDCT(shape=shape, angular_wedges_config=angular_wedges_config)
