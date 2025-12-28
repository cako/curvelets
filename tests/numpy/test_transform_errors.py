"""Test for uncovered forward transform error paths.

Tests cover:
- Lines 520-526: Complex input to real transform error
- Lines 759-765: Fall-through case when use_complex_transform=True but input is not complex
- Line 776: Fall-through case when use_complex_transform=False but input is complex
"""

from __future__ import annotations

import numpy as np
import pytest

from curvelets.numpy import UDCT
from curvelets.numpy._forward_transform import _apply_forward_transform


def test_complex_input_to_real_transform_error(rng):
    """
    Test error when complex array is passed to real transform (lines 520-526).

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator fixture.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> from curvelets.numpy._forward_transform import _apply_forward_transform
    >>> transform = UDCT(shape=(64, 64), use_complex_transform=False)
    >>> data = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
    >>> try:
    ...     _apply_forward_transform(
    ...         data, transform.parameters, transform.windows,
    ...         transform.decimation_ratios, use_complex_transform=False
    ...     )
    ... except ValueError as e:
    ...     print("Error caught:", str(e))
    Error caught: Real transform requires real-valued input...
    """
    # Create transform to get parameters, windows, and decimation_ratios
    transform = UDCT(
        shape=(64, 64),
        num_scales=3,
        wedges_per_direction=3,
        use_complex_transform=False,
    )

    # Create complex input
    data = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))

    # Call _apply_forward_transform directly with use_complex_transform=False
    # This should trigger the error at lines 520-526
    with pytest.raises(
        ValueError,
        match="Real transform requires real-valued input.*Use use_complex_transform=True",
    ):
        _apply_forward_transform(
            data,
            transform.parameters,
            transform.windows,
            transform.decimation_ratios,
            use_complex_transform=False,
        )


def test_forward_complex_transform_with_real_input(rng):
    """
    Test complex transform mode with real input (lines 759-765).

    When use_complex_transform=True but input is not complex, the function
    should still work by falling through to the complex transform path.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator fixture.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> from curvelets.numpy._forward_transform import _apply_forward_transform
    >>> transform = UDCT(shape=(64, 64), use_complex_transform=True)
    >>> data = np.random.randn(64, 64)  # Real input
    >>> coeffs = _apply_forward_transform(
    ...     data, transform.parameters, transform.windows,
    ...     transform.decimation_ratios, use_complex_transform=True
    ... )
    >>> len(coeffs)  # Should have coefficients
    3
    """
    # Create transform with complex transform mode
    transform = UDCT(
        shape=(64, 64),
        num_scales=3,
        wedges_per_direction=3,
        use_complex_transform=True,
    )

    # Create real input (not complex)
    data = rng.normal(size=(64, 64)).astype(np.float64)

    # Call _apply_forward_transform with use_complex_transform=True
    # This should fall through to complex transform even though input is real
    # (lines 759-765)
    coeffs = _apply_forward_transform(
        data,
        transform.parameters,
        transform.windows,
        transform.decimation_ratios,
        use_complex_transform=True,
    )

    # Verify coefficients structure is valid
    assert len(coeffs) == transform.parameters.num_scales
    assert len(coeffs[0]) == 1  # Low frequency band
    assert len(coeffs[0][0]) == 1  # Single wedge at low frequency


def test_forward_real_transform_with_complex_input(rng):
    """
    Test real transform mode with complex input (line 776).

    When use_complex_transform=False but input is complex, the function
    should fall through to the real transform path which will then raise
    an error.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator fixture.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> from curvelets.numpy._forward_transform import _apply_forward_transform
    >>> transform = UDCT(shape=(64, 64), use_complex_transform=False)
    >>> data = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
    >>> try:
    ...     _apply_forward_transform(
    ...         data, transform.parameters,
    ...         transform.windows, transform.decimation_ratios,
    ...         use_complex_transform=False
    ...     )
    ... except ValueError:
    ...     print("Error caught as expected")
    Error caught as expected
    """
    # Create transform with real transform mode
    transform = UDCT(
        shape=(64, 64),
        num_scales=3,
        wedges_per_direction=3,
        use_complex_transform=False,
    )

    # Create complex input
    data = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))

    # Call _apply_forward_transform with use_complex_transform=False
    # This should fall through to real transform path (line 776) which will
    # then raise an error because input is complex
    with pytest.raises(
        ValueError,
        match="Real transform requires real-valued input.*Use use_complex_transform=True",
    ):
        _apply_forward_transform(
            data,
            transform.parameters,
            transform.windows,
            transform.decimation_ratios,
            use_complex_transform=False,
        )
