"""Copying and pickling the NumPy UDCT."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable

import numpy as np
import pytest

from curvelets.numpy import UDCT
from curvelets.numpy._fft import available_fft_engines

SHAPE = (32, 32)

COPIES: dict[str, Callable[[UDCT], UDCT]] = {
    "deepcopy": copy.deepcopy,
    "pickle": lambda obj: pickle.loads(pickle.dumps(obj)),
}


@pytest.mark.parametrize("how", list(COPIES))  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("engine", available_fft_engines())  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("transform_kind", ["real", "complex", "monogenic"])  # type: ignore[untyped-decorator]
def test_copy_of_used_transform_matches(
    transform_kind: str, engine: str, how: str
) -> None:
    """A transform that has run (FFT engine and caches set up) copies to one
    with the same settings and results."""
    udct = UDCT(
        shape=SHAPE,
        num_scales=3,
        wedges_per_direction=3,
        transform_kind=transform_kind,  # type: ignore[arg-type]
        fft_engine=engine,  # type: ignore[arg-type]
    )
    x = np.random.default_rng(0).standard_normal(SHAPE)
    coeffs = udct.forward(x)

    clone = COPIES[how](udct)
    assert clone.fft_engine == udct.fft_engine
    assert clone.transform_kind == udct.transform_kind
    np.testing.assert_allclose(clone.vect(clone.forward(x)), udct.vect(coeffs))
    np.testing.assert_allclose(clone.backward(coeffs), udct.backward(coeffs))
