from __future__ import annotations

import numpy as np

from curvelets.numpy import UDCT
from curvelets.utils import apply_along_wedges, array_split_nd, deepflatten, ndargmax


def test_ndargmax():
    x = np.zeros((10, 10, 10))
    x[1, 1, 1] = 1.0
    m = ndargmax(x)
    assert len(m) == 3
    assert np.unique(m).item() == 1


def test_deepflatten():
    flattened = list(deepflatten([[[[[[1], 2, [[3, 4]]]]]]]))
    np.testing.assert_equal(flattened, np.arange(1, 5))


def test_array_split_nd():
    ary = np.outer(1 + np.arange(2), 2 + np.arange(3))
    split = array_split_nd(ary, 2, 3)
    desired = [
        [np.array([[2]]), np.array([[3]]), np.array([[4]])],
        [np.array([[4]]), np.array([[6]]), np.array([[8]])],
    ]
    assert len(split) == len(desired)
    for s, d in zip(split, desired):
        assert len(s) == len(d)
    for si, di in zip(deepflatten(split), deepflatten(desired)):
        np.testing.assert_equal(si, di)


def test_apply_along_wedges():
    x = np.zeros((32, 32))
    C = UDCT(x.shape, num_scales=3, wedges_per_direction=3)
    y = C.forward(x)
    res = apply_along_wedges(y, lambda w, *_: w.shape)

    for si, di in zip(deepflatten(res), deepflatten(y)):
        np.testing.assert_equal(np.array(si), np.array(di.shape))
