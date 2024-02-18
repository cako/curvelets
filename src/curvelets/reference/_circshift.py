from __future__ import annotations

from numpy import roll


def circshift(arr, shape):
    assert arr.ndim == len(shape)
    return roll(arr, shape, axis=tuple(range(len(shape))))
