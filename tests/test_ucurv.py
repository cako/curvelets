from __future__ import annotations

import numpy as np
import pytest

from curvelets.ucurv import ucurv
from curvelets.ucurv.util import bands2vec, vec2bands

eps = 1e-6
shapes = [
    [
        [256, 256],
    ],
    [
        [32, 32, 32],
    ],
    [
        [16, 16, 16, 16],
    ],
]

configurations = [
    [
        [[3, 3]],
        [[6, 6]],
        [[12, 12]],
        [[12, 12], [24, 24]],
        [[12, 12], [3, 3], [6, 6]],
        [[12, 12], [3, 3], [6, 6], [24, 24]],
    ],
    [
        [[3, 3, 3]],
        [[6, 6, 6]],
        [[12, 12, 12]],
        [[12, 12, 12], [24, 24, 24]],
        # [[12, 12, 12], [3, 3, 3], [6, 6, 6]],
        # [[12, 12, 12], [3, 3, 3], [6, 6, 6], [12, 24, 24]],
    ],
    [
        [[3, 3, 3, 3]],
        #  [[6, 6, 6, 6]],
        #  [[12, 12, 12, 12]],
        #  [[12, 12, 12, 12], [24, 24, 24, 24]],
        #  [[12, 12, 12, 12], [3, 3, 3, 3], [6, 6, 6, 6]],
        #  [[12, 12, 12, 12], [3, 3, 3, 3], [6, 6, 6, 6], [12, 24, 24, 24]],
    ],
]

combinations = [
    (shape, config)
    for shape_list, config_list in zip(shapes, configurations)
    for shape in shape_list
    for config in config_list
]


@pytest.mark.parametrize("shape, cfg", combinations)
def test_ucurv(shape, cfg):
    data = np.random.rand(*shape)
    udct = ucurv.udct(shape, cfg)
    band = ucurv.ucurvfwd(data, udct)
    recon = ucurv.ucurvinv(band, udct)
    are_close = np.all(np.isclose(data, recon, atol=eps))
    assert are_close == True


@pytest.mark.parametrize("shape, cfg", combinations)
def test_vectorize(shape, cfg):
    data = np.random.rand(*shape)
    udct = ucurv.udct(shape, cfg)
    band = ucurv.ucurvfwd(data, udct)
    flat = bands2vec(band)
    unflat = vec2bands(flat, udct)
    recon = ucurv.ucurvinv(unflat, udct)
    are_close = np.all(np.isclose(data, recon, atol=eps))
    assert are_close == True
