from __future__ import annotations

from math import ceil

import numpy as np

from ._fun_meyer import fun_meyer


def angle_fun(Mgrid, direction, n, alpha):
    # % create 2-D grid function-------------------------------------------------

    # angle meyer window
    angd = 2 / n
    ang = angd * np.array([-alpha, alpha, 1 - alpha, 1 + alpha])

    Mang = []
    if direction == 1:
        for jn in range(1, ceil(n / 2) + 1):
            ang2 = -1 + (jn - 1) * angd + ang
            fang = fun_meyer(Mgrid, *ang2)
            Mang.append(fang[None, :])
    elif direction == 2:  # This is weird, both directions are the same code
        for jn in range(1, ceil(n / 2) + 1):
            ang2 = -1 + (jn - 1) * angd + ang
            fang = fun_meyer(Mgrid, *ang2)
            Mang.append(fang[None, :])
    else:
        raise ValueError("unrecognized direction")
    return np.concatenate(Mang, axis=0)
