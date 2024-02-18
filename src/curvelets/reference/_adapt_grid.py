from __future__ import annotations

import numpy as np


def adapt_grid(S1: np.ndarray, S2: np.ndarray):
    x1, x2 = np.meshgrid(S2, S1)

    t1 = np.zeros_like(x1, dtype=float)
    ind = (x1 != 0) & (np.abs(x2) <= np.abs(x1))
    t1[ind] = -x2[ind] / x1[ind]

    t2 = np.zeros_like(x1, dtype=float)
    ind = (x2 != 0) & (np.abs(x1) < np.abs(x2))
    t2[ind] = x1[ind] / x2[ind]

    t3 = t2.copy()
    t3[t2 < 0] = t2[t2 < 0] + 2
    t3[t2 > 0] = t2[t2 > 0] - 2

    M1 = t1 + t3
    M1[x1 >= 0] = -2

    t1 = np.zeros_like(x1, dtype=float)
    ind = (x2 != 0) & (abs(x1) <= abs(x2))
    t1[ind] = -x1[ind] / x2[ind]

    t2 = np.zeros_like(x1, dtype=float)
    ind = (x1 != 0) & (abs(x2) < abs(x1))
    t2[ind] = x2[ind] / x1[ind]

    t3 = t2.copy()
    t3[t2 < 0] = t2[t2 < 0] + 2
    t3[t2 > 0] = t2[t2 > 0] - 2

    M2 = t1 + t3
    M2[x2 >= 0] = -2

    return M2, M1
