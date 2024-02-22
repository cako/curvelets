# import matplotlib.pyplot as plt
from __future__ import annotations

import numpy as np

from ._travel import travel


def angle_kron(angle_fun, nper, param_udct):
    # , nper, param_udct
    krsz = np.ones(3, dtype=int)
    krsz[0] = np.prod(param_udct.size[: nper[0] - 1])
    krsz[1] = np.prod(param_udct.size[nper[0] : nper[1] - 1])
    krsz[2] = np.prod(param_udct.size[nper[1] : param_udct.dim])

    tmp = angle_fun

    tmp1 = np.kron(np.ones((krsz[1], 1), dtype=int), tmp)
    tmp2 = np.kron(np.ones((krsz[2], 1), dtype=int), travel(tmp1)).ravel()
    tmp3 = travel(np.kron(tmp2, np.ones((krsz[0], 1), dtype=int)))
    ang_md = tmp3.reshape(*param_udct.size[::-1]).T

    return ang_md
