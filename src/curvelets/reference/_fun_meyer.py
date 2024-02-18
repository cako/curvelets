from __future__ import annotations

import numpy as np

# from numba import njit
# from numpy.polynomial import Polynomial

# @njit
# def eval_meyer_polynomial(x):
#     return 35.0 * x**4 - 84.0 * x**5 + 70.0 * x**6 - 20.0 * x**7


# @njit
# def fun_meyer_numba(x, p1, p2, p3, p4):
#     y = np.zeros_like(x)

#     win = (x >= p1) & (x <= p2)
#     y[win] = eval_meyer_polynomial((x[win] - p1) / (p2 - p1))

#     win = (x > p2) & (x <= p3)
#     y[win] = 1.0

#     win = (x >= p3) & (x <= p4)
#     y[win] = eval_meyer_polynomial((x[win] - p4) / (p3 - p4))
#     return y


# def fun_meyer_new(x, p1, p2, p3, p4):
#     p = Polynomial([0.0, 0.0, 0.0, 0.0, 35.0, -84.0, 70.0, -20.0])
#     y = np.zeros_like(x)

#     win = (x >= p1) & (x <= p2)
#     y[win] = p((x[win] - p1) / (p2 - p1))

#     win = (x > p2) & (x <= p3)
#     y[win] = 1.0

#     win = (x >= p3) & (x <= p4)
#     y[win] = p((x[win] - p4) / (p3 - p4))
#     return y


def fun_meyer_original(x, p1, p2, p3, p4):
    p = np.array([-20.0, 70.0, -84.0, 35.0, 0.0, 0.0, 0.0, 0.0])
    y = np.zeros_like(x)

    win = (x >= p1) & (x <= p2)
    y[win] = np.polyval(p, (x[win] - p1) / (p2 - p1))

    win = (x > p2) & (x <= p3)
    y[win] = 1.0

    win = (x >= p3) & (x <= p4)
    y[win] = np.polyval(p, (x[win] - p4) / (p3 - p4))
    return y


fun_meyer = fun_meyer_original
