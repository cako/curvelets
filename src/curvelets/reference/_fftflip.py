from __future__ import annotations

import numpy as np

from ._circshift import circshift


# flip M-D matrix function ------------------------------------------------
def fftflip(F, axis):
    Fc = F
    dim = F.ndim
    shiftvec = np.zeros((dim,), dtype=int)
    shiftvec[axis] = 1
    Fc = np.flip(F, axis)
    Fc = circshift(Fc, shiftvec)
    return Fc
