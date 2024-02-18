from __future__ import annotations

import numpy as np


def upsamp(F, decim):
    assert F.ndim == len(decim)
    upsamp_shape = tuple(s * d for s, d in zip(F.shape, decim))
    D = np.zeros(upsamp_shape, dtype=F.dtype)
    D[tuple(slice(None, None, d) for d in decim)] = F[...]
    return D
