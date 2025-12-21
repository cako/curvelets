from __future__ import annotations

import sys
from dataclasses import dataclass, field
from math import ceil, prod
from typing import TypeVar

import numpy as np
import numpy.typing as npt

D_T = TypeVar("D_T", bound=np.floating)


@dataclass(**({"kw_only": True} if sys.version_info >= (3, 10) else {}))
class ParamUDCT:
    dim: int
    size: tuple[int, ...]
    angular_wedges_config: npt.NDArray[np.int_]  # last dimension  == dim
    window_overlap: float
    radial_frequency_params: tuple[float, float, float, float]
    window_threshold: float
    len: int = field(init=False)
    res: int = field(init=False)
    decim: npt.NDArray[np.int_] = field(init=False)
    ind: dict[int, dict[int, np.ndarray]] | None = None
    dec: dict[int, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.len = prod(self.size)
        self.res = len(self.angular_wedges_config)
        self.decim = 2 * (np.asarray(self.angular_wedges_config, dtype=int) // 3)


def circshift(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    assert arr.ndim == len(shape)
    return np.roll(arr, shape, axis=tuple(range(len(shape))))


def adapt_grid(S1: np.ndarray, S2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x1, x2 = np.meshgrid(S2, S1)

    t1: npt.NDArray[np.floating] = np.zeros_like(x1, dtype=float)
    ind = (x1 != 0) & (np.abs(x2) <= np.abs(x1))
    t1[ind] = -x2[ind] / x1[ind]

    t2: npt.NDArray[np.floating] = np.zeros_like(x1, dtype=float)
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


def angle_fun(
    Mgrid: np.ndarray, direction: int, n: int, window_overlap: float
) -> np.ndarray:
    # % create 2-D grid function-------------------------------------------------

    # angle meyer window
    angd = 2 / n
    ang = angd * np.array(
        [-window_overlap, window_overlap, 1 - window_overlap, 1 + window_overlap]
    )

    Mang = []
    # This is weird, both directions are the same code
    if direction in (1, 2):
        for jn in range(1, ceil(n / 2) + 1):
            ang2 = -1 + (jn - 1) * angd + ang
            fang = meyer_window(Mgrid, *ang2)
            Mang.append(fang[None, :])
    else:
        msg = "Unrecognized direction"
        raise ValueError(msg)
    return np.concatenate(Mang, axis=0)


def angle_kron(
    angle_arr: np.ndarray, nper: np.ndarray, param_udct: ParamUDCT
) -> np.ndarray:
    # , nper, param_udct
    krsz: npt.NDArray[np.int_] = np.ones(3, dtype=int)
    krsz[0] = np.prod(param_udct.size[: nper[0] - 1])
    krsz[1] = np.prod(param_udct.size[nper[0] : nper[1] - 1])
    krsz[2] = np.prod(param_udct.size[nper[1] : param_udct.dim])

    tmp1 = np.kron(np.ones((krsz[1], 1), dtype=int), angle_arr)
    tmp2 = np.kron(np.ones((krsz[2], 1), dtype=int), tmp1.ravel()).ravel()
    tmp3 = np.kron(tmp2, np.ones((krsz[0], 1), dtype=int)).ravel()
    return tmp3.reshape(*param_udct.size)


def downsamp(F: np.ndarray, decim: np.ndarray) -> np.ndarray:
    assert F.ndim == len(decim)
    return F[tuple(slice(None, None, d) for d in decim)]


def fftflip(F: np.ndarray, axis: int) -> np.ndarray:
    Fc = F
    dim = F.ndim
    shiftvec: npt.NDArray[np.int_] = np.zeros((dim,), dtype=int)
    shiftvec[axis] = 1
    Fc = np.flip(F, axis)
    return circshift(Fc, tuple(shiftvec))


def _fftflip_all_axes(F: np.ndarray) -> np.ndarray:
    """
    Apply fftflip to all axes of an array.

    This produces X(-omega) from X(omega) in FFT representation.
    After flipping, the array is circshifted by 1 in each dimension
    to maintain proper frequency alignment.

    Parameters
    ----------
    F : np.ndarray
        Input array in FFT representation.

    Returns
    -------
    np.ndarray
        Flipped array representing negative frequencies.
    """
    Fc = F.copy()
    for axis in range(F.ndim):
        Fc = np.flip(Fc, axis)
    shiftvec = tuple(1 for _ in range(F.ndim))
    return circshift(Fc, shiftvec)


# Meyer transition polynomial coefficients for smooth window transitions
# This is a 7th-degree polynomial (with trailing zeros) used to create
# smooth transitions in the Meyer wavelet window function. The polynomial
# provides C^infinity smoothness at the transition boundaries.
MEYER_TRANSITION_POLYNOMIAL: npt.NDArray[np.floating] = np.array(
    [-20.0, 70.0, -84.0, 35.0, 0.0, 0.0, 0.0, 0.0], dtype=float
)


def meyer_window(
    frequency: np.ndarray,
    transition_start: float,
    plateau_start: float,
    plateau_end: float,
    transition_end: float,
) -> np.ndarray:
    """
    Compute Meyer wavelet window function with polynomial transitions.

    This function creates a window function with three distinct regions:
    1. Rising transition: smooth polynomial transition from 0 to 1
       between transition_start and plateau_start
    2. Plateau: constant value of 1.0 between plateau_start and plateau_end
    3. Falling transition: smooth polynomial transition from 1 to 0
       between plateau_end and transition_end

    Values outside the [transition_start, transition_end] range are set to 0.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency values at which to evaluate the window function.
        Can be any shape; output will have the same shape.
    transition_start : float
        Start of the rising transition region. Window value is 0 below this.
    plateau_start : float
        End of rising transition and start of constant plateau region.
        Window value reaches 1.0 at this point.
    plateau_end : float
        End of constant plateau region. Window value is 1.0 up to this point.
    transition_end : float
        End of falling transition region. Window value is 0 above this.

    Returns
    -------
    np.ndarray
        Window function values with same shape as frequency input.
        Values range from 0.0 to 1.0.

    Notes
    -----
    The polynomial transitions use a 7th-degree polynomial to ensure
    C^infinity smoothness at the boundaries. The polynomial is evaluated
    on normalized coordinates [0, 1] within each transition region.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor._utils import meyer_window
    >>>
    >>> # Standard Meyer wavelet parameters
    >>> frequency = np.linspace(-np.pi, 2*np.pi, 100)
    >>> params = np.pi * np.array([-1/3, 1/3, 2/3, 4/3])
    >>> window = meyer_window(frequency, params[0], params[1], params[2], params[3])
    >>>
    >>> # Verify plateau region is 1.0
    >>> plateau_mask = (frequency > params[1]) & (frequency <= params[2])
    >>> np.allclose(window[plateau_mask], 1.0)
    True
    >>>
    >>> # Verify outside boundaries is 0.0
    >>> outside_mask = (frequency < params[0]) | (frequency > params[3])
    >>> np.allclose(window[outside_mask], 0.0)
    True
    >>>
    >>> # Verify window shape matches input
    >>> window.shape == frequency.shape
    True
    """
    window_values = np.zeros_like(frequency)

    # Region 1: Rising transition from transition_start to plateau_start
    # Normalize to [0, 1] and apply polynomial
    rising_mask = (frequency >= transition_start) & (frequency <= plateau_start)
    if np.any(rising_mask):
        normalized_freq = (frequency[rising_mask] - transition_start) / (
            plateau_start - transition_start
        )
        window_values[rising_mask] = np.polyval(
            MEYER_TRANSITION_POLYNOMIAL, normalized_freq
        )

    # Region 2: Constant plateau between plateau_start and plateau_end
    plateau_mask = (frequency > plateau_start) & (frequency <= plateau_end)
    window_values[plateau_mask] = 1.0

    # Region 3: Falling transition from plateau_end to transition_end
    # Normalize to [0, 1] (reversed) and apply polynomial
    falling_mask = (frequency >= plateau_end) & (frequency <= transition_end)
    if np.any(falling_mask):
        normalized_freq = (frequency[falling_mask] - transition_end) / (
            plateau_end - transition_end
        )
        window_values[falling_mask] = np.polyval(
            MEYER_TRANSITION_POLYNOMIAL, normalized_freq
        )

    return window_values


def to_sparse(
    arr: npt.NDArray[D_T], thresh: float
) -> tuple[npt.NDArray[np.intp], npt.NDArray[D_T]]:
    idx = np.argwhere(arr.ravel() > thresh)
    return (idx, arr.ravel()[idx])


def upsamp(F: np.ndarray, decim: np.ndarray) -> np.ndarray:
    assert F.ndim == len(decim)
    upsamp_shape = tuple(s * d for s, d in zip(F.shape, decim))
    D = np.zeros(upsamp_shape, dtype=F.dtype)
    D[tuple(slice(None, None, d) for d in decim)] = F[...]
    return D
