from __future__ import annotations

# pylint: disable=duplicate-code
# Duplicate code with torch implementation is expected
import sys
from dataclasses import dataclass, field

if sys.version_info >= (3, 10):
    from typing import overload
else:
    from typing_extensions import overload

import numpy as np
import numpy.typing as npt

from ._typing import C, F, IntegerNDArray


@dataclass(**({"kw_only": True} if sys.version_info >= (3, 10) else {}))
class ParamUDCT:
    """
    Parameters for Uniform Discrete Curvelet Transform (UDCT).

    This dataclass stores all configuration parameters needed for the UDCT
    transform, including dimensionality, shape, angular wedge configuration,
    and window parameters.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the input data (e.g., (64, 64) for 2D, (32, 32, 32) for 3D).
    angular_wedges_config : IntegerNDArray
        Configuration array specifying the number of angular wedges per scale
        and dimension. Shape is (num_scales - 1, ndim), where num_scales includes
        the lowpass scale. The last dimension must equal `len(shape)`. Each row
        corresponds to a high-frequency scale (scales 1 to num_scales-1), each
        column to a dimension.
    window_overlap : float
        Window overlap parameter controlling the smoothness of window transitions.
        Typically between 0.1 and 0.3. Higher values create smoother transitions
        but may reduce directional selectivity.
    radial_frequency_params : tuple[float, float, float, float]
        Four parameters defining radial frequency bands for Meyer wavelet
        decomposition: (transition_start, plateau_start, plateau_end, transition_end).
        These define the frequency ranges for the bandpass filters.
    window_threshold : float
        Threshold for sparse window storage. Window values below this threshold
        are stored in sparse format. Typically 1e-5 to 1e-6.

    Attributes
    ----------
    ndim : int
        Number of dimensions of the transform. Computed automatically from
        `len(shape)`.
    num_scales : int
        Total number of scales (including lowpass scale 0). Computed automatically
        as 1 + the first dimension of `angular_wedges_config`.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._utils import ParamUDCT
    >>>
    >>> # Create parameters for 2D transform with 4 scales total (1 lowpass + 3 high-frequency)
    >>> params = ParamUDCT(
    ...     shape=(64, 64),
    ...     angular_wedges_config=np.array([[3], [6], [12]]),
    ...     window_overlap=0.15,
    ...     radial_frequency_params=(np.pi/3, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3),
    ...     window_threshold=1e-5
    ... )
    >>> params.ndim  # Number of dimensions (computed from shape)
    2
    >>> params.num_scales  # Total number of scales (including lowpass)
    4
    >>>
    >>> # Create parameters for 3D transform with 3 scales total (1 lowpass + 2 high-frequency)
    >>> params_3d = ParamUDCT(
    ...     shape=(32, 32, 32),
    ...     angular_wedges_config=np.array([[3, 3, 3], [6, 6, 6]]),
    ...     window_overlap=0.15,
    ...     radial_frequency_params=(np.pi/3, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3),
    ...     window_threshold=1e-5
    ... )
    >>> params_3d.ndim
    3
    >>> params_3d.num_scales  # Total number of scales (including lowpass)
    3
    >>> params_3d.shape
    (32, 32, 32)
    """

    shape: tuple[int, ...]
    angular_wedges_config: IntegerNDArray  # last dimension  == len(shape)
    window_overlap: float
    radial_frequency_params: tuple[float, float, float, float]
    window_threshold: float
    ndim: int = field(init=False)
    num_scales: int = field(init=False)

    def __post_init__(self) -> None:
        self.ndim = len(self.shape)
        self.num_scales = 1 + len(self.angular_wedges_config)
        # Validate that angular_wedges_config matches shape dimensionality
        if self.angular_wedges_config.shape[1] != self.ndim:
            msg = (
                f"angular_wedges_config last dimension ({self.angular_wedges_config.shape[1]}) "
                f"must equal shape length ({self.ndim})"
            )
            raise ValueError(msg)


@overload
def circular_shift(array: npt.NDArray[F], shift: tuple[int, ...]) -> npt.NDArray[F]: ...


@overload
def circular_shift(array: npt.NDArray[C], shift: tuple[int, ...]) -> npt.NDArray[C]: ...  # type: ignore[overload-cannot-match]


def circular_shift(
    array: npt.NDArray[np.generic], shift: tuple[int, ...]
) -> npt.NDArray[np.generic]:
    """
    Circularly shift array along all axes.

    This function performs a circular shift (also known as cyclic shift) of
    the input array along all dimensions. The shift amount is specified per
    dimension in the `shift` tuple. Positive values shift in one direction,
    negative values shift in the opposite direction.

    Parameters
    ----------
    array : npt.NDArray[F] | npt.NDArray[C]
        Input array to shift. Can be real-valued (npt.NDArray[F]) or
        complex-valued (npt.NDArray[C]). The array is shifted along
        all axes simultaneously.
    shift : tuple[int, ...]
        Shift amounts for each dimension. Must have length equal to `array.ndim`.
        Each element specifies the number of positions to shift along the
        corresponding axis. Positive values shift forward, negative values
        shift backward.

    Returns
    -------
    npt.NDArray[F] | npt.NDArray[C]
        Circularly shifted array with the same shape and dtype as input.
        Returns npt.NDArray[F] if input is real, npt.NDArray[C] if
        input is complex.

    Notes
    -----
    The circular shift wraps around: elements shifted beyond the array boundary
    appear at the opposite end. This is implemented using `np.roll`, which
    performs the shift along each axis sequentially.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._utils import circular_shift
    >>>
    >>> # 1D array shift
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> shifted = circular_shift(arr, (2,))
    >>> shifted
    array([4, 5, 1, 2, 3])
    >>>
    >>> # 2D array shift
    >>> arr_2d = np.array([[1, 2], [3, 4]])
    >>> shifted_2d = circular_shift(arr_2d, (1, 1))
    >>> shifted_2d
    array([[4, 3],
           [2, 1]])
    >>>
    >>> # Complex array (preserves dtype)
    >>> arr_complex = np.array([1+2j, 3+4j, 5+6j])
    >>> shifted_complex = circular_shift(arr_complex, (1,))
    >>> shifted_complex.dtype
    dtype('complex128')
    """
    assert array.ndim == len(shift)
    # np.roll preserves dtype, so return type matches input type via overloads
    return np.roll(array, shift, axis=tuple(range(len(shift))))


def downsample(
    array: npt.NDArray[np.generic], decimation_ratios: IntegerNDArray
) -> npt.NDArray[np.generic]:
    """
    Downsample array by selecting every Nth element along each dimension.

    This function downsamples an array by selecting elements at regular intervals
    specified by the decimation ratios. For each dimension, every `decimation_ratios[i]`-th
    element is selected, effectively reducing the array size.

    Parameters
    ----------
    array : npt.NDArray[npt.DTypeLike]
        Input array to downsample. Can be any dtype. The array is downsampled
        along all dimensions simultaneously.
    decimation_ratios : IntegerNDArray
        Decimation ratios for each dimension. Must have length equal to `array.ndim`.
        Each element specifies the step size for selecting elements along the
        corresponding axis. For example, a value of 2 means every 2nd element
        is selected.

    Returns
    -------
    npt.NDArray[npt.DTypeLike]
        Downsampled array with the same dtype as input. The shape is reduced
        by the decimation ratios: output_shape[i] = ceil(input_shape[i] / decimation_ratios[i]).

    Notes
    -----
    The downsampling is performed using slice notation with step sizes. This is
    equivalent to `array[::d0, ::d1, ...]` where `d0, d1, ...` are the decimation
    ratios. The output preserves the dtype of the input array.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._utils import downsample
    >>>
    >>> # 1D downsampling
    >>> arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> decim = np.array([2])
    >>> downsampled = downsample(arr, decim)
    >>> downsampled
    array([1, 3, 5, 7])
    >>>
    >>> # 2D downsampling
    >>> arr_2d = np.arange(16).reshape(4, 4)
    >>> decim_2d = np.array([2, 2])
    >>> downsampled_2d = downsample(arr_2d, decim_2d)
    >>> downsampled_2d.shape
    (2, 2)
    >>> downsampled_2d
    array([[ 0,  2],
           [ 8, 10]])
    >>>
    >>> # Preserves dtype
    >>> arr_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    >>> downsampled_float32 = downsample(arr_float32, np.array([2]))
    >>> downsampled_float32.dtype
    dtype('float32')
    """
    assert array.ndim == len(decimation_ratios)
    return array[tuple(slice(None, None, d) for d in decimation_ratios)]


@overload
def flip_fft_all_axes(array: npt.NDArray[F]) -> npt.NDArray[F]: ...


@overload
def flip_fft_all_axes(array: npt.NDArray[C]) -> npt.NDArray[C]: ...  # type: ignore[overload-cannot-match]


def flip_fft_all_axes(
    array: npt.NDArray[np.generic],
) -> npt.NDArray[np.generic]:
    """
    Apply fftflip to all axes of an array.

    This function transforms the frequency domain representation X(omega) to
    X(-omega) by flipping all axes and applying a circular shift. This is
    commonly used in curvelet transforms to handle negative frequency
    components.

    The operation consists of:
    1. Flipping the array along all axes (reversing each dimension)
    2. Circularly shifting by 1 position along all axes to maintain proper
       frequency alignment

    Parameters
    ----------
    array : npt.NDArray[F] | npt.NDArray[C]
        Input array in FFT representation. Can be real-valued (npt.NDArray[F])
        or complex-valued (npt.NDArray[C]). The array is flipped and
        shifted along all dimensions simultaneously.

    Returns
    -------
    npt.NDArray[F] | npt.NDArray[C]
        Flipped array representing negative frequencies. Returns npt.NDArray[F]
        if input is real, npt.NDArray[C] if input is complex. The output
        has the same shape and dtype as the input.

    Notes
    -----
    This operation is the frequency-domain equivalent of time-reversal. After
    flipping and shifting, the array represents the negative frequency
    components of the original signal. The circular shift by 1 ensures that
    the zero frequency component is properly aligned.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._utils import flip_fft_all_axes
    >>>
    >>> # 1D array
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> flipped = flip_fft_all_axes(arr)
    >>> flipped
    array([5, 1, 2, 3, 4])
    >>>
    >>> # 2D array
    >>> arr_2d = np.array([[1, 2], [3, 4]])
    >>> flipped_2d = flip_fft_all_axes(arr_2d)
    >>> flipped_2d
    array([[4, 3],
           [2, 1]])
    >>>
    >>> # Complex array (preserves dtype)
    >>> arr_complex = np.array([1+2j, 3+4j, 5+6j])
    >>> flipped_complex = flip_fft_all_axes(arr_complex)
    >>> flipped_complex.dtype
    dtype('complex128')
    """
    flipped_array = array.copy()
    for axis in range(array.ndim):
        flipped_array = np.flip(flipped_array, axis)
    shift_vector = tuple(1 for _ in range(array.ndim))
    # Overloads handle type narrowing at call sites
    return circular_shift(flipped_array, shift_vector)


# Meyer transition polynomial coefficients for smooth window transitions
# This is a 7th-degree polynomial (with trailing zeros) used to create
# smooth transitions in the Meyer wavelet window function. The polynomial
# provides C^infinity smoothness at the transition boundaries.
MEYER_TRANSITION_POLYNOMIAL: npt.NDArray[np.floating[object]] = np.array(
    [-20.0, 70.0, -84.0, 35.0, 0.0, 0.0, 0.0, 0.0], dtype=float
)


def meyer_window(
    frequency: npt.NDArray[F],
    transition_start: float,
    plateau_start: float,
    plateau_end: float,
    transition_end: float,
) -> npt.NDArray[F]:
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
    >>> from curvelets.numpy._utils import meyer_window
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
    # Handle case where transition_start == plateau_start (no transition)
    if transition_start != plateau_start:
        rising_mask = (frequency >= transition_start) & (frequency <= plateau_start)
        if np.any(rising_mask):
            normalized_freq = (frequency[rising_mask] - transition_start) / (
                plateau_start - transition_start
            )
            window_values[rising_mask] = np.polyval(
                MEYER_TRANSITION_POLYNOMIAL, normalized_freq
            )
    # If transition_start == plateau_start, frequencies at that boundary
    # are handled by the plateau region below

    # Region 2: Constant plateau between plateau_start and plateau_end
    # Include frequencies at plateau_start and plateau_end boundaries
    plateau_mask = (frequency >= plateau_start) & (frequency <= plateau_end)
    window_values[plateau_mask] = 1.0

    # Region 3: Falling transition from plateau_end to transition_end
    # Normalize to [0, 1] (reversed) and apply polynomial
    # Handle case where plateau_end == transition_end (no transition)
    if plateau_end != transition_end:
        falling_mask = (frequency >= plateau_end) & (frequency <= transition_end)
        if np.any(falling_mask):
            normalized_freq = (frequency[falling_mask] - transition_end) / (
                plateau_end - transition_end
            )
            window_values[falling_mask] = np.polyval(
                MEYER_TRANSITION_POLYNOMIAL, normalized_freq
            )
    # If plateau_end == transition_end, frequencies at that boundary
    # remain 0 (already set by initialization)

    return window_values


def upsample(
    array: npt.NDArray[np.generic], decimation_ratios: IntegerNDArray
) -> npt.NDArray[np.generic]:
    """
    Upsample array by inserting zeros at regular intervals.

    This function upsamples an array by creating a larger array and placing
    the original array elements at positions specified by the decimation ratios.
    All other positions are filled with zeros. The output shape is the input
    shape multiplied by the decimation ratios.

    Parameters
    ----------
    array : npt.NDArray[npt.DTypeLike]
        Input array to upsample. Can be any dtype. The array is upsampled
        along all dimensions simultaneously.
    decimation_ratios : IntegerNDArray
        Upsampling ratios for each dimension. Must have length equal to `array.ndim`.
        Each element specifies the spacing between original elements in the
        upsampled array. For example, a value of 2 means original elements
        are placed every 2nd position, with zeros in between.

    Returns
    -------
    npt.NDArray[npt.DTypeLike]
        Upsampled array with the same dtype as input. The shape is increased
        by the decimation ratios: output_shape[i] = input_shape[i] * decimation_ratios[i].

    Notes
    -----
    The upsampling is performed by creating a zero-filled array of the target
    size, then placing the original array elements at positions `[::d0, ::d1, ...]`
    where `d0, d1, ...` are the decimation ratios. This is the inverse operation
    of `downsample` when applied with the same ratios.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._utils import upsample
    >>>
    >>> # 1D upsampling
    >>> arr = np.array([1, 2, 3, 4])
    >>> decim = np.array([2])
    >>> upsampled = upsample(arr, decim)
    >>> upsampled
    array([1, 0, 2, 0, 3, 0, 4, 0])
    >>>
    >>> # 2D upsampling
    >>> arr_2d = np.array([[1, 2], [3, 4]])
    >>> decim_2d = np.array([2, 2])
    >>> upsampled_2d = upsample(arr_2d, decim_2d)
    >>> upsampled_2d.shape
    (4, 4)
    >>> upsampled_2d
    array([[1, 0, 2, 0],
           [0, 0, 0, 0],
           [3, 0, 4, 0],
           [0, 0, 0, 0]])
    >>>
    >>> # Preserves dtype
    >>> arr_complex = np.array([1+2j, 3+4j], dtype=np.complex64)
    >>> upsampled_complex = upsample(arr_complex, np.array([2]))
    >>> upsampled_complex.dtype
    dtype('complex64')
    """
    assert array.ndim == len(decimation_ratios)
    upsampled_shape = tuple(s * d for s, d in zip(array.shape, decimation_ratios))
    upsampled_array = np.zeros(upsampled_shape, dtype=array.dtype)
    upsampled_array[tuple(slice(None, None, d) for d in decimation_ratios)] = array[...]
    return upsampled_array


def from_sparse_new(
    arr_list: tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]:
    """
    Identity function for sparse array format.

    This function is kept for backward compatibility. It simply returns the
    input tuple unchanged. The sparse format is (indices, values).

    Parameters
    ----------
    arr_list : tuple[NDArray[intp], NDArray[floating]]
        Sparse array format as a tuple of (indices, values).

    Returns
    -------
    tuple[NDArray[intp], NDArray[floating]]
        The input tuple unchanged.

    Notes
    -----
    This function is internal and used primarily in tests. It exists for
    compatibility with the old API.
    """
    return arr_list
