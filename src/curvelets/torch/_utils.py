"""Utility functions for PyTorch UDCT implementation."""

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import torch


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
    angular_wedges_config : torch.Tensor
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
    >>> import torch
    >>> from curvelets.torch._utils import ParamUDCT
    >>>
    >>> # Create parameters for 2D transform with 4 scales total (1 lowpass + 3 high-frequency)
    >>> params = ParamUDCT(
    ...     shape=(64, 64),
    ...     angular_wedges_config=torch.tensor([[3], [6], [12]]),
    ...     window_overlap=0.15,
    ...     radial_frequency_params=(torch.pi/3, 2*torch.pi/3, 2*torch.pi/3, 4*torch.pi/3),
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
    ...     angular_wedges_config=torch.tensor([[3, 3, 3], [6, 6, 6]]),
    ...     window_overlap=0.15,
    ...     radial_frequency_params=(torch.pi/3, 2*torch.pi/3, 2*torch.pi/3, 4*torch.pi/3),
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
    angular_wedges_config: torch.Tensor
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


def circular_shift(tensor: torch.Tensor, shift: tuple[int, ...]) -> torch.Tensor:
    """
    Circularly shift tensor along all axes.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to shift.
    shift : tuple[int, ...]
        Shift amounts for each dimension.

    Returns
    -------
    torch.Tensor
        Circularly shifted tensor with the same shape and dtype as input.
    """
    assert tensor.ndim == len(shift)
    return torch.roll(tensor, shifts=shift, dims=tuple(range(len(shift))))


def downsample(tensor: torch.Tensor, decimation_ratios: torch.Tensor) -> torch.Tensor:
    """
    Downsample tensor by selecting every Nth element along each dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to downsample.
    decimation_ratios : torch.Tensor
        Decimation ratios for each dimension.

    Returns
    -------
    torch.Tensor
        Downsampled tensor.
    """
    assert tensor.ndim == len(decimation_ratios)
    # Convert to list of ints for slicing
    ratios = decimation_ratios.tolist()
    return tensor[tuple(slice(None, None, int(d)) for d in ratios)]


def upsample(tensor: torch.Tensor, decimation_ratios: torch.Tensor) -> torch.Tensor:
    """
    Upsample tensor by inserting zeros at regular intervals.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to upsample.
    decimation_ratios : torch.Tensor
        Upsampling ratios for each dimension.

    Returns
    -------
    torch.Tensor
        Upsampled tensor with zeros inserted.
    """
    assert tensor.ndim == len(decimation_ratios)
    ratios = decimation_ratios.tolist()
    upsampled_shape = tuple(int(s * d) for s, d in zip(tensor.shape, ratios))
    upsampled_tensor = torch.zeros(
        upsampled_shape, dtype=tensor.dtype, device=tensor.device
    )
    upsampled_tensor[tuple(slice(None, None, int(d)) for d in ratios)] = tensor
    return upsampled_tensor


def flip_fft_all_axes(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply fftflip to all axes of a tensor.

    This function transforms the frequency domain representation X(omega) to
    X(-omega) by flipping all axes and applying a circular shift.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor in FFT representation.

    Returns
    -------
    torch.Tensor
        Flipped tensor representing negative frequencies.
    """
    flipped_tensor = tensor.clone()
    for axis in range(tensor.ndim):
        flipped_tensor = torch.flip(flipped_tensor, dims=(axis,))
    shift_vector = tuple(1 for _ in range(tensor.ndim))
    return circular_shift(flipped_tensor, shift_vector)


# Meyer transition polynomial coefficients for smooth window transitions
MEYER_TRANSITION_POLYNOMIAL = torch.tensor(
    [-20.0, 70.0, -84.0, 35.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64
)


def meyer_window(
    frequency: torch.Tensor,
    transition_start: float,
    plateau_start: float,
    plateau_end: float,
    transition_end: float,
) -> torch.Tensor:
    """
    Compute Meyer wavelet window function with polynomial transitions.

    Parameters
    ----------
    frequency : torch.Tensor
        Frequency values at which to evaluate the window function.
    transition_start : float
        Start of the rising transition region.
    plateau_start : float
        End of rising transition and start of constant plateau region.
    plateau_end : float
        End of constant plateau region.
    transition_end : float
        End of falling transition region.

    Returns
    -------
    torch.Tensor
        Window function values with same shape as frequency input.
    """
    # Use numpy for polyval since PyTorch doesn't have it
    # Convert to numpy, compute, convert back
    # Store original device and dtype to restore later
    original_device = frequency.device
    original_dtype = frequency.dtype
    # Move to CPU for NumPy conversion (required for GPU tensors)
    freq_np = frequency.cpu().numpy()
    window_values = np.zeros_like(freq_np)

    poly_coeffs = np.array([-20.0, 70.0, -84.0, 35.0, 0.0, 0.0, 0.0, 0.0])

    # Region 1: Rising transition from transition_start to plateau_start
    if transition_start != plateau_start:
        rising_mask = (freq_np >= transition_start) & (freq_np <= plateau_start)
        if np.any(rising_mask):
            normalized_freq = (freq_np[rising_mask] - transition_start) / (
                plateau_start - transition_start
            )
            window_values[rising_mask] = np.polyval(poly_coeffs, normalized_freq)

    # Region 2: Constant plateau between plateau_start and plateau_end
    plateau_mask = (freq_np >= plateau_start) & (freq_np <= plateau_end)
    window_values[plateau_mask] = 1.0

    # Region 3: Falling transition from plateau_end to transition_end
    if plateau_end != transition_end:
        falling_mask = (freq_np >= plateau_end) & (freq_np <= transition_end)
        if np.any(falling_mask):
            normalized_freq = (freq_np[falling_mask] - transition_end) / (
                plateau_end - transition_end
            )
            window_values[falling_mask] = np.polyval(poly_coeffs, normalized_freq)

    return torch.from_numpy(window_values).to(
        dtype=original_dtype, device=original_device
    )


def from_sparse_new(
    arr_list: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Identity function for sparse array format.

    Parameters
    ----------
    arr_list : tuple[torch.Tensor, torch.Tensor]
        Sparse array format as a tuple of (indices, values).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The input tuple unchanged.
    """
    return arr_list
