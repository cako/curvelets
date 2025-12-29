"""Meyer wavelet transform for PyTorch UDCT implementation."""

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from __future__ import annotations

import numpy as np
import torch

from ._utils import circular_shift, meyer_window


class MeyerWavelet:
    """
    Multi-dimensional Meyer wavelet transform with pre-computed filters.

    This class provides forward and backward Meyer wavelet transforms with
    filters pre-computed during initialization for improved performance. The
    forward transform returns all subbands in a nested list structure, and the
    backward transform accepts this full structure for reconstruction.

    The filter computation uses the same method as UDCT's `_create_bandpass_windows()`
    for num_scales=2, ensuring compatibility when used with UDCT in wavelet mode
    with num_scales=2 and high_frequency_mode="wavelet".

    Parameters
    ----------
    shape : tuple[int, ...]
        Expected shape of input signals. Used for validation and to determine
        the number of dimensions. All dimensions must be even (divisible by 2).

    Attributes
    ----------
    shape : tuple[int, ...]
        Expected signal shape (all dimensions are even).
    dimension : int
        Number of dimensions.

    Raises
    ------
    ValueError
        If any dimension in shape is odd (not divisible by 2).

    Notes
    -----
    This implementation requires all dimensions to be even. Odd-length signals
    are not supported due to the downsampling strategy used in the transform,
    which produces mismatched subband sizes that cannot be correctly
    reconstructed.

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch import MeyerWavelet
    >>> wavelet = MeyerWavelet(shape=(64, 64))
    >>> signal = torch.randn(64, 64)
    >>> coefficients = wavelet.forward(signal)
    >>> len(coefficients)  # 2 subband groups
    2
    >>> coefficients[0][0].shape  # Lowpass subband
    torch.Size([32, 32])
    >>> len(coefficients[1])  # Highpass subbands
    3
    >>> reconstructed = wavelet.backward(coefficients)
    >>> torch.allclose(signal, reconstructed, atol=1e-10)
    True
    >>> # Odd dimensions raise an error
    >>> try:
    ...     MeyerWavelet(shape=(65, 65))
    ... except ValueError:
    ...     print("Odd dimensions not supported")
    Odd dimensions not supported
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        """
        Initialize Meyer wavelet transform.

        All required filters are pre-computed during initialization based on
        the input shape. This ensures optimal performance during forward and
        backward transforms.

        Parameters
        ----------
        shape : tuple[int, ...]
            Expected shape of input signals. Used for validation and to
            determine which filters to pre-compute. All dimensions must be
            even (divisible by 2).

        Raises
        ------
        ValueError
            If any dimension in shape is odd (not divisible by 2).

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> wavelet.shape
        (64, 64)
        >>> wavelet.dimension
        2
        >>> len(wavelet._filters)
        1
        >>> # Odd dimensions raise an error
        >>> try:
        ...     MeyerWavelet(shape=(65, 65))
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: All dimensions must be even, got shape (65, 65) with odd dimensions at indices [0, 1]
        """
        # Validate that all dimensions are even
        odd_dimensions = [i for i, dim in enumerate(shape) if dim % 2 != 0]
        if odd_dimensions:
            error_msg = (
                f"All dimensions must be even, got shape {shape} "
                f"with odd dimensions at indices {odd_dimensions}"
            )
            raise ValueError(error_msg)

        self.shape = shape
        self.dimension = len(shape)
        self._filters: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # Pre-compute all required filters
        self._initialize_filters()

    def _initialize_filters(self) -> None:
        """
        Pre-compute all filters needed for the transform.

        Determines unique filter sizes from the input shape and pre-computes
        all required filters. Filters are stored in `_filters` for
        direct access during forward and backward transforms.
        """
        # Determine unique filter sizes (one per unique dimension size)
        required_filter_sizes = set(self.shape)

        # Pre-compute all filters
        for signal_length in required_filter_sizes:
            self._filters[signal_length] = self._compute_single_filter(signal_length)

    def _compute_single_filter(
        self, signal_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a single Meyer wavelet filter pair for given signal length.

        Uses the same computation method as UDCT's `_create_bandpass_windows()` for
        num_scales=2 compatibility. The lowpass filter is computed using:
        - Frequency grid: :math:`\\text{np.linspace}(-1.5\\pi, 0.5\\pi, \\text{signal\\_length}, \\text{endpoint=False})`
        - Meyer window parameters: :math:`[-2, -1, \\pi/3, 2\\pi/3]`
        - Special case for num_scales=2: adds :math:`\\text{meyer\\_window}(|\\text{freq} + 2\\pi|, ...)`
        - Circular shift by :math:`\\text{signal\\_length} // 4` before square root

        The highpass filter is computed as the complement of the lowpass to ensure
        perfect reconstruction: `|lowpass|² + |highpass|² = 1`.

        Parameters
        ----------
        signal_length : int
            Length of the signal along the transform dimension.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Lowpass and highpass filters as 1D tensors of length signal_length.
            The filters satisfy perfect reconstruction: `|lowpass|² + |highpass|² = 1`.
        """
        # Compute frequency grid (matching UDCT)
        frequency_grid = np.linspace(
            -1.5 * np.pi, 0.5 * np.pi, signal_length, endpoint=False
        )

        # Meyer window parameters (matching UDCT for num_scales=2)
        meyer_params = (-2.0, -1.0, np.pi / 3, 2 * np.pi / 3)
        abs_frequency_grid = np.abs(frequency_grid)

        # Convert to torch tensor for meyer_window
        abs_frequency_grid_tensor = torch.from_numpy(abs_frequency_grid)

        # Compute Meyer window function
        window_values = meyer_window(abs_frequency_grid_tensor, *meyer_params)

        # Add num_scales=2 special case (matching UDCT)
        freq_shifted = np.abs(frequency_grid + 2 * np.pi)
        freq_shifted_tensor = torch.from_numpy(freq_shifted)
        window_values += meyer_window(freq_shifted_tensor, *meyer_params)

        # Lowpass filter: circular-shifted and square-rooted (matching UDCT)
        lowpass_window_sq_shifted = circular_shift(window_values, (signal_length // 4,))
        lowpass_filter = torch.sqrt(lowpass_window_sq_shifted)

        # Highpass filter: complement of lowpass for perfect reconstruction
        # Compute as complement of the shifted lowpass window to ensure |lowpass|² + |highpass|² = 1
        highpass_filter = torch.sqrt(1.0 - lowpass_window_sq_shifted)

        return lowpass_filter, highpass_filter

    def _forward_transform_1d(
        self, signal: torch.Tensor, axis_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 1D Meyer wavelet forward transform along specified axis.

        Parameters
        ----------
        signal : torch.Tensor
            Input tensor (real or complex).
        axis_index : int
            Axis along which to apply the transform.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Lowpass and highpass subbands. Output dtype matches input:
            real input produces real output, complex input produces complex output.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = torch.randn(64, 64)
        >>> lowpass, highpass = wavelet._forward_transform_1d(signal, 0)
        >>> lowpass.shape
        torch.Size([32, 64])
        >>> highpass.shape
        torch.Size([32, 64])
        """
        last_axis_index = signal.ndim - 1
        signal = torch.swapaxes(signal, axis_index, last_axis_index)
        signal_shape = signal.shape
        signal_length = signal_shape[-1]

        # Get pre-computed filters
        lowpass_filter, highpass_filter = self._filters[signal_length]

        # Move filters to same device as signal
        lowpass_filter = lowpass_filter.to(signal.device)
        highpass_filter = highpass_filter.to(signal.device)

        # Reshape filters for broadcasting
        filter_shape = [1] * signal.ndim
        filter_shape[-1] = signal_length
        lowpass_filter = lowpass_filter.reshape(filter_shape)
        highpass_filter = highpass_filter.reshape(filter_shape)

        # Preserve input dtype
        input_dtype = signal.dtype
        is_complex_input = signal.is_complex()

        # Transform to frequency domain
        signal_frequency_domain = torch.fft.fft(signal, dim=last_axis_index)  # pylint: disable=not-callable

        # Apply filters and transform back
        lowpass_full_resolution = torch.fft.ifft(  # pylint: disable=not-callable
            lowpass_filter.to(signal_frequency_domain.dtype) * signal_frequency_domain,
            dim=last_axis_index,
        )
        highpass_full_resolution = torch.fft.ifft(  # pylint: disable=not-callable
            highpass_filter.to(signal_frequency_domain.dtype) * signal_frequency_domain,
            dim=last_axis_index,
        )

        # Preserve complex values for complex input, take real for real input
        if not is_complex_input:
            # Take real part and cast back to original real dtype
            lowpass_full_resolution = lowpass_full_resolution.real
            highpass_full_resolution = highpass_full_resolution.real
        lowpass_full_resolution = lowpass_full_resolution.to(input_dtype)
        highpass_full_resolution = highpass_full_resolution.to(input_dtype)

        # Downsample by factor of 2 (take every other sample)
        lowpass_subband = lowpass_full_resolution[..., ::2]
        highpass_subband = highpass_full_resolution[..., 1::2]

        # Swap axes back to original order
        lowpass_subband = torch.swapaxes(lowpass_subband, axis_index, last_axis_index)
        highpass_subband = torch.swapaxes(highpass_subband, axis_index, last_axis_index)

        return lowpass_subband, highpass_subband

    def _inverse_transform_1d(
        self,
        lowpass_subband: torch.Tensor,
        highpass_subband: torch.Tensor,
        axis_index: int,
    ) -> torch.Tensor:
        """
        Apply 1D Meyer wavelet inverse transform along specified axis.

        Parameters
        ----------
        lowpass_subband : torch.Tensor
            Lowpass subband (real or complex).
        highpass_subband : torch.Tensor
            Highpass subband (real or complex).
        axis_index : int
            Axis along which to apply the transform.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor. Output dtype matches input: real input produces
            real output, complex input produces complex output.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = torch.randn(64, 64)
        >>> lowpass, highpass = wavelet._forward_transform_1d(signal, 0)
        >>> reconstructed = wavelet._inverse_transform_1d(lowpass, highpass, 0)
        >>> torch.allclose(signal, reconstructed, atol=1e-10)
        True
        """
        last_axis_index = lowpass_subband.ndim - 1
        lowpass_subband = torch.swapaxes(lowpass_subband, axis_index, last_axis_index)
        highpass_subband = torch.swapaxes(highpass_subband, axis_index, last_axis_index)

        # Compute upsampled shape
        upsampled_shape = list(lowpass_subband.shape)
        upsampled_shape[-1] = 2 * upsampled_shape[-1]

        # Determine dtype based on input - preserve the original dtype
        is_complex = lowpass_subband.is_complex() or highpass_subband.is_complex()

        # Determine the appropriate dtype for upsampled arrays
        if is_complex:
            if lowpass_subband.is_complex():
                dtype = lowpass_subband.dtype
            elif highpass_subband.is_complex():
                dtype = highpass_subband.dtype
            else:
                dtype = torch.result_type(lowpass_subband, highpass_subband)
        else:
            dtype = lowpass_subband.dtype

        # Preserve the input dtype for final output casting
        input_dtype = lowpass_subband.dtype

        # Pre-allocate upsampled arrays
        lowpass_upsampled = torch.zeros(
            upsampled_shape, dtype=dtype, device=lowpass_subband.device
        )
        highpass_upsampled = torch.zeros(
            upsampled_shape, dtype=dtype, device=highpass_subband.device
        )

        # Interleave subbands (upsample by inserting zeros)
        lowpass_upsampled[..., ::2] = lowpass_subband
        highpass_upsampled[..., 1::2] = highpass_subband

        signal_length = upsampled_shape[-1]

        # Get pre-computed filters
        lowpass_filter, highpass_filter = self._filters[signal_length]

        # Move filters to same device as subbands
        lowpass_filter = lowpass_filter.to(lowpass_upsampled.device)
        highpass_filter = highpass_filter.to(highpass_upsampled.device)

        # Reshape filters for broadcasting
        filter_shape = [1] * lowpass_upsampled.ndim
        filter_shape[-1] = signal_length
        lowpass_filter = lowpass_filter.reshape(filter_shape)
        highpass_filter = highpass_filter.reshape(filter_shape)

        # Transform to frequency domain and combine
        combined_frequency_domain = lowpass_filter.to(dtype) * torch.fft.fft(  # pylint: disable=not-callable
            lowpass_upsampled, dim=last_axis_index
        ) + highpass_filter.to(dtype) * torch.fft.fft(  # pylint: disable=not-callable
            highpass_upsampled, dim=last_axis_index
        )

        # Transform back to spatial domain
        reconstructed_full_resolution = torch.fft.ifft(  # pylint: disable=not-callable
            combined_frequency_domain, dim=last_axis_index
        )

        # Preserve complex values for complex input, take real for real input
        if is_complex:
            # Cast back to appropriate complex dtype
            reconstructed_signal = 2 * reconstructed_full_resolution
            reconstructed_signal = reconstructed_signal.to(dtype)
        else:
            # Take real part and cast back to original real dtype
            reconstructed_signal = 2 * reconstructed_full_resolution.real
            reconstructed_signal = reconstructed_signal.to(input_dtype)

        return torch.swapaxes(reconstructed_signal, axis_index, last_axis_index)

    def forward(self, signal: torch.Tensor) -> list[list[torch.Tensor]]:
        """
        Apply multi-dimensional Meyer wavelet forward transform.

        Decomposes the input signal into 2^dimension subbands organized into 2
        subband groups. Returns all subbands in a nested list structure similar
        to UDCT.

        Parameters
        ----------
        signal : torch.Tensor
            Input tensor (real or complex). Must match the shape specified
            during initialization.

        Returns
        -------
        list[list[torch.Tensor]]
            All subbands organized into 2 subband groups:

            - coefficients[0]: [lowpass] - single lowpass subband (1 subband)
            - coefficients[1]: [highpass_0, highpass_1, ...] - all highpass
              subbands (2^dimension - 1 subbands)

            Each subband has shape approximately half the input shape in each
            dimension.

        Raises
        ------
        ValueError
            If signal shape does not match expected shape.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = torch.randn(64, 64)
        >>> coefficients = wavelet.forward(signal)
        >>> len(coefficients)  # 2 subband groups
        2
        >>> coefficients[0][0].shape  # Lowpass subband
        torch.Size([32, 32])
        >>> len(coefficients[1])  # Highpass subbands
        3
        """
        # Validate input shape
        if signal.shape != self.shape:
            error_msg = f"Signal shape {signal.shape} does not match expected shape {self.shape}"
            raise ValueError(error_msg)

        # Start with the full signal
        current_bands = [signal]

        # Apply 1D transform along each dimension
        for dimension_index in range(self.dimension):
            new_bands: list[torch.Tensor] = []
            for band in current_bands:
                lowpass_subband, highpass_subband = self._forward_transform_1d(
                    band, dimension_index
                )
                new_bands.append(lowpass_subband)
                new_bands.append(highpass_subband)
            current_bands = new_bands

        # Return structure: [[lowpass], [highpass_0, highpass_1, ...]]
        # len(result) == 2 (num_subbands)
        return [[current_bands[0]], current_bands[1:]]

    def backward(self, coefficients: list[list[torch.Tensor]]) -> torch.Tensor:
        """
        Apply multi-dimensional Meyer wavelet inverse transform.

        Reconstructs the original signal from the full coefficient structure
        returned by forward().

        Parameters
        ----------
        coefficients : list[list[torch.Tensor]]
            Full coefficient structure from forward() with 2 subband groups:

            - coefficients[0]: [lowpass] - single lowpass subband
            - coefficients[1]: [highpass_0, highpass_1, ...] - all highpass
              subbands

        Returns
        -------
        torch.Tensor
            Reconstructed signal with shape matching the original input.

        Raises
        ------
        ValueError
            If coefficients structure is invalid (must have 2 subband groups).

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = torch.randn(64, 64)
        >>> coefficients = wavelet.forward(signal)
        >>> reconstructed = wavelet.backward(coefficients)
        >>> torch.allclose(signal, reconstructed, atol=1e-10)
        True
        """
        # Validate coefficient structure
        if len(coefficients) != 2:
            error_msg = (
                f"coefficients must have 2 subband groups, got {len(coefficients)}"
            )
            raise ValueError(error_msg)

        # Extract lowpass and highpass bands
        lowpass_subband = coefficients[0][0]
        highpass_bands: list[torch.Tensor] = coefficients[1]

        # Combine lowpass with highpass bands
        all_bands = [lowpass_subband, *highpass_bands]

        # Apply inverse transform along each dimension in reverse order
        current_bands = all_bands
        for dimension_index in range(self.dimension - 1, -1, -1):
            new_bands: list[torch.Tensor] = []
            for band_index in range(len(current_bands) // 2):
                reconstructed = self._inverse_transform_1d(
                    current_bands[2 * band_index],
                    current_bands[2 * band_index + 1],
                    dimension_index,
                )
                new_bands.append(reconstructed)
            current_bands = new_bands

        # Return the fully reconstructed signal
        return current_bands[0]
