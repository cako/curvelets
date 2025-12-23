from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt

from ._typing import ComplexFloatingNDArray, FloatingNDArray
from ._utils import meyer_window


class MeyerWavelet:
    """
    Multi-dimensional Meyer wavelet transform with pre-computed filters.

    This class provides forward and backward Meyer wavelet transforms with
    filters pre-computed during initialization for improved performance. The
    class stores highpass bands internally after forward transform, eliminating
    the need for external state management.

    Parameters
    ----------
    shape : tuple[int, ...]
        Expected shape of input signals. Used for validation and to determine
        the number of dimensions.

    Attributes
    ----------
    shape : tuple[int, ...]
        Expected signal shape.
    dimension : int
        Number of dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor import MeyerWavelet
    >>> wavelet = MeyerWavelet(shape=(64, 64))
    >>> signal = np.random.randn(64, 64)
    >>> lowpass = wavelet.forward(signal)
    >>> lowpass.shape
    (32, 32)
    >>> reconstructed = wavelet.backward(lowpass)
    >>> np.allclose(signal, reconstructed, atol=1e-10)
    True
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
            determine which filters to pre-compute.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> wavelet.shape
        (64, 64)
        >>> wavelet.dimension
        2
        >>> len(wavelet._filter_cache)
        1
        """
        self.shape = shape
        self.dimension = len(shape)
        self._highpass_bands: list[npt.NDArray] | None = None
        self._filters: dict[int, tuple[npt.NDArray, npt.NDArray]] = {}
        self._is_complex: bool | None = None

        # Pre-compute all required filters
        self._initialize_filters()

    def _initialize_filters(self) -> None:
        """
        Pre-compute all filters needed for the transform.

        Determines unique filter sizes from the input shape and pre-computes
        all required filters. Filters are stored in `_filter_cache` for
        direct access during forward and backward transforms.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> len(wavelet._filter_cache)
        1
        >>> 64 in wavelet._filter_cache
        True
        """
        # Determine unique filter sizes (one per unique dimension size)
        required_filter_sizes = set(self.shape)

        # Pre-compute all filters
        for signal_length in required_filter_sizes:
            self._filters[signal_length] = self._compute_single_filter(signal_length)

    def _compute_single_filter(
        self, signal_length: int
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Compute a single Meyer wavelet filter pair for given signal length.

        The lowpass filter is fftshifted, while the highpass filter is not.

        Parameters
        ----------
        signal_length : int
            Length of the signal along the transform dimension.

        Returns
        -------
        tuple[npt.NDArray, npt.NDArray]
            Lowpass and highpass filters as 1D arrays of length signal_length.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64,))
        >>> lowpass, highpass = wavelet._compute_single_filter(64)
        >>> lowpass.shape
        (64,)
        >>> highpass.shape
        (64,)
        """
        # Compute frequency grid
        frequency_step = 2 * np.pi / signal_length
        frequency_grid = (
            np.linspace(0, 2 * np.pi - frequency_step, signal_length) - np.pi / 2
        )

        # Meyer frequency parameters: [-pi/3, pi/3, 2*pi/3, 4*pi/3]
        meyer_frequency_parameters = np.pi * np.array([-1 / 3, 1 / 3, 2 / 3, 4 / 3])

        # Compute Meyer window function
        window_values = meyer_window(
            frequency_grid,
            meyer_frequency_parameters[0],  # transition_start
            meyer_frequency_parameters[1],  # plateau_start
            meyer_frequency_parameters[2],  # plateau_end
            meyer_frequency_parameters[3],  # transition_end
        )

        # Lowpass filter: fftshifted and square-rooted
        lowpass_filter = np.sqrt(np.fft.fftshift(window_values))

        # Highpass filter: not shifted, square-rooted
        highpass_filter = np.sqrt(window_values)

        return lowpass_filter, highpass_filter

    def _forward_transform_1d(
        self, signal: npt.NDArray, axis_index: int
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Apply 1D Meyer wavelet forward transform along specified axis.

        Parameters
        ----------
        signal : npt.NDArray
            Input array (real or complex).
        axis_index : int
            Axis along which to apply the transform.

        Returns
        -------
        tuple[npt.NDArray, npt.NDArray]
            Lowpass and highpass subbands. Output dtype matches input:
            real input produces real output, complex input produces complex output.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = np.random.randn(64, 64)
        >>> lowpass, highpass = wavelet._forward_transform_1d(signal, 0)
        >>> lowpass.shape
        (32, 64)
        >>> highpass.shape
        (32, 64)
        """
        last_axis_index = signal.ndim - 1
        signal = np.swapaxes(signal, axis_index, last_axis_index)
        signal_shape = signal.shape
        signal_length = signal_shape[-1]

        # Get pre-computed filters
        lowpass_filter, highpass_filter = self._filters[signal_length]

        # Reshape filters for broadcasting
        lowpass_filter = np.reshape(lowpass_filter, (1, signal_length))
        highpass_filter = np.reshape(highpass_filter, (1, signal_length))

        # Preserve input dtype
        input_dtype = signal.dtype
        is_complex_input = np.iscomplexobj(signal)

        # Transform to frequency domain
        signal_frequency_domain = np.fft.fft(signal, axis=last_axis_index)

        # Apply filters and transform back
        lowpass_full_resolution = np.fft.ifft(
            lowpass_filter * signal_frequency_domain, axis=last_axis_index
        )
        highpass_full_resolution = np.fft.ifft(
            highpass_filter * signal_frequency_domain, axis=last_axis_index
        )

        # Preserve complex values for complex input, take real for real input
        # Preserve the original dtype when taking real part or casting complex
        if not is_complex_input:
            # Take real part and cast back to original real dtype
            lowpass_full_resolution = lowpass_full_resolution.real
            highpass_full_resolution = highpass_full_resolution.real
        lowpass_full_resolution = lowpass_full_resolution.astype(input_dtype)
        highpass_full_resolution = highpass_full_resolution.astype(input_dtype)

        # Downsample by factor of 2 (take every other sample)
        lowpass_subband = lowpass_full_resolution[..., ::2]
        highpass_subband = highpass_full_resolution[..., 1::2]

        # Swap axes back to original order
        lowpass_subband = np.swapaxes(lowpass_subband, axis_index, last_axis_index)
        highpass_subband = np.swapaxes(highpass_subband, axis_index, last_axis_index)

        return lowpass_subband, highpass_subband

    def _inverse_transform_1d(
        self,
        lowpass_subband: npt.NDArray,
        highpass_subband: npt.NDArray,
        axis_index: int,
    ) -> npt.NDArray:
        """
        Apply 1D Meyer wavelet inverse transform along specified axis.

        Parameters
        ----------
        lowpass_subband : npt.NDArray
            Lowpass subband (real or complex).
        highpass_subband : npt.NDArray
            Highpass subband (real or complex).
        axis_index : int
            Axis along which to apply the transform.

        Returns
        -------
        npt.NDArray
            Reconstructed array. Output dtype matches input: real input produces
            real output, complex input produces complex output.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = np.random.randn(64, 64)
        >>> lowpass, highpass = wavelet._forward_transform_1d(signal, 0)
        >>> reconstructed = wavelet._inverse_transform_1d(lowpass, highpass, 0)
        >>> np.allclose(signal, reconstructed, atol=1e-10)
        True
        """
        last_axis_index = lowpass_subband.ndim - 1
        lowpass_subband = np.swapaxes(lowpass_subband, axis_index, last_axis_index)
        highpass_subband = np.swapaxes(highpass_subband, axis_index, last_axis_index)

        # Compute upsampled shape
        upsampled_shape = list(lowpass_subband.shape)
        upsampled_shape[-1] = 2 * upsampled_shape[-1]

        # Determine dtype based on input - preserve the original dtype
        is_complex = np.iscomplexobj(lowpass_subband) or np.iscomplexobj(
            highpass_subband
        )
        # Preserve the input dtype (use lowpass_subband as reference)
        input_dtype = lowpass_subband.dtype
        dtype = input_dtype

        # Pre-allocate upsampled arrays
        lowpass_upsampled = np.zeros(upsampled_shape, dtype=dtype)
        highpass_upsampled = np.zeros(upsampled_shape, dtype=dtype)

        # Interleave subbands (upsample by inserting zeros)
        lowpass_upsampled[..., ::2] = lowpass_subband
        highpass_upsampled[..., 1::2] = highpass_subband

        signal_length = upsampled_shape[-1]

        # Get pre-computed filters
        lowpass_filter, highpass_filter = self._filters[signal_length]

        # Reshape filters for broadcasting
        lowpass_filter = np.reshape(lowpass_filter, (1, signal_length))
        highpass_filter = np.reshape(highpass_filter, (1, signal_length))

        # Transform to frequency domain and combine
        combined_frequency_domain = lowpass_filter * np.fft.fft(
            lowpass_upsampled, axis=last_axis_index
        ) + highpass_filter * np.fft.fft(highpass_upsampled, axis=last_axis_index)

        # Transform back to spatial domain
        reconstructed_full_resolution = np.fft.ifft(
            combined_frequency_domain, axis=last_axis_index
        )

        # Preserve complex values for complex input, take real for real input
        # Preserve the original dtype when taking real part or casting complex
        if is_complex:
            # Cast back to original complex dtype (FFT may promote complex64 to complex128)
            reconstructed_signal = 2 * reconstructed_full_resolution
        else:
            # Take real part and cast back to original real dtype
            reconstructed_signal = 2 * reconstructed_full_resolution.real
        reconstructed_signal = reconstructed_signal.astype(input_dtype)

        return np.swapaxes(reconstructed_signal, axis_index, last_axis_index)

    @overload
    def forward(self, signal: ComplexFloatingNDArray) -> ComplexFloatingNDArray: ...

    @overload
    def forward(self, signal: FloatingNDArray) -> FloatingNDArray: ...  # type: ignore[overload-cannot-match]

    def forward(
        self, signal: FloatingNDArray | ComplexFloatingNDArray
    ) -> FloatingNDArray | ComplexFloatingNDArray:
        """
        Apply multi-dimensional Meyer wavelet forward transform.

        Decomposes the input signal into 2^dimension subbands. The first subband
        (lowpass) is returned, while the remaining highpass subbands are stored
        internally for use in backward().

        Parameters
        ----------
        signal : npt.NDArray
            Input array (real or complex). Must match the shape specified
            during initialization.

        Returns
        -------
        npt.NDArray
            Lowpass subband (first of 2^dimension subbands). Shape is
            approximately half the input shape in each dimension.

        Raises
        ------
        ValueError
            If signal shape does not match expected shape.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = np.random.randn(64, 64)
        >>> lowpass = wavelet.forward(signal)
        >>> lowpass.shape
        (32, 32)
        >>> len(wavelet._highpass_bands)
        3
        """
        # Validate input shape
        if signal.shape != self.shape:
            error_msg = f"Signal shape {signal.shape} does not match expected shape {self.shape}"
            raise ValueError(error_msg)

        # Track if input is complex
        self._is_complex = np.iscomplexobj(signal)

        # Start with the full signal
        current_bands = [signal]

        # Apply 1D transform along each dimension
        for dimension_index in range(self.dimension):
            new_bands: list[npt.NDArray] = []
            for band in current_bands:
                lowpass_subband, highpass_subband = self._forward_transform_1d(
                    band, dimension_index
                )
                new_bands.append(lowpass_subband)
                new_bands.append(highpass_subband)
            current_bands = new_bands

        # Store highpass bands (all except the first)
        self._highpass_bands = current_bands[1:]

        # Return only the lowpass subband (first band)
        return current_bands[0]

    @overload
    def backward(
        self, lowpass_subband: ComplexFloatingNDArray
    ) -> ComplexFloatingNDArray: ...

    @overload
    def backward(self, lowpass_subband: FloatingNDArray) -> FloatingNDArray: ...  # type: ignore[overload-cannot-match]

    def backward(
        self, lowpass_subband: FloatingNDArray | ComplexFloatingNDArray
    ) -> FloatingNDArray | ComplexFloatingNDArray:
        """
        Apply multi-dimensional Meyer wavelet inverse transform.

        Reconstructs the original signal from the lowpass subband and the
        highpass bands stored during forward().

        Parameters
        ----------
        lowpass_subband : npt.NDArray
            Lowpass subband from forward transform. Must match the shape
            returned by forward().

        Returns
        -------
        npt.NDArray
            Reconstructed signal with shape matching the original input.

        Raises
        ------
        RuntimeError
            If forward() has not been called first.
        ValueError
            If lowpass_subband shape does not match expected shape.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = np.random.randn(64, 64)
        >>> lowpass = wavelet.forward(signal)
        >>> reconstructed = wavelet.backward(lowpass)
        >>> np.allclose(signal, reconstructed, atol=1e-10)
        True
        """
        # Validate that forward() was called first
        if self._highpass_bands is None:
            error_msg = (
                "forward() must be called before backward(). "
                "Highpass bands are not available."
            )
            raise RuntimeError(error_msg)

        # Combine lowpass with stored highpass bands
        all_bands = [lowpass_subband, *self._highpass_bands]

        # Apply inverse transform along each dimension in reverse order
        current_bands = all_bands
        for dimension_index in range(self.dimension - 1, -1, -1):
            new_bands: list[npt.NDArray] = []
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
