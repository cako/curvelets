from __future__ import annotations

import logging
import sys
from math import prod
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

if sys.version_info <= (3, 9):
    from typing import List  # noqa: UP035
else:
    List = list

from ._backward_transform import _apply_backward_transform
from ._backward_transform_monogenic import _apply_backward_transform_monogenic
from ._forward_transform import (
    _apply_forward_transform_complex,
    _apply_forward_transform_monogenic,
    _apply_forward_transform_real,
)
from ._riesz import riesz_filters
from ._typing import C, F, MUDCTCoefficients, UDCTWindows, _to_complex_dtype
from ._udct_windows import UDCTWindow
from ._utils import ParamUDCT, from_sparse_new


class UDCT:
    """
    Uniform Discrete Curvelet Transform (UDCT) implementation.

    This class provides forward and backward curvelet transforms with support
    for both real and complex transforms.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the input data.
    angular_wedges_config : :obj:`np.ndarray <numpy.ndarray>`, optional
        Configuration array specifying the number of angular wedges per scale
        and dimension. Shape is (num_scales - 1, dimension), where num_scales
        includes the lowpass scale. If provided, cannot be used together with
        num_scales/wedges_per_direction. Default is None.
    num_scales : int, optional
        Total number of scales (including lowpass scale 0). Must be >= 2.
        Used when angular_wedges_config is not provided. Default is 3.
    wedges_per_direction : int, optional
        Number of angular wedges per direction at the coarsest scale.
        The number of wedges doubles at each finer scale. Must be >= 3.
        Used when angular_wedges_config is not provided. Default is 3.
    window_overlap : float, optional
        Window overlap parameter controlling the smoothness of window transitions.
        If None and using num_scales/wedges_per_direction, automatically chosen
        based on wedges_per_direction. Default is None (auto) or 0.15.
    radial_frequency_params : tuple[float, float, float, float], optional
        Radial frequency parameters defining the frequency bands.
        Default is (pi/3, 2*pi/3, 2*pi/3, 4*pi/3).
    window_threshold : float, optional
        Threshold for sparse window storage (values below this are stored as sparse).
        Default is 1e-5.
    high_frequency_mode : {"curvelet", "wavelet"}, optional
        High frequency mode. "curvelet" uses curvelets at all scales,
        "wavelet" creates a single ring-shaped window (bandpass filter only,
        no angular components) at the highest scale with decimation=1.
        Default is "curvelet".
    use_complex_transform : bool, optional
        If True, use complex transform which separates positive and negative
        frequency components into different bands. Each band is scaled by
        sqrt(0.5). If False (default), use real transform where each band
        captures both +/- frequencies combined.

    Attributes
    ----------
    shape : tuple[int, ...]
        Shape of the input data.
    high_frequency_mode : str
        High frequency mode.
    use_complex_transform : bool
        Whether complex transform is enabled.
    parameters : ParamUDCT
        Internal UDCT parameters.
    windows : UDCTWindows
        Curvelet windows in sparse format.
    decimation_ratios : list
        Decimation ratios for each scale/direction.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> # Create a 2D transform using num_scales (simplified interface)
    >>> transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
    >>> data = np.random.randn(64, 64)
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon, atol=1e-4)
    True
    >>> # Create using angular_wedges_config (advanced interface)
    >>> cfg = np.array([[3, 3], [6, 6]])
    >>> transform2 = UDCT(shape=(64, 64), angular_wedges_config=cfg)
    >>> coeffs2 = transform2.forward(data)
    >>> recon2 = transform2.backward(coeffs2)
    >>> np.allclose(data, recon2, atol=1e-4)
    True
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        angular_wedges_config: np.ndarray | None = None,
        num_scales: int | None = None,
        wedges_per_direction: int | None = None,
        window_overlap: float | None = None,
        radial_frequency_params: tuple[float, float, float, float] | None = None,
        window_threshold: float = 1e-5,
        high_frequency_mode: Literal["curvelet", "wavelet"] = "curvelet",
        use_complex_transform: bool = False,
    ) -> None:
        # Store basic attributes
        self.shape = shape
        self.high_frequency_mode = high_frequency_mode
        self.use_complex_transform = use_complex_transform

        # Calculate necessary parameters
        params_dict = self._initialize_parameters(
            shape=shape,
            angular_wedges_config=angular_wedges_config,
            num_scales=num_scales,
            wedges_per_direction=wedges_per_direction,
            window_overlap=window_overlap,
            radial_frequency_params=radial_frequency_params,
            window_threshold=window_threshold,
            high_frequency_mode=high_frequency_mode,
        )

        # Create ParamUDCT object
        self.parameters = ParamUDCT(
            shape=params_dict["internal_shape"],
            angular_wedges_config=params_dict["angular_wedges_config"],
            window_overlap=params_dict["window_overlap"],
            radial_frequency_params=params_dict["radial_frequency_params"],
            window_threshold=params_dict["window_threshold"],
        )

        # Calculate windows
        self.windows, self.decimation_ratios, self.indices = self._initialize_windows()

    @staticmethod
    def _compute_optimal_window_overlap(
        wedges_per_scale: npt.NDArray[np.int_],
    ) -> float:
        """
        Compute optimal window_overlap from Nguyen & Chauris (2010) constraint.

        The constraint (2^(s/N))(1+2a)(1+a) < N must hold for all scales.
        This method solves for the theoretical maximum at each scale,
        takes the minimum across all scales, and returns 10% of this value.

        The 10% factor was empirically determined to give better reconstruction
        quality than both the theoretical maximum and previous hardcoded defaults.

        Parameters
        ----------
        wedges_per_scale : ndarray
            Array of wedge counts per scale.

        Returns
        -------
        float
            Optimal window_overlap value (10% of theoretical maximum).

        References
        ----------
        .. [1] Nguyen, T. T., and H. Chauris, 2010, "Uniform Discrete Curvelet
           Transform": IEEE Transactions on Signal Processing, 58, 3618-3634.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> # For wedges_per_direction=3, num_scales=3
        >>> wedges_per_scale = np.array([3, 6])
        >>> alpha = UDCT._compute_optimal_window_overlap(wedges_per_scale)
        >>> 0.1 < alpha < 0.2  # Expected around 0.11
        True
        """
        min_overlap = float("inf")

        for scale_idx, num_wedges in enumerate(wedges_per_scale, start=1):
            k = 2 ** (scale_idx / num_wedges)
            # Solve: 2k*a^2 + 3k*a + (k - N) = 0
            # a = (-3 + sqrt(1 + 8*N/k)) / 4
            discriminant = 1 + 8 * num_wedges / k
            a_max = (-3 + np.sqrt(discriminant)) / 4
            min_overlap = min(min_overlap, a_max)

        # Use 10% of theoretical max for optimal reconstruction quality
        return 0.1 * min_overlap

    @staticmethod
    def _compute_from_angular_wedges_config(
        angular_wedges_config: np.ndarray,
        window_overlap: float | None,
    ) -> tuple[np.ndarray, float]:
        """
        Compute angular wedges configuration and window overlap from provided config.

        Parameters
        ----------
        angular_wedges_config : :obj:`np.ndarray <numpy.ndarray>`
            Configuration array specifying the number of angular wedges per scale
            and dimension. Shape is (num_scales, dimension).
        window_overlap : float | None
            Window overlap parameter. If None, automatically computed using
            the Nguyen & Chauris (2010) constraint formula.

        Returns
        -------
        tuple[:obj:`np.ndarray <numpy.ndarray>`, float]
            Tuple of (computed_angular_wedges_config, computed_window_overlap).
        """
        # Use provided angular_wedges_config directly
        computed_angular_wedges_config = angular_wedges_config

        # Validate that all wedge counts are divisible by 3
        # According to Nguyen & Chauris (2010), the decimation ratio formula
        # uses integer division by 3, so wedges must be divisible by 3
        invalid_wedges = computed_angular_wedges_config[
            computed_angular_wedges_config % 3 != 0
        ]
        if len(invalid_wedges) > 0:
            invalid_values = np.unique(invalid_wedges).tolist()
            msg = (
                f"All values in angular_wedges_config must be divisible by 3. "
                f"Found invalid values: {invalid_values}. "
                "According to the Nguyen & Chauris (2010) paper specification, "
                "the decimation ratio formula requires integer division by 3. "
                "Valid values are 3, 6, 9, 12, etc."
            )
            raise ValueError(msg)

        # Use provided window_overlap or compute optimal from first dimension
        if window_overlap is not None:
            computed_window_overlap = window_overlap
        else:
            # Compute from first dimension's wedges (all dimensions should be consistent)
            wedges_per_scale = angular_wedges_config[:, 0]
            computed_window_overlap = UDCT._compute_optimal_window_overlap(
                wedges_per_scale
            )

        return computed_angular_wedges_config, computed_window_overlap

    @staticmethod
    def _compute_from_num_scales(
        num_scales: int | None,
        wedges_per_direction: int | None,
        window_overlap: float | None,
        dimension: int,
    ) -> tuple[np.ndarray, float]:
        """
        Compute angular wedges configuration and window overlap from num_scales.

        Parameters
        ----------
        num_scales : int | None
            Total number of scales (including lowpass scale 0). Must be >= 2.
            If None, defaults to 3.
        wedges_per_direction : int | None
            Number of angular wedges per direction at the coarsest scale.
            Must be >= 3. If None, defaults to 3.
        window_overlap : float | None
            Window overlap parameter. If None, auto-selected based on
            wedges_per_direction.
        dimension : int
            Number of dimensions.

        Returns
        -------
        tuple[:obj:`np.ndarray <numpy.ndarray>`, float]
            Tuple of (computed_angular_wedges_config, computed_window_overlap).

        Raises
        ------
        ValueError
            If num_scales < 2 or wedges_per_direction < 3.
        """
        # Use num_scales/wedges_per_direction
        if num_scales is None:
            num_scales = 3
        if wedges_per_direction is None:
            wedges_per_direction = 3

        if num_scales < 2:
            msg = "num_scales must be >= 2"
            raise ValueError(msg)
        if wedges_per_direction < 3:
            msg = "wedges_per_direction must be >= 3"
            raise ValueError(msg)
        if wedges_per_direction % 3 != 0:
            msg = (
                f"wedges_per_direction={wedges_per_direction} must be divisible by 3. "
                "According to the Nguyen & Chauris (2010) paper specification, "
                "the decimation ratio formula requires integer division by 3. "
                "Valid values are 3, 6, 9, 12, etc."
            )
            raise ValueError(msg)

        # Convert to angular_wedges_config
        wedges_per_scale: npt.NDArray[np.int_] = (
            wedges_per_direction * 2 ** np.arange(num_scales - 1)
        ).astype(int)
        computed_angular_wedges_config = np.tile(wedges_per_scale[:, None], dimension)

        # Auto-select window_overlap if not provided
        if window_overlap is None:
            computed_window_overlap = UDCT._compute_optimal_window_overlap(
                wedges_per_scale
            )
        else:
            computed_window_overlap = window_overlap

        # Validate window_overlap
        for scale_idx, num_wedges in enumerate(wedges_per_scale, start=1):
            const = (
                2 ** (scale_idx / num_wedges)
                * (1 + 2 * computed_window_overlap)
                * (1 + computed_window_overlap)
            )
            if const >= num_wedges:
                msg = (
                    f"window_overlap={computed_window_overlap:.3f} does not respect the relationship "
                    f"(2^{scale_idx}/{num_wedges})(1+2a)(1+a) = {const:.3f} < {num_wedges} for scale {scale_idx + 1}"
                )
                logging.warning(msg)

        return computed_angular_wedges_config, computed_window_overlap

    def _initialize_parameters(
        self,
        shape: tuple[int, ...],
        angular_wedges_config: np.ndarray | None,
        num_scales: int | None,
        wedges_per_direction: int | None,
        window_overlap: float | None,
        radial_frequency_params: tuple[float, float, float, float] | None,
        window_threshold: float,
        high_frequency_mode: Literal["curvelet", "wavelet"],
    ) -> dict[str, Any]:
        """
        Calculate all necessary parameters for UDCT initialization.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the input data.
        angular_wedges_config : :obj:`np.ndarray <numpy.ndarray>` | None
            Configuration array, or None to use num_scales/wedges_per_direction.
        num_scales : int | None
            Number of scales (used when angular_wedges_config is None).
        wedges_per_direction : int | None
            Wedges per direction (used when angular_wedges_config is None).
        window_overlap : float | None
            Window overlap parameter.
        radial_frequency_params : tuple[float, float, float, float] | None
            Radial frequency parameters.
        window_threshold : float
            Window threshold.
        high_frequency_mode : str
            High frequency mode.

        Returns
        -------
        dict
            Dictionary containing all calculated parameters.
        """
        dimension = len(shape)

        # Determine which initialization style to use
        if angular_wedges_config is not None:
            if num_scales is not None or wedges_per_direction is not None:
                msg = "Cannot specify both angular_wedges_config and num_scales/wedges_per_direction"
                raise ValueError(msg)
            computed_angular_wedges_config, computed_window_overlap = (
                self._compute_from_angular_wedges_config(
                    angular_wedges_config, window_overlap
                )
            )
        else:
            computed_angular_wedges_config, computed_window_overlap = (
                self._compute_from_num_scales(
                    num_scales, wedges_per_direction, window_overlap, dimension
                )
            )

        # Compute num_scales from computed_angular_wedges_config for validation
        computed_num_scales = 1 + len(computed_angular_wedges_config)

        # Validate scale requirements
        if computed_num_scales < 2:
            msg = "requires at least 2 scales total (num_scales >= 2)"
            raise ValueError(msg)

        # Calculate internal shape
        internal_shape = shape

        # Set default radial_frequency_params if not provided
        if radial_frequency_params is None:
            computed_radial_frequency_params: tuple[float, float, float, float] = tuple(
                np.array([1.0, 2.0, 2.0, 4.0]) * np.pi / 3
            )
        else:
            computed_radial_frequency_params = radial_frequency_params

        return {
            "dimension": dimension,
            "internal_shape": internal_shape,
            "angular_wedges_config": computed_angular_wedges_config,
            "window_overlap": computed_window_overlap,
            "radial_frequency_params": computed_radial_frequency_params,
            "window_threshold": window_threshold,
        }

    def _initialize_windows(
        self,
    ) -> tuple[
        UDCTWindows, list[npt.NDArray[np.int_]], dict[int, dict[int, np.ndarray]]
    ]:
        """
        Calculate curvelet windows, decimation ratios, and indices.

        Returns
        -------
        tuple
            (windows, decimation_ratios, indices)
        """
        window_computer = UDCTWindow(self.parameters, self.high_frequency_mode)
        return window_computer.compute()

    def vect(
        self, coefficients: list[list[list[npt.NDArray[C]]]]
    ) -> npt.NDArray[np.complexfloating]:
        """
        Convert structured coefficients to vector representation.

        Parameters
        ----------
        coefficients : list[list[list[``npt.NDArray[C]``]]]
            Structured curvelet coefficients.

        Returns
        -------
        :obj:`np.ndarray <numpy.ndarray>`
            Flattened vector of all coefficients.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> vec = transform.vect(coeffs)
        >>> vec.shape
        (4096,)
        """
        coefficients_vec = []
        for scale_coeffs in coefficients:
            for direction_coeffs in scale_coeffs:
                for wedge_coeffs in direction_coeffs:
                    coefficients_vec.append(wedge_coeffs.ravel())
        return np.concatenate(coefficients_vec)

    def struct(
        self, coefficients_vec: npt.NDArray[np.complexfloating]
    ) -> list[list[list[npt.NDArray[C]]]]:
        """
        Convert vector representation to structured coefficients.

        Parameters
        ----------
        coefficients_vec : :obj:`np.ndarray <numpy.ndarray>`
            Flattened vector of coefficients.

        Returns
        -------
        list[list[list[``npt.NDArray[C]``]]]
            Structured curvelet coefficients.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> vec = transform.vect(coeffs)
        >>> coeffs_recon = transform.struct(vec)
        >>> len(coeffs_recon) == len(coeffs)
        True
        """
        begin_idx = 0
        coefficients: list[list[list[npt.NDArray[C]]]] = []
        internal_shape = np.array(self.parameters.shape)
        for scale_idx, decimation_ratios_scale in enumerate(self.decimation_ratios):
            coefficients.append([])
            # In complex transform mode, we have 2*dim directions per scale (for scale > 0)
            # but decimation_ratios_scale only has dim rows, so we need to handle this
            if self.use_complex_transform and scale_idx > 0:
                num_directions = 2 * self.parameters.ndim
            else:
                num_directions = len(decimation_ratios_scale)

            for direction_idx in range(num_directions):
                coefficients[scale_idx].append([])
                # In complex transform mode, directions >= dim reuse windows and decimation ratios
                # from directions < dim. Negative frequency directions (dim..2*dim-1) use the same
                # windows and decimation ratios as positive frequency directions (0..dim-1)
                # For "wavelet" mode at highest scale, windows only has 1 direction, so use index 0
                if (
                    self.use_complex_transform
                    and scale_idx > 0
                    and direction_idx >= self.parameters.ndim
                ):
                    window_direction_idx = direction_idx % self.parameters.ndim
                    # Clamp to available windows (for "wavelet" mode at highest scale)
                    window_direction_idx = min(
                        window_direction_idx, len(self.windows[scale_idx]) - 1
                    )
                    decimation_ratio_dir = decimation_ratios_scale[
                        min(window_direction_idx, len(decimation_ratios_scale) - 1), :
                    ]
                else:
                    window_direction_idx = min(
                        direction_idx, len(self.windows[scale_idx]) - 1
                    )
                    decimation_ratio_dir = decimation_ratios_scale[
                        min(direction_idx, len(decimation_ratios_scale) - 1), :
                    ]

                for _ in self.windows[scale_idx][window_direction_idx]:
                    shape_decimated = internal_shape // decimation_ratio_dir
                    end_idx = begin_idx + prod(shape_decimated)
                    wedge = coefficients_vec[begin_idx:end_idx].reshape(shape_decimated)
                    coefficients[scale_idx][direction_idx].append(wedge)
                    begin_idx = end_idx
        return coefficients

    def from_sparse(
        self, arr_sparse: tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """
        Convert sparse window format to dense array.

        Parameters
        ----------
        arr_sparse : tuple[:obj:`NDArray <numpy.typing.NDArray>` [:obj:`intp <numpy.intp>`], :obj:`NDArray <numpy.typing.NDArray>` [:obj:`floating <numpy.floating>`]]
            Sparse window format as a tuple of (indices, values).

        Returns
        -------
        :obj:`NDArray <numpy.typing.NDArray>` [:obj:`floating <numpy.floating>`]
            Dense array with the same shape as the transform input.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
        >>> # Get a sparse window
        >>> sparse_window = transform.windows[0][0][0]
        >>> # Convert to dense
        >>> dense_window = transform.from_sparse(sparse_window)
        >>> dense_window.shape
        (64, 64)
        """
        idx, val = from_sparse_new(arr_sparse)
        arr_full = np.zeros(self.parameters.shape, dtype=val.dtype)
        arr_full.flat[idx] += val
        return arr_full

    def forward(
        self, image: npt.NDArray[F] | npt.NDArray[C]
    ) -> list[list[list[npt.NDArray[np.complexfloating]]]]:
        """
        Apply forward curvelet transform.

        Parameters
        ----------
        image : ``npt.NDArray[F]`` | ``npt.NDArray[C]``
            Input data with shape matching self.shape. Can be real-valued
            (``npt.NDArray[F]``) or complex-valued (``npt.NDArray[C]``).

        Returns
        -------
        list[list[list[``npt.NDArray[C]``]]]
            Curvelet coefficients as nested list structure.
            When use_complex_transform=True, directions are doubled (first dim directions
            for positive frequencies, next dim for negative).
            Coefficients have complex dtype matching the input:
            - np.float32 input -> np.complex64 coefficients
            - np.float64 input -> np.complex128 coefficients
            - np.complex64 input -> np.complex64 coefficients
            - np.complex128 input -> np.complex128 coefficients

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> len(coeffs)  # Number of scales
        4
        """
        np.testing.assert_equal(self.shape, image.shape)

        # Apply curvelet transform
        # Runtime checks determine the appropriate transform path
        if self.use_complex_transform:
            if np.iscomplexobj(image):
                result = _apply_forward_transform_complex(
                    image,
                    self.parameters,
                    self.windows,
                    self.decimation_ratios,
                )
            else:
                # Convert real to complex for complex transform
                # Preserve input dtype: float32 -> complex64, float64 -> complex128
                complex_dtype = _to_complex_dtype(image.dtype)
                result = _apply_forward_transform_complex(
                    image.astype(complex_dtype),
                    self.parameters,
                    self.windows,
                    self.decimation_ratios,
                )
        elif not np.iscomplexobj(image):
            result = _apply_forward_transform_real(
                image,
                self.parameters,
                self.windows,
                self.decimation_ratios,
            )
        else:
            # Convert complex to real for real transform
            result = _apply_forward_transform_real(
                image.real,
                self.parameters,
                self.windows,
                self.decimation_ratios,
            )

        return result

    def backward(self, coefficients: list[list[list[npt.NDArray[C]]]]) -> np.ndarray:
        """
        Apply backward curvelet transform (reconstruction).

        Parameters
        ----------
        coefficients : list[list[list[``npt.NDArray[C]``]]]
            Curvelet coefficients from forward transform.

        Returns
        -------
        :obj:`np.ndarray <numpy.ndarray>`
            Reconstructed data with shape matching self.shape.
            Returns complex array when use_complex_transform=True (required for complex inputs),
            real array when use_complex_transform=False.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> recon = transform.backward(coeffs)
        >>> np.allclose(data, recon, atol=1e-4)
        True
        """
        return _apply_backward_transform(
            coefficients,
            self.parameters,
            self.windows,
            self.decimation_ratios,
            use_complex_transform=self.use_complex_transform,  # type: ignore[call-overload]
        )

    def forward_monogenic(self, image: npt.NDArray[F]) -> MUDCTCoefficients:
        """
        Apply forward monogenic curvelet transform.

        This method applies the monogenic extension to the curvelet transform,
        producing quaternion-like coefficients with three components per band:
        scalar (same as standard UDCT), Riesz_1, and Riesz_2. This enables
        meaningful amplitude/phase decomposition over all scales.

        .. note::
            **2D Limitation**: The monogenic curvelet transform is mathematically
            defined only for 2D signals according to Storath 2010. While this
            implementation accepts arbitrary dimensions, only the first two Riesz
            components (R_1 and R_2) are computed, which is correct only for 2D
            inputs. For 3D or higher dimensions, the transform will only use R_1
            and R_2, which may not provide a complete monogenic representation.

        Parameters
        ----------
        image : npt.NDArray[F]
            Input image or volume. Must be real-valued (floating point dtype).
            Complex inputs are not supported as the monogenic transform is
            defined only for real-valued functions.

        Returns
        -------
        MUDCTCoefficients
            Quaternion-like coefficients as nested list structure with tuples
            of 3 arrays: (scalar_component, riesz_1_component, riesz_2_component).
            Structure: coefficients[scale][direction][wedge] = (coeff_0, coeff_1, coeff_2)
            where:
            - coeff_0 (scalar): Complex array (matches UDCT behavior in real transform mode)
            - coeff_1, coeff_2 (Riesz): Real arrays

        Raises
        ------
        ValueError
            If input is complex-valued. Use forward() for complex inputs.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward_monogenic(data)
        >>> len(coeffs)  # Number of scales
        4
        >>> isinstance(coeffs[0][0][0], tuple)  # Each coefficient is a tuple
        True
        >>> len(coeffs[0][0][0])  # Tuple has 3 components
        3
        >>> # Compute amplitude
        >>> scalar, riesz1, riesz2 = coeffs[1][0][0]
        >>> amplitude = np.sqrt(scalar**2 + riesz1**2 + riesz2**2)
        """
        np.testing.assert_equal(self.shape, image.shape)

        if np.iscomplexobj(image):
            msg = (
                "Monogenic transform requires real-valued input. "
                "Got complex array. Use forward() for complex inputs."
            )
            raise ValueError(msg)

        return _apply_forward_transform_monogenic(
            image,
            self.parameters,
            self.windows,
            self.decimation_ratios,
        )

    def backward_monogenic(
        self, coefficients: MUDCTCoefficients
    ) -> tuple[npt.NDArray[F], npt.NDArray[F], npt.NDArray[F]]:
        """
        Apply backward monogenic curvelet transform (reconstruction).

        This method implements the full reproducing formula from Storath 2010:
        M_f(x) = ∫ M_f(a,b,θ) · M_β_{abθ}(x) db dθ da/a³

        The quaternion multiplication expands to three components:
        - Scalar: c₀·β - c₁·R₁β - c₂·R₂β → reconstructs f
        - Riesz_1: -c₀·R₁β + c₁·β → reconstructs -R₁f
        - Riesz_2: -c₀·R₂β + c₂·β → reconstructs -R₂f

        .. note::
            **2D Limitation**: The monogenic curvelet transform is mathematically
            defined only for 2D signals according to Storath 2010. While this
            implementation accepts arbitrary dimensions, only the first two Riesz
            components (R_1 and R_2) are used in the reconstruction, which is
            correct only for 2D inputs.

        Parameters
        ----------
        coefficients : MUDCTCoefficients
            Monogenic curvelet coefficients from forward_monogenic().
            Structure: coefficients[scale][direction][wedge] = (coeff_0, coeff_1, coeff_2)

        Returns
        -------
        tuple[npt.NDArray[F], npt.NDArray[F], npt.NDArray[F]]
            Tuple of three real-valued arrays with shape matching self.shape:
            - scalar: Original input f (reconstructed using full reproducing formula)
            - riesz_1: First Riesz component -R_1 f
            - riesz_2: Second Riesz component -R_2 f

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward_monogenic(data)
        >>> scalar, riesz1, riesz2 = transform.backward_monogenic(coeffs)
        >>> np.allclose(data, scalar, atol=1e-4)
        True
        >>> # Verify Riesz components match direct Riesz transform
        >>> from curvelets.numpy._riesz import riesz_filters
        >>> filters = riesz_filters((64, 64))
        >>> data_fft = np.fft.fftn(data)
        >>> riesz1_direct = np.fft.ifftn(data_fft * filters[0]).real
        >>> np.allclose(-riesz1_direct, riesz1, atol=1e-4)
        True
        """
        return _apply_backward_transform_monogenic(
            coefficients,
            self.parameters,
            self.windows,
            self.decimation_ratios,
        )

    def monogenic(
        self, image: npt.NDArray[F]
    ) -> tuple[npt.NDArray[F], npt.NDArray[F], npt.NDArray[F]]:
        """
        Compute monogenic signal directly without curvelet transform.

        This method computes the monogenic signal Mf = f + i(-R₁f) + j(-R₂f)
        directly from the input function without performing the curvelet transform.
        The monogenic signal provides a quaternion-like representation that enables
        meaningful amplitude/phase decomposition.

        .. note::
            **2D Limitation**: The monogenic signal is mathematically defined only
            for 2D signals according to Storath 2010. While this implementation
            accepts arbitrary dimensions, only the first two Riesz components (R_1
            and R_2) are computed, which is correct only for 2D inputs. For 3D or
            higher dimensions, the transform will only use R_1 and R_2, which may
            not provide a complete monogenic representation.

        Parameters
        ----------
        image : npt.NDArray[F]
            Input image or volume. Must be real-valued (floating point dtype).
            Must have shape matching self.shape. Complex inputs are not supported
            as the monogenic transform is defined only for real-valued functions.

        Returns
        -------
        tuple[npt.NDArray[F], npt.NDArray[F], npt.NDArray[F]]
            Tuple of three real-valued arrays with shape matching self.shape:
            - scalar: Original input f (unchanged)
            - riesz_1: First Riesz component -R_1 f
            - riesz_2: Second Riesz component -R_2 f

        Raises
        ------
        ValueError
            If input is complex-valued. Use forward() for complex inputs.
        AssertionError
            If input shape does not match self.shape.

        Notes
        -----
        This method computes the monogenic signal directly using Riesz transforms
        without performing the curvelet decomposition. It is mathematically equivalent
        to backward_monogenic(forward_monogenic(f)) but computationally simpler.

        The monogenic signal is defined as:
        Mf = f + i(-R₁f) + j(-R₂f)

        where R₁ and R₂ are the Riesz transforms. Since Python doesn't have native
        quaternion types, this method returns the three components as a tuple.

        References
        ----------
        .. [1] Storath, M., 2010, *The monogenic curvelet transform*: 2010 IEEE
           International Conference on Image Processing.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> data = np.random.randn(64, 64)
        >>> scalar, riesz1, riesz2 = transform.monogenic(data)
        >>> scalar.shape
        (64, 64)
        >>> np.allclose(data, scalar)
        True
        >>> # Verify it matches backward_monogenic(forward_monogenic(f))
        >>> coeffs = transform.forward_monogenic(data)
        >>> scalar2, riesz1_2, riesz2_2 = transform.backward_monogenic(coeffs)
        >>> np.allclose(scalar, scalar2, atol=1e-4)
        True
        >>> np.allclose(riesz1, riesz1_2, atol=1e-4)
        True
        >>> np.allclose(riesz2, riesz2_2, atol=1e-4)
        True
        """
        np.testing.assert_equal(self.shape, image.shape)

        if np.iscomplexobj(image):
            msg = (
                "Monogenic transform requires real-valued input. "
                "Got complex array. Use forward() for complex inputs."
            )
            raise ValueError(msg)

        # Compute Riesz filters
        riesz_filters_list = riesz_filters(self.shape)

        # Compute FFT of input
        image_frequency = np.fft.fftn(image)

        # Compute Riesz components: -R₁f and -R₂f
        riesz1 = -np.fft.ifftn(image_frequency * riesz_filters_list[0]).real
        riesz2 = -np.fft.ifftn(image_frequency * riesz_filters_list[1]).real

        # Preserve input dtype
        real_dtype = image.dtype
        riesz1 = riesz1.astype(real_dtype)
        riesz2 = riesz2.astype(real_dtype)

        return (image.copy(), riesz1, riesz2)
