# pylint: disable=too-many-lines,duplicate-code
# Duplicate code with torch implementation is expected
from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from math import ceil
from typing import Union

import numpy as np
import numpy.typing as npt

from ._typing import F, IntegerNDArray, IntpNDArray, UDCTWindows
from ._utils import ParamUDCT, circular_shift, meyer_window


class UDCTWindow:
    """
    Window computation for Uniform Discrete Curvelet Transform.

    This class encapsulates all window computation functionality for the UDCT,
    including bandpass filter creation, angle function computation, window
    normalization, and sparse format conversion. The class is initialized with
    UDCT parameters and provides a compute() method to generate curvelet windows.

    Parameters
    ----------
    parameters : ParamUDCT
        UDCT parameters containing transform configuration.
    high_frequency_mode : str, optional
        High frequency mode. "curvelet" uses curvelets at all scales,
        "wavelet" creates a single ring-shaped window (bandpass filter only,
        no angular components) at the highest scale with decimation=1.
        Default is "curvelet".

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._utils import ParamUDCT
    >>> from curvelets.numpy._udct_windows import UDCTWindow
    >>>
    >>> # Create parameters for 2D transform with 4 scales total (1 lowpass + 3 high-frequency)
    >>> params = ParamUDCT(
    ...     shape=(64, 64),
    ...     angular_wedges_config=np.array([[3], [6], [12]]),
    ...     window_overlap=0.15,
    ...     window_threshold=1e-5,
    ...     radial_frequency_params=(np.pi/3, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3)
    ... )
    >>>
    >>> # Create window computer and compute windows
    >>> window_computer = UDCTWindow(params)
    >>> windows, decimation_ratios, indices = window_computer.compute()
    >>>
    >>> # Check structure
    >>> len(windows)  # Number of scales
    4
    >>> len(windows[0][0])  # Low-frequency band has 1 window
    1
    >>> len(windows[1][0])  # First high-frequency scale has multiple wedges
    3
    """

    def __init__(
        self, parameters: ParamUDCT, high_frequency_mode: str = "curvelet"
    ) -> None:
        """
        Initialize UDCTWindow with parameters from ParamUDCT.

        Parameters
        ----------
        parameters : ParamUDCT
            UDCT parameters containing transform configuration.
        high_frequency_mode : str, optional
            High frequency mode. "curvelet" uses curvelets at all scales,
            "wavelet" creates a single ring-shaped window (bandpass filter only) at the highest scale
            with decimation=1. Default is "curvelet".

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy._utils import ParamUDCT
        >>> from curvelets.numpy._udct_windows import UDCTWindow
        >>> params = ParamUDCT(
        ...     ndim=2,
        ...     shape=(64, 64),
        ...     angular_wedges_config=np.array([[3], [6], [12]]),
        ...     window_overlap=0.15,
        ...     radial_frequency_params=(np.pi/3, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3),
        ...     window_threshold=1e-5
        ... )
        >>> window_computer = UDCTWindow(params)
        >>> window_computer.num_scales
        3
        >>> window_computer.shape
        (64, 64)
        """
        self.num_scales = parameters.num_scales
        self.shape = parameters.shape
        self.dimension = parameters.ndim
        self.radial_frequency_params = parameters.radial_frequency_params
        self.angular_wedges_config = parameters.angular_wedges_config
        self.window_overlap = parameters.window_overlap
        self.window_threshold = parameters.window_threshold
        self.high_frequency_mode = high_frequency_mode

    @staticmethod
    def _compute_angle_component(
        x_primary: npt.NDArray[F], x_secondary: npt.NDArray[F]
    ) -> npt.NDArray[F]:
        """
        Compute one angle component from meshgrid coordinates.

        Parameters
        ----------
        x_primary : npt.NDArray[F]
            Primary coordinate grid (used for conditions).
        x_secondary : npt.NDArray[F]
            Secondary coordinate grid.

        Returns
        -------
        npt.NDArray[F]
            Angle component array.
        """
        # Compute angle component using piecewise function:
        # When primary coordinate dominates (|x_secondary| <= |x_primary|), use -x_secondary/x_primary
        primary_ratio = np.zeros_like(x_primary)
        mask = (x_primary != 0) & (np.abs(x_secondary) <= np.abs(x_primary))
        primary_ratio[mask] = -x_secondary[mask] / x_primary[mask]

        # When secondary coordinate dominates (|x_primary| < |x_secondary|), use x_primary/x_secondary
        secondary_ratio = np.zeros_like(x_primary)
        mask = (x_secondary != 0) & (np.abs(x_primary) < np.abs(x_secondary))
        secondary_ratio[mask] = x_primary[mask] / x_secondary[mask]

        # Wrap secondary_ratio to the correct range by adding/subtracting 2
        wrapped_ratio = secondary_ratio.copy()
        wrapped_ratio[secondary_ratio < 0] = secondary_ratio[secondary_ratio < 0] + 2
        wrapped_ratio[secondary_ratio > 0] = secondary_ratio[secondary_ratio > 0] - 2

        # Combine ratios and set special case for x_primary >= 0
        # Create result array with same dtype as input to preserve TypeVar F
        angle_component = np.zeros_like(x_primary)
        angle_component[:] = primary_ratio + wrapped_ratio
        angle_component[x_primary >= 0] = -2
        return angle_component

    @staticmethod
    def _create_angle_grids_from_frequency_grids(
        frequency_grid_1: npt.NDArray[F], frequency_grid_2: npt.NDArray[F]
    ) -> tuple[npt.NDArray[F], npt.NDArray[F]]:
        """
        Adapt frequency grids for angle computation.

        Parameters
        ----------
        frequency_grid_1 : npt.NDArray[F]
            First frequency grid.
        frequency_grid_2 : npt.NDArray[F]
            Second frequency grid.

        Returns
        -------
        tuple[npt.NDArray[F], npt.NDArray[F]]
            Adapted grid arrays angle_grid_2 and angle_grid_1.
        """
        meshgrid_dim1, meshgrid_dim2 = np.meshgrid(frequency_grid_2, frequency_grid_1)

        # Compute angle_grid_1 using meshgrid_dim1 as primary, meshgrid_dim2 as secondary
        angle_grid_1 = UDCTWindow._compute_angle_component(meshgrid_dim1, meshgrid_dim2)

        # Compute angle_grid_2 using meshgrid_dim2 as primary, meshgrid_dim1 as secondary (swapped)
        angle_grid_2 = UDCTWindow._compute_angle_component(meshgrid_dim2, meshgrid_dim1)

        return angle_grid_2, angle_grid_1

    @staticmethod
    def _create_angle_functions(
        angle_grid: npt.NDArray[F],
        direction: int,
        num_angular_wedges: int,
        window_overlap: float,
    ) -> npt.NDArray[F]:
        """
        Create angle functions using Meyer windows.

        Parameters
        ----------
        angle_grid : npt.NDArray[F]
            Angle grid.
        direction : int
            Direction index (1 or 2).
        num_angular_wedges : int
            Number of angular wedges.
        window_overlap : float
            Window overlap parameter.

        Returns
        -------
        npt.NDArray[F]
            Array of angle functions.
        """
        # Compute angular window spacing and boundaries
        angular_spacing = 2 / num_angular_wedges
        angular_boundaries = angular_spacing * np.array(
            [-window_overlap, window_overlap, 1 - window_overlap, 1 + window_overlap]
        )

        # Generate angle functions using Meyer windows
        # Note: Both direction 1 and 2 use the same computation because the
        # angle function is symmetric with respect to direction. The direction
        # parameter is kept for API consistency and potential future extensions.
        angle_functions_list: list[npt.NDArray[F]] = []
        if direction in (1, 2):
            for wedge_index in range(1, ceil(num_angular_wedges / 2) + 1):
                ang2 = -1 + (wedge_index - 1) * angular_spacing + angular_boundaries
                window_values = meyer_window(angle_grid, *ang2)
                angle_functions_list.append(window_values[None, :])
        else:
            error_msg = f"Unrecognized direction: {direction}. Must be 1 or 2."
            raise ValueError(error_msg)
        return np.concatenate(angle_functions_list, axis=0)

    @staticmethod
    def _compute_angle_kronecker_product(
        angle_function_1d: npt.NDArray[F],
        dimension_permutation: IntegerNDArray,
        shape: tuple[int, ...],
        dimension: int,
    ) -> npt.NDArray[F]:
        """
        Compute Kronecker product for angle functions.

        Parameters
        ----------
        angle_function_1d : npt.NDArray[F]
            Angle function array.
        dimension_permutation : IntegerNDArray
            Dimension permutation indices.
        shape : tuple[int, ...]
            Shape of the input data.
        dimension : int
            Dimensionality of the transform.

        Returns
        -------
        npt.NDArray[F]
            Kronecker product result with shape matching input shape.
            The dtype matches the input angle_function_1d dtype (preserves TypeVar F).
        """
        # Pre-compute dimension sizes for Kronecker product
        kronecker_dimension_sizes: npt.NDArray[np.int_] = np.array(
            [
                np.prod(shape[: dimension_permutation[0] - 1]),
                np.prod(shape[dimension_permutation[0] : dimension_permutation[1] - 1]),
                np.prod(shape[dimension_permutation[1] : dimension]),
            ],
            dtype=int,
        )

        # Expand 1D angle function to N-D using multi-step Kronecker products:
        # This matches the original angle_kron implementation which uses travel() = T.ravel()
        # Step 1: Expand along dimension 1 (kronecker_dimension_sizes[1])
        kron_step1 = np.kron(
            np.ones((kronecker_dimension_sizes[1], 1), dtype=int), angle_function_1d
        )
        # Use travel() = T.ravel() instead of ravel() to match original implementation
        kron_step1_travel = kron_step1.T.ravel()
        kron_step2 = np.kron(
            np.ones((kronecker_dimension_sizes[2], 1), dtype=int), kron_step1_travel
        ).ravel()
        # Use travel() = T.ravel() for the final step as well
        kron_step3 = (
            np.kron(kron_step2, np.ones((kronecker_dimension_sizes[0], 1), dtype=int))
        ).T.ravel()
        # Match original: reshape with reversed size and transpose
        return kron_step3.reshape(*shape[::-1]).T.astype(angle_function_1d.dtype)

    @staticmethod
    def _flip_with_fft_shift(input_array: npt.NDArray[F], axis: int) -> npt.NDArray[F]:
        """
        Flip array along specified axis with frequency domain shift.

        Parameters
        ----------
        input_array : npt.NDArray[F]
            Input array.
        axis : int
            Axis along which to flip.

        Returns
        -------
        npt.NDArray[F]
            Flipped and shifted array.
        """
        shift_vector = [0] * input_array.ndim
        shift_vector[axis] = 1
        flipped_array = np.flip(input_array, axis)
        return circular_shift(flipped_array, tuple(shift_vector))

    @staticmethod
    def _to_sparse(
        arr: npt.NDArray[F], threshold: float
    ) -> tuple[IntpNDArray, npt.NDArray[F]]:
        """
        Convert array to sparse format.

        Parameters
        ----------
        arr : npt.NDArray[F]
            Input array.
        threshold : float
            Threshold for sparse storage (values above threshold are kept).

        Returns
        -------
        tuple[IntpNDArray, npt.NDArray[F]]
            Tuple of (indices, values) where indices are positions and values
            are the array values at those positions.
        """
        arr_flat = arr.ravel()
        indices = np.argwhere(arr_flat > threshold)
        return (indices, arr_flat[indices])

    @staticmethod
    def _nchoosek(n: Union[Iterable[int], IntegerNDArray], k: int) -> IntegerNDArray:
        """
        Generate all combinations of k elements from n.

        Parameters
        ----------
        n : Iterable[int] | IntegerNDArray
            Iterable containing elements to choose from (list, array, range, etc.).
        k : int
            Number of elements to choose in each combination.

        Returns
        -------
        npt.NDArray[np.int_]
            Array of shape (C(n,k), k) containing all combinations.
        """
        return np.asarray(list(combinations(n, k)), dtype=int)

    @staticmethod
    def _create_bandpass_windows(
        num_scales: int,
        shape: tuple[int, ...],
        radial_frequency_params: tuple[float, float, float, float],
    ) -> tuple[dict[int, npt.NDArray[np.float64]], dict[int, npt.NDArray[np.float64]]]:
        """
            Create bandpass windows using Meyer wavelets for radial frequency decomposition.

        This function generates frequency-domain bandpass filters by constructing
        Meyer wavelet windows for each dimension and scale, then combining them
        using Kronecker products to create multi-dimensional bandpass filters.

        Parameters
        ----------
        num_scales : int
            Total number of scales (including lowpass scale) for the transform.
        shape : tuple[int, ...]
            Shape of the input data, determines frequency grid size.
        radial_frequency_params : tuple[float, float, float, float]
            Four parameters defining radial frequency bands:
            - params[0], params[1]: Lower frequency band boundaries
            - params[2], params[3]: Upper frequency band boundaries

        Returns
        -------
        frequency_grid : dict[int, npt.NDArray[F]]
            Dictionary mapping dimension index to frequency grid array.
            Each grid spans [-1.5*pi, 0.5*pi) with size matching shape[dimension].
        bandpass_windows : dict[int, npt.NDArray[F]]
            Dictionary mapping scale index to bandpass window array.
            Scale 0 is low-frequency, scales 1..(num_scales-1) are high-frequency bands.
            Each window has shape matching input `shape`.
        """
        dimension = len(shape)
        frequency_grid: dict[int, npt.NDArray[np.float64]] = {}
        meyer_windows: dict[tuple[int, int], npt.NDArray[np.float64]] = {}
        for dimension_idx in range(dimension):
            frequency_grid[dimension_idx] = np.linspace(
                -1.5 * np.pi, 0.5 * np.pi, shape[dimension_idx], endpoint=False
            )

            meyer_params = np.array([-2, -1, *radial_frequency_params[:2]])
            abs_frequency_grid = np.abs(frequency_grid[dimension_idx])
            meyer_windows[(num_scales - 1, dimension_idx)] = meyer_window(
                abs_frequency_grid, *meyer_params
            )
            if num_scales == 2:
                meyer_windows[(num_scales - 1, dimension_idx)] += meyer_window(
                    np.abs(frequency_grid[dimension_idx] + 2 * np.pi), *meyer_params
                )
            meyer_params[2:] = radial_frequency_params[2:]
            meyer_windows[(num_scales, dimension_idx)] = meyer_window(
                abs_frequency_grid, *meyer_params
            )

            for scale_idx in range(num_scales - 2, 0, -1):
                meyer_params[2:] = radial_frequency_params[:2]
                meyer_params[2:] /= 2 ** (num_scales - 1 - scale_idx)
                meyer_windows[(scale_idx, dimension_idx)] = meyer_window(
                    abs_frequency_grid, *meyer_params
                )

        bandpass_windows: dict[int, npt.NDArray[np.float64]] = {}
        for scale_idx in range(num_scales - 1, 0, -1):
            low_freq = np.array([1.0], dtype=np.float64)
            high_freq = np.array([1.0], dtype=np.float64)
            for dimension_idx in range(dimension - 1, -1, -1):
                low_freq = np.kron(
                    meyer_windows[(scale_idx, dimension_idx)], low_freq
                ).astype(np.float64)
                high_freq = np.kron(
                    meyer_windows[(scale_idx + 1, dimension_idx)], high_freq
                ).astype(np.float64)
            low_freq_nd = low_freq.reshape(*shape)
            high_freq_nd = high_freq.reshape(*shape)
            bandpass_nd = high_freq_nd - low_freq_nd
            bandpass_nd[bandpass_nd < 0] = 0
            bandpass_windows[scale_idx] = bandpass_nd
        bandpass_windows[0] = low_freq_nd
        return frequency_grid, bandpass_windows

    @staticmethod
    def _create_direction_mappings(
        dimension: int, num_scales: int
    ) -> list[IntegerNDArray]:
        """
            Create direction mappings for each resolution scale.

        For each resolution, creates a mapping indicating which dimensions
        need angle function calculations on each hyperpyramid. This is used
        to determine which dimensions are used for angular decomposition
        at each scale.

        Parameters
        ----------
        dimension : int
            Dimensionality of the transform.
        num_scales : int
            Total number of scales (including lowpass scale).

        Returns
        -------
        list[IntegerNDArray]
            List of arrays, one per resolution. Each array has shape (dimension, dimension-1)
            and contains indices of dimensions used for angle calculations on each hyperpyramid.
        """
        return [
            np.c_[
                [
                    np.r_[
                        np.arange(dimension_idx),
                        np.arange(dimension_idx + 1, dimension),
                    ]
                    for dimension_idx in range(dimension)
                ]
            ]
            for scale_idx in range(num_scales - 1)
        ]

    @staticmethod
    def _create_angle_info(
        frequency_grid: dict[int, npt.NDArray[F]],
        dimension: int,
        num_scales: int,
        angular_wedges_config: IntegerNDArray,
        window_overlap: float,
        high_frequency_mode: str = "curvelet",
    ) -> tuple[
        dict[int, dict[tuple[int, int], npt.NDArray[F]]],
        dict[int, dict[tuple[int, int], IntegerNDArray]],
    ]:
        """
        Create angle functions and indices for window computation.

        Parameters
        ----------
        frequency_grid : dict[int, npt.NDArray[F]]
            Dictionary mapping dimension index to frequency grid.
        dimension : int
            Dimensionality of the transform.
        num_scales : int
            Total number of scales (including lowpass scale).
        angular_wedges_config : IntegerNDArray
            Configuration array specifying number of angular wedges per scale and dimension.
        window_overlap : float
            Window overlap parameter.
        high_frequency_mode : str, optional
            High frequency mode. When "wavelet", overlap is set to 0 for the highest scale.
            Default is "curvelet".

        Returns
        -------
        tuple[dict[int, dict[tuple[int, int], npt.NDArray[F]]], dict[int, dict[tuple[int, int], IntegerNDArray]]]
            Tuple of (angle_functions, angle_indices) dictionaries.
        """
        dimension_permutations = UDCTWindow._nchoosek(np.arange(dimension), 2)
        angle_grid: dict[tuple[int, int], npt.NDArray[F]] = {}
        for pair_index, dimension_pair in enumerate(dimension_permutations):
            angle_grids = UDCTWindow._create_angle_grids_from_frequency_grids(
                frequency_grid[dimension_pair[0]], frequency_grid[dimension_pair[1]]
            )
            angle_grid[(pair_index, 0)] = angle_grids[0]
            angle_grid[(pair_index, 1)] = angle_grids[1]

        angle_functions: dict[int, dict[tuple[int, int], npt.NDArray[F]]] = {}
        angle_indices: dict[int, dict[tuple[int, int], IntegerNDArray]] = {}
        for scale_idx in range(num_scales - 1):
            angle_functions[scale_idx] = {}
            angle_indices[scale_idx] = {}
            # Use overlap=0 for highest scale when mode="wavelet"
            # Note: scale_idx is 0-indexed for high-frequency scales (0 to num_scales-2)
            # In compute(), actual scale indices are 1 to num_scales-1, and they access
            # angle_functions[scale_idx - 1]. So the highest scale (num_scales - 1) maps
            # to angle_functions[num_scales - 2] here. Therefore, we check scale_idx == num_scales - 2
            # to identify the highest scale.
            is_highest_scale = scale_idx == num_scales - 2
            overlap_to_use = (
                0.0
                if (high_frequency_mode == "wavelet" and is_highest_scale)
                else window_overlap
            )
            for dimension_idx in range(dimension):
                angle_function_index = 0
                for hyperpyramid_idx in range(dimension_permutations.shape[0]):
                    for direction_idx in range(dimension_permutations.shape[1]):
                        if (
                            dimension_permutations[hyperpyramid_idx, direction_idx]
                            == dimension_idx
                        ):
                            angle_functions[scale_idx][
                                (dimension_idx, angle_function_index)
                            ] = UDCTWindow._create_angle_functions(
                                angle_grid[(hyperpyramid_idx, direction_idx)],
                                direction_idx + 1,
                                angular_wedges_config[
                                    scale_idx,
                                    dimension_permutations[
                                        hyperpyramid_idx, 1 - direction_idx
                                    ],
                                ],
                                overlap_to_use,
                            )
                            angle_indices[scale_idx][
                                (dimension_idx, angle_function_index)
                            ] = dimension_permutations[hyperpyramid_idx, :] + 1
                            angle_function_index += 1
        return angle_functions, angle_indices

    @staticmethod
    def _inplace_normalize_windows(
        windows: UDCTWindows,
        size: tuple[int, ...],
        dimension: int,
        num_scales: int,
    ) -> None:
        """
        Normalize windows in-place to ensure tight frame property.

        Parameters
        ----------
        windows : UDCTWindows
            Windows to normalize (modified in-place).
        size : tuple[int, ...]
            Size of the windows.
        dimension : int
            Dimensionality of the transform.
        num_scales : int
            Total number of scales (including lowpass scale).
        """
        # Phase 1: Compute sum of squares of all windows (including flipped versions)
        # This ensures the tight frame property: sum of squares equals 1 at each frequency
        sum_squared_windows = np.zeros(size)
        indices, values = windows[0][0][0]
        idx_flat = indices.ravel()
        val_flat = values.ravel()
        sum_squared_windows.flat[idx_flat] += val_flat**2
        for scale_idx in range(1, num_scales):
            # Iterate over actual number of directions (may be less than dimension for "wavelet" mode)
            for direction_idx in range(len(windows[scale_idx])):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    indices, values = windows[scale_idx][direction_idx][wedge_idx]
                    idx_flat = indices.ravel()
                    val_flat = values.ravel()
                    sum_squared_windows.flat[idx_flat] += val_flat**2
                    # Also accumulate flipped version for symmetry
                    # Only flip if we have the full number of directions (not for "wavelet" mode at highest scale)
                    # For "wavelet" mode, the ring window is already symmetric, so no flipping needed
                    if len(windows[scale_idx]) == dimension:
                        temp_window = np.zeros(size)
                        temp_window.flat[idx_flat] = val_flat**2
                        temp_window = UDCTWindow._flip_with_fft_shift(
                            temp_window, direction_idx
                        )
                        sum_squared_windows += temp_window

        # Phase 2: Normalize each window by dividing by sqrt(sum of squares)
        # This ensures perfect reconstruction (tight frame property)
        sum_squared_windows = np.sqrt(sum_squared_windows)
        sum_squared_windows_flat = sum_squared_windows.ravel()
        indices, values = windows[0][0][0]
        idx_flat = indices.ravel()
        val_flat = values.ravel()
        val_flat[:] /= sum_squared_windows_flat[idx_flat]
        for scale_idx in range(1, num_scales):
            # Iterate over actual number of directions (may be less than dimension for "wavelet" mode)
            for direction_idx in range(len(windows[scale_idx])):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    indices, values = windows[scale_idx][direction_idx][wedge_idx]
                    idx_flat = indices.ravel()
                    val_flat = values.ravel()
                    val_flat[:] /= sum_squared_windows_flat[idx_flat]

    @staticmethod
    def _calculate_decimation_ratios_with_lowest(
        num_scales: int,
        dimension: int,
        angular_wedges_config: IntegerNDArray,
        direction_mappings: list[IntegerNDArray],
        high_frequency_mode: str = "curvelet",
    ) -> list[IntegerNDArray]:
        """
        Calculate decimation ratios for each scale and direction.

        Parameters
        ----------
        num_scales : int
            Total number of scales (including lowpass scale).
        dimension : int
            Dimensionality of the transform.
        angular_wedges_config : IntegerNDArray
            Configuration array specifying number of angular wedges.
        direction_mappings : list[IntegerNDArray]
            Direction mappings for each resolution.
        high_frequency_mode : str, optional
            High frequency mode. For "wavelet" mode, highest scale has decimation=1.
            Default is "curvelet".

        Returns
        -------
        list[IntegerNDArray]
            List of decimation ratio arrays, one per scale.
        """
        decimation_ratios: list[IntegerNDArray] = [
            np.full((1, dimension), fill_value=2 ** (num_scales - 2), dtype=int)
        ]
        for scale_idx in range(1, num_scales):
            # For "wavelet" mode at highest scale, use decimation=1 with shape (1, dim)
            if high_frequency_mode == "wavelet" and scale_idx == num_scales - 1:
                decimation_ratios.append(np.ones((1, dimension), dtype=int))
            else:
                decimation_ratios.append(
                    np.full(
                        (dimension, dimension),
                        fill_value=2 ** (num_scales - scale_idx),
                        dtype=int,
                    )
                )
                for direction_idx in range(dimension):
                    other_directions = direction_mappings[scale_idx - 1][
                        direction_idx, :
                    ]
                    decimation_ratios[scale_idx][direction_idx, other_directions] = (
                        2
                        * angular_wedges_config[scale_idx - 1, other_directions]
                        * 2 ** (num_scales - 1 - scale_idx)
                        // 3
                    )
        return decimation_ratios

    @staticmethod
    def _inplace_sort_windows(
        windows: UDCTWindows,
        indices: dict[int, dict[int, IntegerNDArray]],
        num_scales: int,
    ) -> None:
        """
        Sort windows in-place by their angular indices.

        Parameters
        ----------
        windows : UDCTWindows
            Windows to sort (modified in-place).
        indices : dict[int, dict[int, IntegerNDArray]]
            Angular indices dictionary (modified in-place).
        num_scales : int
            Total number of scales (including lowpass scale).
        """
        for scale_idx in range(1, num_scales):
            # Iterate over actual dimension indices present (may be less than dimension for "wavelet" mode)
            for dimension_idx in indices[scale_idx]:
                angular_index_array = indices[scale_idx][dimension_idx]

                max_index_value = angular_index_array.max() + 1
                sorted_indices = np.argsort(
                    sum(
                        max_index_value**position_weight
                        * angular_index_array[:, column_index]
                        for column_index, position_weight in enumerate(
                            range(angular_index_array.shape[1] - 1, -1, -1)
                        )
                    )
                )

                indices[scale_idx][dimension_idx] = angular_index_array[sorted_indices]
                windows[scale_idx][dimension_idx] = [
                    windows[scale_idx][dimension_idx][idx] for idx in sorted_indices
                ]

    @staticmethod
    def _build_angle_indices_1d(
        scale_idx: int,
        dimension_idx: int,
        angle_functions: dict[int, dict[tuple[int, int], npt.NDArray[F]]],
        dimension: int,
    ) -> IntegerNDArray:
        """
        Build multi-dimensional angle index combinations using Kronecker products.

        Parameters
        ----------
        scale_idx : int
            Scale index (1-based).
        dimension_idx : int
            Dimension index (0-based).
        angle_functions : dict[int, dict[tuple[int, int], npt.NDArray[F]]]
            Dictionary of angle functions by scale and dimension.
        dimension : int
            Dimensionality of the transform.

        Returns
        -------
        IntegerNDArray
            Array of shape (num_windows, dim-1) containing angle index combinations.
        """
        angle_indices_1d = np.arange(
            len(angle_functions[scale_idx - 1][(dimension_idx, 0)])
        )[:, None]
        for angle_dim_idx in range(1, dimension - 1):
            num_angles = len(
                angle_functions[scale_idx - 1][(dimension_idx, angle_dim_idx)]
            )
            angle_indices_2d = np.arange(
                len(angle_functions[scale_idx - 1][(dimension_idx, angle_dim_idx)])
            )[:, None]
            kron_1 = np.kron(angle_indices_1d, np.ones((num_angles, 1), dtype=int))
            kron_2 = np.kron(
                np.ones((angle_indices_1d.shape[0], 1), dtype=int),
                angle_indices_2d,
            )
            angle_indices_1d = np.c_[kron_1, kron_2]
        return angle_indices_1d

    @staticmethod
    def _process_single_window(
        scale_idx: int,
        dimension_idx: int,
        window_index: int,
        angle_indices_1d: IntegerNDArray,
        angle_functions: dict[int, dict[tuple[int, int], npt.NDArray[F]]],
        angle_indices: dict[int, dict[tuple[int, int], IntegerNDArray]],
        bandpass_windows: dict[int, npt.NDArray[F]],
        direction_mappings: list[IntegerNDArray],
        max_angles_per_dim: IntegerNDArray,
        shape: tuple[int, ...],
        dimension: int,
        window_threshold: float,
    ) -> tuple[list[tuple[IntpNDArray, npt.NDArray[F]]], IntegerNDArray]:
        r"""
        Process a single window_index value, constructing curvelet windows with symmetry.

        This method implements the window construction algorithm from Nguyen & Chauris
        (2010), Section IV. It builds a single curvelet window by combining radial
        (bandpass) and angular (directional) components, then generates symmetric
        flipped versions to satisfy the partition of unity condition required for
        the tight frame property.

        The method processes one window_index completely independently, building the
        window, generating flipped versions for symmetry, and converting to sparse
        format. Each call is stateless with no shared state.

        References
        ----------
        .. [1] Nguyen, T. T., and H. Chauris, 2010, "Uniform Discrete Curvelet
           Transform": IEEE Transactions on Signal Processing, 58, 3618-3634.
           DOI: 10.1109/TSP.2010.2047666

        Parameters
        ----------
        scale_idx : int
            High-frequency scale index, ranging from 1 to num_scales-1 (inclusive).
            Note: Scale 0 (lowpass) is handled separately in :py:meth:`UDCTWindow.compute`
            and is never passed to this method. When accessing pre-computed arrays like
            `angle_functions` or `bandpass_windows`, use `scale_idx - 1` because
            those arrays are 0-indexed for high-frequency scales.
            Corresponds to resolution level j in the paper (Section IV).
        dimension_idx : int
            Dimension index (0-based), corresponding to direction l in the paper.
        window_index : int
            Window index to process, selecting a specific angular wedge combination.
        angle_indices_1d : IntegerNDArray
            Pre-computed angle index combinations from Kronecker products.
        angle_functions : dict[int, dict[tuple[int, int], npt.NDArray[F]]]
            Dictionary of angle functions A_{j,l} by scale and dimension (Section IV).
        angle_indices : dict[int, dict[tuple[int, int], IntegerNDArray]]
            Dictionary of angle indices by scale and dimension.
        bandpass_windows : dict[int, npt.NDArray[F]]
            Dictionary of Meyer wavelet-based bandpass filters F_j by scale (Section IV).
        direction_mappings : list[IntegerNDArray]
            Direction mappings for each resolution, used to determine flip axes.
        max_angles_per_dim : IntegerNDArray
            Maximum angles per dimension, used to determine which indices need flipping.
        shape : tuple[int, ...]
            Shape of the input data.
        dimension : int
            Dimensionality of the transform.
        window_threshold : float
            Threshold for sparse window storage.

        Returns
        -------
        tuple[list[tuple[IntpNDArray, npt.NDArray[F]]], IntegerNDArray]
            Tuple containing:
            - List of window tuples (indices, values) for this window_index
              (including original and flipped versions) in sparse format
            - angle_indices_2d : IntegerNDArray, shape (num_windows, dim-1)
              Array of angle indices for all windows (original + flipped versions).
              Each row corresponds to one window, and each column corresponds to
              one angular dimension. Values are 0-based indices indicating which
              angular wedge is used in each dimension. The first row contains the
              original angle indices from angle_indices_1d[window_index, :], and
              subsequent rows contain flipped versions generated for symmetry.

        Notes
        -----
        **Window Construction (Section IV, Nguyen & Chauris 2010)**:
        The base window is constructed as :math:`W_{j,l} = F_j \cdot A_{j,l}`, where:
        - :math:`F_j` is the Meyer wavelet-based bandpass filter for scale :math:`j`
        - :math:`A_{j,l}` is the angular function for direction :math:`l`, constructed via
          Kronecker products of 1D angle functions

        The window is then shifted by :math:`\text{size}//4` in each dimension and
        square-rooted to center it in frequency space.

        **Symmetry Generation**:
        For each window at angle index :math:`i`, this method generates symmetric
        windows at reflected angle indices. The reflection formula is
        :math:`i' = \text{max\_angles} - 1 - i` for each angular dimension.

        The flipped windows are created by applying frequency-domain flips along
        appropriate axes using :py:meth:`UDCTWindow._flip_with_fft_shift`. This ensures
        proper coverage of both positive and negative frequencies, which is required
        for the partition of unity condition (normalization is performed separately in
        :py:meth:`UDCTWindow.compute`).

        **Angle Indices Tracking**:
        The angle_indices_2d array tracks which angular wedges are used by each
        window. For a given window_index, we start with a single set of angle
        indices from angle_indices_1d, then generate flipped versions by
        reflecting indices across different angular dimensions. The first row
        contains the original angle indices, and subsequent rows contain flipped
        versions.
        """
        # Step 1: Build base window W_{j,l} = F_j · A_{j,l} (Section IV, Nguyen & Chauris 2010)
        # Initialize with unity: W = 1
        window: npt.NDArray[F] = np.ones(shape, dtype=float)

        # Multiply by angular functions A_{j,l} for each angular dimension
        # These provide directional selectivity via Kronecker products
        for angle_dim_idx in range(dimension - 1):
            angle_idx = angle_indices_1d.reshape(len(angle_indices_1d), -1)[
                window_index, angle_dim_idx
            ]
            # Get 1D angle function for this dimension and angle index
            # Note: scale_idx - 1 because angle_functions is 0-indexed for high-frequency scales
            # (scale 0 is lowpass, handled separately; scales 1..(num_scales-1) map to indices 0..(num_scales-2))
            angle_func = angle_functions[scale_idx - 1][(dimension_idx, angle_dim_idx)][
                angle_idx
            ]
            angle_idx_mapping = angle_indices[scale_idx - 1][
                (dimension_idx, angle_dim_idx)
            ]
            # Expand 1D angle function to N-D using Kronecker products
            # This creates the multi-dimensional angular wedge A_{j,l}
            kron_angle = UDCTWindow._compute_angle_kronecker_product(
                angle_func, angle_idx_mapping, shape, dimension
            )
            window *= kron_angle

        # Multiply by bandpass filter F_j (Meyer wavelet-based, Section IV)
        # This provides scale selectivity
        # Note: bandpass_windows[scale_idx] directly because bandpass_windows[0] is lowpass,
        # and bandpass_windows[1..(num_scales-1)] correspond to high-frequency scales 1..(num_scales-1)
        window *= bandpass_windows[scale_idx]

        # Apply frequency shift (size//4 in each dimension) and square root
        # The shift centers the window in frequency space, and square root
        # ensures proper normalization for the partition of unity condition
        window = np.sqrt(circular_shift(window, tuple(s // 4 for s in shape)))

        # Step 2: Generate symmetric flipped versions for partition of unity
        # The partition of unity requires: |W_0(ω)|² + ∑|W_{j,l}(ω)|² + ∑|W_{j,l}(-ω)|² = 1
        # (Section IV, Nguyen & Chauris 2010)
        # We need symmetric windows W_{j,l}(-ω) to cover negative frequencies

        def needs_flipping(
            angle_idx: IntegerNDArray, flip_dimension_index: int
        ) -> bool:
            """
            Check if an angle index needs flipping for the given dimension.

            Only indices in the lower half need flipping to avoid duplicates.
            Condition: 2*(i+1) <= max_angles ensures we only flip indices
            that haven't been generated from a previous flip.
            """
            return bool(
                2 * (angle_idx[flip_dimension_index] + 1)
                <= max_angles_per_dim[flip_dimension_index]
            )

        def flip_angle_idx(
            angle_idx: IntegerNDArray, flip_dimension_index: int
        ) -> IntegerNDArray:
            """
            Compute the flipped angle index for the given dimension.

            Reflection formula: i' = max_angles - 1 - i
            This creates the symmetric counterpart needed for W_{j,l}(-ω)
            in the partition of unity condition.
            """
            flipped = angle_idx.copy()
            flipped[flip_dimension_index] = (
                max_angles_per_dim[flip_dimension_index]
                - 1
                - angle_idx[flip_dimension_index]
            )
            return flipped

        # First pass: compute all angle indices that will exist (without creating windows)
        # Build the complete set of angle indices (original + all flipped versions)
        # This determines how many windows we need to create
        def add_flipped_indices(
            indices_list: list[IntegerNDArray], flip_dimension_index: int
        ) -> list[IntegerNDArray]:
            """
            Add flipped indices for a given dimension.

            For each index in the list that needs flipping, generate its
            symmetric counterpart. This builds the complete set of angle
            indices needed for the partition of unity.
            """
            needs_flip_mask = np.array(
                [needs_flipping(idx, flip_dimension_index) for idx in indices_list]
            )
            flipped_indices = [
                flip_angle_idx(indices_list[function_index], flip_dimension_index)
                for function_index in np.where(needs_flip_mask)[0]
            ]
            return indices_list + flipped_indices

        # Start with the original angle index for this window_index
        angle_indices_list = [angle_indices_1d[window_index, :].copy()]
        # Iterate through angular dimensions from dim-2 down to 0
        # This ensures we generate all necessary symmetric combinations
        for flip_dimension_index in range(dimension - 2, -1, -1):
            angle_indices_list = add_flipped_indices(
                angle_indices_list, flip_dimension_index
            )

        # Create arrays once with final size (pre-allocate for efficiency)
        num_windows = len(angle_indices_list)
        angle_indices_2d = np.array(angle_indices_list)
        window_functions = np.zeros((num_windows, *shape), dtype=window.dtype)

        # Second pass: fill in windows, applying flips as needed
        # Build mapping from angle index to its source window and flip dimension
        # This allows us to efficiently generate flipped windows from their sources
        angle_to_source: dict[tuple[int, ...], tuple[int, int]] = {
            tuple(angle_indices_2d[0]): (0, -1)  # Original has no source
        }
        # Build the complete mapping by iterating through dimensions
        # We need to track which window index corresponds to each angle index
        # as we build the mapping, so we can use the correct source window
        angle_idx_to_window_idx: dict[tuple[int, ...], int] = {
            tuple(angle_indices_2d[0]): 0
        }
        for flip_dimension_index in range(dimension - 2, -1, -1):
            # Iterate through all angle indices we've seen so far (not just sources)
            for source_idx_tuple, source_window_idx in list(
                angle_idx_to_window_idx.items()
            ):
                source_angle_idx = np.array(source_idx_tuple)
                if needs_flipping(source_angle_idx, flip_dimension_index):
                    flipped_angle_idx = flip_angle_idx(
                        source_angle_idx, flip_dimension_index
                    )
                    flipped_angle_idx_tuple = tuple(flipped_angle_idx)
                    if flipped_angle_idx_tuple not in angle_to_source:
                        # Use the window index that corresponds to the source angle index
                        # This ensures we flip from the correct window (not always window 0)
                        angle_to_source[flipped_angle_idx_tuple] = (
                            source_window_idx,
                            flip_dimension_index,
                        )
                        # Track the new angle index -> window index mapping
                        # The window index will be assigned when we fill in windows
                        # For now, we'll use the next available index
                        angle_idx_to_window_idx[flipped_angle_idx_tuple] = len(
                            angle_idx_to_window_idx
                        )

        # Fill windows by iterating through angle_indices_2d in order
        # Store the original window (computed in Step 1)
        window_functions[0] = window

        # Generate flipped windows W_{j,l}(-ω) by applying frequency-domain flips
        # These symmetric windows are needed for the partition of unity condition
        for window_idx in range(1, num_windows):
            angle_idx_tuple = tuple(angle_indices_2d[window_idx])
            source_window_idx, flip_dimension_index = angle_to_source[angle_idx_tuple]
            # Get the physical axis along which to flip (from direction mappings)
            # Note: scale_idx - 1 because direction_mappings is 0-indexed for high-frequency scales
            # (scale 0 is lowpass, handled separately; scales 1..(num_scales-1) map to indices 0..(num_scales-2))
            flip_axis_dimension = int(
                direction_mappings[scale_idx - 1][dimension_idx, flip_dimension_index]
            )
            # Apply frequency-domain flip: W_flipped = flip_with_fft_shift(W_source, axis)
            # This creates the symmetric window for negative frequencies
            window_functions[window_idx] = UDCTWindow._flip_with_fft_shift(
                window_functions[source_window_idx], flip_axis_dimension
            )

        # Convert all window functions to sparse format
        window_tuples = [
            UDCTWindow._to_sparse(
                window_functions[function_index],
                window_threshold,
            )
            for function_index in range(window_functions.shape[0])
        ]

        return window_tuples, angle_indices_2d

    def compute(
        self,
    ) -> tuple[UDCTWindows, list[IntegerNDArray], dict[int, dict[int, IntegerNDArray]]]:
        r"""
        Compute curvelet windows in frequency domain for UDCT transform.

        This method implements the window construction algorithm from Nguyen & Chauris
        (2010), Section IV. It generates frequency-domain windows by combining Meyer
        wavelet-based bandpass filters with angular wedges, then normalizes them to
        satisfy the partition of unity condition for the tight frame property.

        References
        ----------
        .. [1] Nguyen, T. T., and H. Chauris, 2010, "Uniform Discrete Curvelet
           Transform": IEEE Transactions on Signal Processing, 58, 3618-3634.
           DOI: 10.1109/TSP.2010.2047666

        Returns
        -------
        windows : UDCTWindows
            Curvelet windows in sparse format. Structure is:
            windows[scale][direction][wedge] = (indices, values) tuple
            where scale 0 is low-frequency, scales 1..(num_scales-1) are high-frequency bands.
            For "wavelet" mode at highest scale, windows[highest_scale] has 1 direction with 1 wedge
            (ring-shaped window encompassing the entire frequency ring, no angular components).
        decimation_ratios : list[IntegerNDArray]
            Decimation ratios for each scale and direction. Structure:
            - decimation_ratios[0]: shape (1, dim) for low-frequency band
            - decimation_ratios[scale]: shape (dim, dim) for scale > 0 (normal modes)
            - decimation_ratios[highest_scale]: shape (1, dim) with values=1 for "wavelet" mode
        indices : dict[int, dict[int, IntegerNDArray]]
            Angular indices for each window. Structure:
            indices[scale][direction] = array of shape (num_wedges, dim-1)
            containing angular indices for each wedge

        Notes
        -----
        **Window Construction (Section IV, Nguyen & Chauris 2010)**:
        Constructs windows :math:`W_{j,l} = F_j \cdot A_{j,l}` where :math:`F_j` are
        Meyer wavelet-based bandpass filters (via :py:meth:`UDCTWindow._create_bandpass_windows`)
        and :math:`A_{j,l}` are angular functions (via :py:meth:`UDCTWindow._create_angle_info`).
        Low-frequency window (scale 0) is handled separately; high-frequency windows
        (scales 1..(num_scales-1)) are generated via :py:meth:`UDCTWindow._process_single_window`.

        **Partition of Unity (Section IV)**:
        Windows are normalized via :py:meth:`UDCTWindow._inplace_normalize_windows` to
        satisfy:

        .. math::
           |W_0(\omega)|^2 + \sum_{j,l} |W_{j,l}(\omega)|^2 + \sum_{j,l} |W_{j,l}(-\omega)|^2 = 1

        This ensures a tight frame, enabling perfect reconstruction and energy
        preservation: :math:`\|f\|^2 = \sum |c_{j,l,k}|^2`. Windows are then sorted
        via :py:meth:`UDCTWindow._inplace_sort_windows` for consistent ordering.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy._utils import ParamUDCT
        >>> from curvelets.numpy._udct_windows import UDCTWindow
        >>>
        >>> # Create parameters for 2D transform with 3 scales
        >>> params = ParamUDCT(
        ...     ndim=2,
        ...     shape=(64, 64),
        ...     angular_wedges_config=np.array([[3], [6], [12]]),
        ...     window_overlap=0.15,
        ...     window_threshold=1e-5,
        ...     radial_frequency_params=(np.pi/3, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3)
        ... )
        >>>
        >>> # Compute windows
        >>> window_computer = UDCTWindow(params)
        >>> windows, decimation_ratios, indices = window_computer.compute()
        >>>
        >>> # Check structure
        >>> len(windows)  # Number of scales
        4
        >>> len(windows[0][0])  # Low-frequency band has 1 window
        1
        >>> len(windows[1][0])  # First high-frequency scale has multiple wedges
        3
        >>>
        >>> # Check decimation ratios
        >>> decimation_ratios[0].shape  # Low-frequency
        (1, 2)
        >>> decimation_ratios[1].shape  # High-frequency
        (2, 2)
        >>>
        >>> # Check indices structure
        >>> indices[1][0].shape  # Angular indices for first direction
        (3, 1)
        """
        frequency_grid, bandpass_windows = UDCTWindow._create_bandpass_windows(
            num_scales=self.num_scales,
            shape=self.shape,
            radial_frequency_params=self.radial_frequency_params,
        )
        low_frequency_window = circular_shift(
            np.sqrt(bandpass_windows[0]), tuple(s // 4 for s in self.shape)
        )

        # convert to sparse format
        windows: UDCTWindows = []
        windows.append([])
        windows[0].append([])
        windows[0][0] = [
            UDCTWindow._to_sparse(low_frequency_window, self.window_threshold)
        ]

        indices: dict[int, dict[int, IntegerNDArray]] = {}
        indices[0] = {}
        indices[0][0] = np.zeros((1, 1), dtype=int)
        direction_mappings = UDCTWindow._create_direction_mappings(
            dimension=self.dimension, num_scales=self.num_scales
        )
        angle_functions, angle_indices = UDCTWindow._create_angle_info(
            frequency_grid,
            dimension=self.dimension,
            num_scales=self.num_scales,
            angular_wedges_config=self.angular_wedges_config,
            window_overlap=self.window_overlap,
            high_frequency_mode=self.high_frequency_mode,
        )

        decimation_ratios = UDCTWindow._calculate_decimation_ratios_with_lowest(
            num_scales=self.num_scales,
            dimension=self.dimension,
            angular_wedges_config=self.angular_wedges_config,
            direction_mappings=direction_mappings,
            high_frequency_mode=self.high_frequency_mode,
        )

        # Generate windows for each high-frequency scale (scales 1 to num_scales-1)
        # Each scale contains multiple directions (one per dimension), and each
        # direction contains multiple wedges (windows with different angular orientations)
        for scale_idx in range(1, self.num_scales):
            windows.append([])
            indices[scale_idx] = {}

            # For "wavelet" mode at highest scale, create a single ring-shaped window
            # (bandpass filter only, no angular components)
            if (
                self.high_frequency_mode == "wavelet"
                and scale_idx == self.num_scales - 1
            ):
                # Create ring window: use complement of lowpass filter to cover entire high-frequency ring
                # For the highest scale, we want all frequencies NOT in the lowpass
                # This ensures we cover the entire frequency ring without gaps
                # Get the lowpass filter (bandpass_windows[0] is the lowpass)
                lowpass_filter = bandpass_windows[0].copy()
                # Ring window is complement: 1 - lowpass (all high frequencies)
                ring_window: npt.NDArray[np.float64] = 1.0 - lowpass_filter
                # Ensure non-negative (should already be, but be safe)
                ring_window = np.maximum(ring_window, 0.0)

                # Apply frequency shift and square root (matching normal window creation)
                ring_window = np.sqrt(
                    circular_shift(ring_window, tuple(s // 4 for s in self.shape))
                )

                # Convert to sparse format
                ring_window_sparse = UDCTWindow._to_sparse(
                    ring_window, self.window_threshold
                )

                # Store as single direction with single window
                windows[scale_idx].append([ring_window_sparse])
                indices[scale_idx][0] = np.zeros((1, self.dimension - 1), dtype=int)
            else:
                # Normal curvelet mode: process each direction (dimension) independently
                for dimension_idx in range(self.dimension):
                    # Build angle index combinations once per (scale_idx, dimension_idx)
                    # This determines which angular wedges will be created
                    angle_indices_1d = UDCTWindow._build_angle_indices_1d(
                        scale_idx=scale_idx,
                        dimension_idx=dimension_idx,
                        angle_functions=angle_functions,
                        dimension=self.dimension,
                    )
                    num_windows = angle_indices_1d.shape[0]
                    max_angles_per_dim = self.angular_wedges_config[
                        scale_idx - 1,
                        direction_mappings[scale_idx - 1][dimension_idx, :],
                    ]

                    # Process each window_index independently using list comprehension
                    # Each call returns (list of window tuples, angle_indices_2d array)
                    # Windows include original and flipped versions for symmetry
                    window_results = [
                        UDCTWindow._process_single_window(
                            scale_idx=scale_idx,
                            dimension_idx=dimension_idx,
                            window_index=window_index,
                            angle_indices_1d=angle_indices_1d,
                            angle_functions=angle_functions,
                            angle_indices=angle_indices,
                            bandpass_windows=bandpass_windows,
                            direction_mappings=direction_mappings,
                            max_angles_per_dim=max_angles_per_dim,
                            shape=self.shape,
                            dimension=self.dimension,
                            window_threshold=self.window_threshold,
                        )
                        for window_index in range(num_windows)
                    ]

                    # Combine results from all window_index values:
                    # - Flatten nested window lists into a single list
                    # - Concatenate all angle_indices_2d arrays into one array
                    all_windows = [
                        window
                        for window_list, _ in window_results
                        for window in window_list
                    ]
                    all_angle_indices = [angle_idx for _, angle_idx in window_results]
                    angle_index_array = (
                        np.concatenate(all_angle_indices, axis=0)
                        if all_angle_indices
                        else np.zeros((0, self.dimension - 1), dtype=int)
                    )

                    # Store results for this (scale_idx, dimension_idx) combination
                    windows[scale_idx].append(all_windows)
                    indices[scale_idx][dimension_idx] = angle_index_array

        UDCTWindow._inplace_normalize_windows(
            windows,
            size=self.shape,
            dimension=self.dimension,
            num_scales=self.num_scales,
        )

        UDCTWindow._inplace_sort_windows(
            windows=windows,
            indices=indices,
            num_scales=self.num_scales,
        )

        return windows, decimation_ratios, indices
