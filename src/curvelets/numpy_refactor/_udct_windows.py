from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from math import ceil
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from ._typing import (
    FloatingNDArray,
    IntegerNDArray,
    IntpNDArray,
    UDCTWindows,
)
from ._utils import ParamUDCT, circshift, meyer_window

D_T = TypeVar("D_T", bound=np.floating)


class UDCTWindow:
    """
    Window computation for Uniform Discrete Curvelet Transform.

    This class encapsulates all window computation functionality for the UDCT,
    including bandpass filter creation, angle function computation, window
    normalization, and sparse format conversion. All methods are stateless
    staticmethods, making the class a namespace for window computation utilities.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor._utils import ParamUDCT
    >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
    >>>
    >>> # Create parameters for 2D transform with 3 scales
    >>> params = ParamUDCT(
    ...     size=(64, 64),
    ...     res=3,
    ...     dim=2,
    ...     angular_wedges_config=np.array([[3], [6], [12]]),
    ...     window_overlap=0.15,
    ...     window_threshold=1e-5,
    ...     radial_frequency_params=(np.pi/3, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3)
    ... )
    >>>
    >>> # Compute windows
    >>> windows, decimation_ratios, indices = UDCTWindow.compute(params)
    >>>
    >>> # Check structure
    >>> len(windows)  # Number of scales (0 + res)
    4
    >>> len(windows[0][0])  # Low-frequency band has 1 window
    1
    >>> len(windows[1][0])  # First high-frequency scale has multiple wedges
    3
    """

    @staticmethod
    def _compute_angle_component(
        x_primary: FloatingNDArray, x_secondary: FloatingNDArray
    ) -> FloatingNDArray:
        """
        Compute one angle component from meshgrid coordinates.

        Parameters
        ----------
        x_primary : FloatingNDArray
            Primary coordinate grid (used for conditions).
        x_secondary : FloatingNDArray
            Secondary coordinate grid.

        Returns
        -------
        FloatingNDArray
            Angle component array.
        """
        # Compute angle component using piecewise function:
        # When primary coordinate dominates (|x_secondary| <= |x_primary|), use -x_secondary/x_primary
        primary_ratio: npt.NDArray[np.floating] = np.zeros_like(x_primary, dtype=float)
        mask = (x_primary != 0) & (np.abs(x_secondary) <= np.abs(x_primary))
        primary_ratio[mask] = -x_secondary[mask] / x_primary[mask]

        # When secondary coordinate dominates (|x_primary| < |x_secondary|), use x_primary/x_secondary
        secondary_ratio: npt.NDArray[np.floating] = np.zeros_like(
            x_primary, dtype=float
        )
        mask = (x_secondary != 0) & (np.abs(x_primary) < np.abs(x_secondary))
        secondary_ratio[mask] = x_primary[mask] / x_secondary[mask]

        # Wrap secondary_ratio to the correct range by adding/subtracting 2
        wrapped_ratio = secondary_ratio.copy()
        wrapped_ratio[secondary_ratio < 0] = secondary_ratio[secondary_ratio < 0] + 2
        wrapped_ratio[secondary_ratio > 0] = secondary_ratio[secondary_ratio > 0] - 2

        # Combine ratios and set special case for x_primary >= 0
        angle_component = primary_ratio + wrapped_ratio
        angle_component[x_primary >= 0] = -2
        return angle_component

    @staticmethod
    def _create_angle_grids_from_frequency_grids(
        frequency_grid_1: FloatingNDArray, frequency_grid_2: FloatingNDArray
    ) -> tuple[FloatingNDArray, FloatingNDArray]:
        """
        Adapt frequency grids for angle computation.

        Parameters
        ----------
        frequency_grid_1 : FloatingNDArray
            First frequency grid.
        frequency_grid_2 : FloatingNDArray
            Second frequency grid.

        Returns
        -------
        tuple[FloatingNDArray, FloatingNDArray]
            Adapted grid arrays angle_grid_2 and angle_grid_1.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> frequency_grid_1 = np.linspace(-1.5*np.pi, 0.5*np.pi, 64, endpoint=False)
        >>> frequency_grid_2 = np.linspace(-1.5*np.pi, 0.5*np.pi, 64, endpoint=False)
        >>> angle_grid_2, angle_grid_1 = UDCTWindow._create_angle_grids_from_frequency_grids(frequency_grid_1, frequency_grid_2)
        >>> angle_grid_2.shape
        (64, 64)
        """
        meshgrid_dim1, meshgrid_dim2 = np.meshgrid(frequency_grid_2, frequency_grid_1)

        # Compute angle_grid_1 using meshgrid_dim1 as primary, meshgrid_dim2 as secondary
        angle_grid_1 = UDCTWindow._compute_angle_component(meshgrid_dim1, meshgrid_dim2)

        # Compute angle_grid_2 using meshgrid_dim2 as primary, meshgrid_dim1 as secondary (swapped)
        angle_grid_2 = UDCTWindow._compute_angle_component(meshgrid_dim2, meshgrid_dim1)

        return angle_grid_2, angle_grid_1

    @staticmethod
    def _create_angle_functions(
        angle_grid: FloatingNDArray,
        direction: int,
        num_angular_wedges: int,
        window_overlap: float,
    ) -> FloatingNDArray:
        """
        Create angle functions using Meyer windows.

        Parameters
        ----------
        angle_grid : FloatingNDArray
            Angle grid.
        direction : int
            Direction index (1 or 2).
        num_angular_wedges : int
            Number of angular wedges.
        window_overlap : float
            Window overlap parameter.

        Returns
        -------
        FloatingNDArray
            Array of angle functions.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> angle_grid = np.linspace(-2, 2, 64)
        >>> angle_funcs = UDCTWindow._create_angle_functions(angle_grid, 1, 3, 0.15)
        >>> angle_funcs.shape[0]
        2
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
        angle_functions_list = []
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
        angle_function_1d: FloatingNDArray,
        dimension_permutation: IntegerNDArray,
        param_udct: ParamUDCT,
    ) -> FloatingNDArray:
        """
        Compute Kronecker product for angle functions.

        Parameters
        ----------
        angle_function_1d : FloatingNDArray
            Angle function array.
        dimension_permutation : IntegerNDArray
            Dimension permutation indices.
        param_udct : ParamUDCT
            UDCT parameters.

        Returns
        -------
        FloatingNDArray
            Kronecker product result with shape matching param_udct.size.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._utils import ParamUDCT
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> params = ParamUDCT(
        ...     size=(64, 64), res=3, dim=2,
        ...     angular_wedges_config=np.array([[3], [6], [12]]),
        ...     window_overlap=0.15, window_threshold=1e-5,
        ...     radial_frequency_params=(np.pi/3, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3)
        ... )
        >>> angle_function_1d = np.ones(64)
        >>> dimension_permutation = np.array([1, 2])
        >>> result = UDCTWindow._compute_angle_kronecker_product(angle_function_1d, dimension_permutation, params)
        >>> result.shape
        (64, 64)
        """
        # Pre-compute dimension sizes for Kronecker product
        kronecker_dimension_sizes: npt.NDArray[np.int_] = np.array(
            [
                np.prod(param_udct.size[: dimension_permutation[0] - 1]),
                np.prod(
                    param_udct.size[
                        dimension_permutation[0] : dimension_permutation[1] - 1
                    ]
                ),
                np.prod(param_udct.size[dimension_permutation[1] : param_udct.dim]),
            ],
            dtype=int,
        )

        # Expand 1D angle function to N-D using multi-step Kronecker products:
        # Step 1: Expand along dimension 1 (kronecker_dimension_sizes[1])
        kron_step1 = np.kron(
            np.ones((kronecker_dimension_sizes[1], 1), dtype=int), angle_function_1d
        )
        kron_step1_flat = kron_step1.ravel()
        kron_step2 = np.kron(
            np.ones((kronecker_dimension_sizes[2], 1), dtype=int), kron_step1_flat
        ).ravel()
        kron_step3 = np.kron(
            kron_step2, np.ones((kronecker_dimension_sizes[0], 1), dtype=int)
        ).ravel()
        return kron_step3.reshape(*param_udct.size)

    @staticmethod
    def _flip_with_fft_shift(
        input_array: FloatingNDArray, axis: int
    ) -> FloatingNDArray:
        """
        Flip array along specified axis with frequency domain shift.

        Parameters
        ----------
        input_array : FloatingNDArray
            Input array.
        axis : int
            Axis along which to flip.

        Returns
        -------
        FloatingNDArray
            Flipped and shifted array.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> flipped = UDCTWindow._flip_with_fft_shift(arr, 0)
        >>> flipped.shape
        (2, 2)
        """
        num_dimensions = input_array.ndim
        shift_vector: npt.NDArray[np.int_] = np.zeros((num_dimensions,), dtype=int)
        shift_vector[axis] = 1
        flipped_array = np.flip(input_array, axis)
        return circshift(flipped_array, tuple(shift_vector))

    @staticmethod
    def _to_sparse(
        arr: npt.NDArray[D_T], threshold: float
    ) -> tuple[IntpNDArray, npt.NDArray[D_T]]:
        """
        Convert array to sparse format.

        Parameters
        ----------
        arr : npt.NDArray[D_T]
            Input array.
        threshold : float
            Threshold for sparse storage (values above threshold are kept).

        Returns
        -------
        tuple[IntpNDArray, npt.NDArray[D_T]]
            Tuple of (indices, values) where indices are positions and values
            are the array values at those positions.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> arr = np.array([0.1, 0.5, 0.01, 0.8])
        >>> indices, values = UDCTWindow._to_sparse(arr, 0.2)
        >>> len(values)
        2
        """
        arr_flat = arr.ravel()
        indices = np.argwhere(arr_flat > threshold)
        return (indices, arr_flat[indices])

    @staticmethod
    def _nchoosek(n: Iterable[int] | IntegerNDArray, k: int) -> IntegerNDArray:
        """
        Generate all combinations of k elements from n.

        Parameters
        ----------
        n : Iterable[int] | npt.NDArray[np.int_]
            Iterable containing elements to choose from (list, array, range, etc.).
        k : int
            Number of elements to choose in each combination.

        Returns
        -------
        npt.NDArray[np.int_]
            Array of shape (C(n,k), k) containing all combinations.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> result = UDCTWindow._nchoosek([0, 1, 2], 2)
        >>> result.shape
        (3, 2)
        """
        return np.asarray(list(combinations(n, k)), dtype=int)

    @staticmethod
    def _create_bandpass_windows(
        num_scales: int,
        shape: tuple[int, ...],
        radial_frequency_params: tuple[float, float, float, float],
    ) -> tuple[dict[int, FloatingNDArray], dict[int, FloatingNDArray]]:
        """
            Create bandpass windows using Meyer wavelets for radial frequency decomposition.

        This function generates frequency-domain bandpass filters by constructing
        Meyer wavelet windows for each dimension and scale, then combining them
        using Kronecker products to create multi-dimensional bandpass filters.

        Parameters
        ----------
        num_scales : int
            Number of resolution scales for the transform.
        shape : tuple[int, ...]
            Shape of the input data, determines frequency grid size.
        radial_frequency_params : tuple[float, float, float, float]
            Four parameters defining radial frequency bands:
            - params[0], params[1]: Lower frequency band boundaries
            - params[2], params[3]: Upper frequency band boundaries

        Returns
        -------
        frequency_grid : dict[int, FloatingNDArray]
            Dictionary mapping dimension index to frequency grid array.
            Each grid spans [-1.5*pi, 0.5*pi) with size matching shape[dimension].
        bandpass_windows : dict[int, FloatingNDArray]
            Dictionary mapping scale index to bandpass window array.
            Scale 0 is low-frequency, scales 1..num_scales are high-frequency bands.
            Each window has shape matching input `shape`.

            Examples
            --------
            >>> import numpy as np
            >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
            >>> freq_grid, bandpass = UDCTWindow._create_bandpass_windows(
            ...     3, (64, 64), (np.pi/3, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3)
            ... )
            >>> len(bandpass)
            4
            >>> bandpass[0].shape
            (64, 64)
        """
        dimension = len(shape)
        frequency_grid: dict[int, FloatingNDArray] = {}
        meyer_windows: dict[tuple[int, int], FloatingNDArray] = {}
        for dimension_idx in range(dimension):
            frequency_grid[dimension_idx] = np.linspace(
                -1.5 * np.pi, 0.5 * np.pi, shape[dimension_idx], endpoint=False
            )

            meyer_params = np.array([-2, -1, *radial_frequency_params[:2]])
            abs_frequency_grid = np.abs(frequency_grid[dimension_idx])
            meyer_windows[(num_scales, dimension_idx)] = meyer_window(
                abs_frequency_grid, *meyer_params
            )
            if num_scales == 1:
                meyer_windows[(num_scales, dimension_idx)] += meyer_window(
                    np.abs(frequency_grid[dimension_idx] + 2 * np.pi), *meyer_params
                )
            meyer_params[2:] = radial_frequency_params[2:]
            meyer_windows[(num_scales + 1, dimension_idx)] = meyer_window(
                abs_frequency_grid, *meyer_params
            )

            for scale_idx in range(num_scales - 1, 0, -1):
                meyer_params[2:] = radial_frequency_params[:2]
                meyer_params[2:] /= 2 ** (num_scales - scale_idx)
                meyer_windows[(scale_idx, dimension_idx)] = meyer_window(
                    abs_frequency_grid, *meyer_params
                )

        bandpass_windows: dict[int, FloatingNDArray] = {}
        for scale_idx in range(num_scales, 0, -1):
            low_freq = np.array([1.0])
            high_freq = np.array([1.0])
            for dimension_idx in range(dimension - 1, -1, -1):
                low_freq = np.kron(meyer_windows[(scale_idx, dimension_idx)], low_freq)
                high_freq = np.kron(
                    meyer_windows[(scale_idx + 1, dimension_idx)], high_freq
                )
            low_freq_nd = low_freq.reshape(*shape)
            high_freq_nd = high_freq.reshape(*shape)
            bandpass_nd = high_freq_nd - low_freq_nd
            bandpass_nd[bandpass_nd < 0] = 0
            bandpass_windows[scale_idx] = bandpass_nd
        bandpass_windows[0] = low_freq_nd
        return frequency_grid, bandpass_windows

    @staticmethod
    def _create_direction_mappings(
        dimension: int, num_resolutions: int
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
        num_resolutions : int
            Number of resolution scales.

        Returns
        -------
        list[IntegerNDArray]
            List of arrays, one per resolution. Each array has shape (dimension, dimension-1)
            and contains indices of dimensions used for angle calculations on each hyperpyramid.

            Examples
            --------
            >>> import numpy as np
            >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
            >>> mappings = UDCTWindow._create_direction_mappings(2, 3)
            >>> len(mappings)
            3
            >>> mappings[0].shape
            (2, 1)
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
            for scale_idx in range(num_resolutions)
        ]

    @staticmethod
    def _create_angle_info(
        frequency_grid: dict[int, FloatingNDArray],
        dimension: int,
        num_resolutions: int,
        angular_wedges_config: IntegerNDArray,
        window_overlap: float,
    ) -> tuple[
        dict[int, dict[tuple[int, int], FloatingNDArray]],
        dict[int, dict[tuple[int, int], IntegerNDArray]],
    ]:
        """
        Create angle functions and indices for window computation.

        Parameters
        ----------
        frequency_grid : dict[int, FloatingNDArray]
            Dictionary mapping dimension index to frequency grid.
        dimension : int
            Dimensionality of the transform.
        num_resolutions : int
            Number of resolution scales.
        angular_wedges_config : IntegerNDArray
            Configuration array specifying number of angular wedges per scale and dimension.
        window_overlap : float
            Window overlap parameter.

        Returns
        -------
        tuple[dict[int, dict[tuple[int, int], FloatingNDArray]], dict[int, dict[tuple[int, int], IntegerNDArray]]]
            Tuple of (angle_functions, angle_indices) dictionaries.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> freq_grid = {0: np.linspace(-1.5*np.pi, 0.5*np.pi, 64, endpoint=False),
        ...              1: np.linspace(-1.5*np.pi, 0.5*np.pi, 64, endpoint=False)}
        >>> angle_funcs, angle_indices = UDCTWindow._create_angle_info(
        ...     freq_grid, 2, 3, np.array([[3], [6], [12]]), 0.15
        ... )
        >>> len(angle_funcs)
        3
        """
        dimension_permutations = UDCTWindow._nchoosek(np.arange(dimension), 2)
        angle_grid: dict[tuple[int, int], FloatingNDArray] = {}
        for pair_index, dimension_pair in enumerate(dimension_permutations):
            angle_grids = UDCTWindow._create_angle_grids_from_frequency_grids(
                frequency_grid[dimension_pair[0]], frequency_grid[dimension_pair[1]]
            )
            angle_grid[(pair_index, 0)] = angle_grids[0]
            angle_grid[(pair_index, 1)] = angle_grids[1]

        angle_functions: dict[int, dict[tuple[int, int], FloatingNDArray]] = {}
        angle_indices: dict[int, dict[tuple[int, int], IntegerNDArray]] = {}
        for scale_idx in range(num_resolutions):
            angle_functions[scale_idx] = {}
            angle_indices[scale_idx] = {}
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
                                window_overlap,
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
        num_resolutions: int,
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
        num_resolutions : int
            Number of resolution scales.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> from curvelets.numpy_refactor._typing import UDCTWindows
        >>> windows: UDCTWindows = [[[(np.array([[0]]), np.array([1.0]))]]]
        >>> UDCTWindow._inplace_normalize_windows(windows, (64, 64), 2, 3)
        """
        # Phase 1: Compute sum of squares of all windows (including flipped versions)
        # This ensures the tight frame property: sum of squares equals 1 at each frequency
        sum_squared_windows = np.zeros(size)
        indices, values = windows[0][0][0]
        idx_flat = indices.ravel()
        val_flat = values.ravel()
        sum_squared_windows.flat[idx_flat] += val_flat**2
        for scale_idx in range(1, num_resolutions + 1):
            for direction_idx in range(dimension):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    indices, values = windows[scale_idx][direction_idx][wedge_idx]
                    idx_flat = indices.ravel()
                    val_flat = values.ravel()
                    sum_squared_windows.flat[idx_flat] += val_flat**2
                    # Also accumulate flipped version for symmetry
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
        for scale_idx in range(1, num_resolutions + 1):
            for direction_idx in range(dimension):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    indices, values = windows[scale_idx][direction_idx][wedge_idx]
                    idx_flat = indices.ravel()
                    val_flat = values.ravel()
                    val_flat[:] /= sum_squared_windows_flat[idx_flat]

    @staticmethod
    def _calculate_decimation_ratios_with_lowest(
        num_resolutions: int,
        dimension: int,
        angular_wedges_config: IntegerNDArray,
        direction_mappings: list[IntegerNDArray],
    ) -> list[IntegerNDArray]:
        """
        Calculate decimation ratios for each scale and direction.

        Parameters
        ----------
        num_resolutions : int
            Number of resolution scales.
        dimension : int
            Dimensionality of the transform.
        angular_wedges_config : IntegerNDArray
            Configuration array specifying number of angular wedges.
        direction_mappings : list[IntegerNDArray]
            Direction mappings for each resolution.

        Returns
        -------
        list[IntegerNDArray]
            List of decimation ratio arrays, one per scale.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> mappings = UDCTWindow._create_mdirs(2, 3)
        >>> ratios = UDCTWindow._calculate_decimation_ratios_with_lowest(
        ...     3, 2, np.array([[3], [6], [12]]), mappings
        ... )
        >>> len(ratios)
        4
        """
        decimation_ratios: list[IntegerNDArray] = [
            np.full((1, dimension), fill_value=2 ** (num_resolutions - 1), dtype=int)
        ]
        for scale_idx in range(1, num_resolutions + 1):
            decimation_ratios.append(
                np.full(
                    (dimension, dimension),
                    fill_value=2.0 ** (num_resolutions - scale_idx + 1),
                    dtype=int,
                )
            )
            for direction_idx in range(dimension):
                other_directions = direction_mappings[scale_idx - 1][direction_idx, :]
                decimation_ratios[scale_idx][direction_idx, other_directions] = (
                    2
                    * angular_wedges_config[scale_idx - 1, other_directions]
                    * 2 ** (num_resolutions - scale_idx)
                    // 3
                )
        return decimation_ratios

    @staticmethod
    def _inplace_sort_windows(
        windows: UDCTWindows,
        indices: dict[int, dict[int, IntegerNDArray]],
        num_resolutions: int,
        dimension: int,
    ) -> None:
        """
        Sort windows in-place by their angular indices.

        Parameters
        ----------
        windows : UDCTWindows
            Windows to sort (modified in-place).
        indices : dict[int, dict[int, IntegerNDArray]]
            Angular indices dictionary (modified in-place).
        num_resolutions : int
            Number of resolution scales.
        dimension : int
            Dimensionality of the transform.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> from curvelets.numpy_refactor._typing import UDCTWindows
        >>> windows: UDCTWindows = [[[(np.array([[0]]), np.array([1.0]))]]]
        >>> indices = {1: {0: np.array([[0]])}}
        >>> UDCTWindow._inplace_sort_windows(windows, indices, 3, 2)
        """
        for scale_idx in range(1, num_resolutions + 1):
            for dimension_idx in range(dimension):
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
        angle_functions: dict[int, dict[tuple[int, int], FloatingNDArray]],
        parameters: ParamUDCT,
    ) -> IntegerNDArray:
        """
        Build multi-dimensional angle index combinations using Kronecker products.

        Parameters
        ----------
        scale_idx : int
            Scale index (1-based).
        dimension_idx : int
            Dimension index (0-based).
        angle_functions : dict[int, dict[tuple[int, int], FloatingNDArray]]
            Dictionary of angle functions by scale and dimension.
        parameters : ParamUDCT
            UDCT parameters.

        Returns
        -------
        IntegerNDArray
            Array of shape (num_windows, dim-1) containing angle index combinations.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._utils import ParamUDCT
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> params = ParamUDCT(
        ...     size=(64, 64), res=3, dim=2,
        ...     angular_wedges_config=np.array([[3], [6], [12]]),
        ...     window_overlap=0.15, window_threshold=1e-5,
        ...     radial_frequency_params=(np.pi/3, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3)
        ... )
        >>> # This would be called internally by compute()
        """
        angle_indices_1d = np.arange(
            len(angle_functions[scale_idx - 1][(dimension_idx, 0)])
        )[:, None]
        for angle_dim_idx in range(1, parameters.dim - 1):
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
        angle_functions: dict[int, dict[tuple[int, int], FloatingNDArray]],
        angle_indices: dict[int, dict[tuple[int, int], IntegerNDArray]],
        bandpass_windows: dict[int, FloatingNDArray],
        direction_mappings: list[IntegerNDArray],
        max_angles_per_dim: IntegerNDArray,
        parameters: ParamUDCT,
    ) -> tuple[list[tuple[IntpNDArray, FloatingNDArray]], IntegerNDArray]:
        """
        Process a single window_index value completely independently.

        This method processes one window_index, building the window, generating
        flipped versions for symmetry, and converting to sparse format. Each
        call is completely independent with no shared state.

        Parameters
        ----------
        scale_idx : int
            Scale index (1-based).
        dimension_idx : int
            Dimension index (0-based).
        window_index : int
            Window index to process.
        angle_indices_1d : IntegerNDArray
            Pre-computed angle index combinations.
        angle_functions : dict[int, dict[tuple[int, int], FloatingNDArray]]
            Dictionary of angle functions by scale and dimension.
        angle_indices : dict[int, dict[tuple[int, int], IntegerNDArray]]
            Dictionary of angle indices by scale and dimension.
        bandpass_windows : dict[int, FloatingNDArray]
            Dictionary of bandpass windows by scale.
        direction_mappings : list[IntegerNDArray]
            Direction mappings for each resolution.
        max_angles_per_dim : IntegerNDArray
            Maximum angles per dimension.
        parameters : ParamUDCT
            UDCT parameters.

        Returns
        -------
        tuple[list[tuple[IntpNDArray, FloatingNDArray]], IntegerNDArray]]
            Tuple containing:
            - List of window tuples (indices, values) for this window_index
              (including original and flipped versions)
            - angle_indices_2d array for this window_index

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._utils import ParamUDCT
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> # This would be called internally by compute()
        """
        # Build window for the given window_index
        window: FloatingNDArray = np.ones(parameters.size, dtype=float)
        for angle_dim_idx in range(parameters.dim - 1):
            angle_idx = angle_indices_1d.reshape(len(angle_indices_1d), -1)[
                window_index, angle_dim_idx
            ]
            angle_func = angle_functions[scale_idx - 1][(dimension_idx, angle_dim_idx)][
                angle_idx
            ]
            angle_idx_mapping = angle_indices[scale_idx - 1][
                (dimension_idx, angle_dim_idx)
            ]
            kron_angle = UDCTWindow._compute_angle_kronecker_product(
                angle_func, angle_idx_mapping, parameters
            )
            window *= kron_angle
        window *= bandpass_windows[scale_idx]
        window = np.sqrt(circshift(window, tuple(s // 4 for s in parameters.size)))

        window_functions = []
        window_functions.append(window)

        angle_indices_2d = angle_indices_1d[window_index : window_index + 1, :] + 1

        # Generate flipped window versions for symmetry
        # For each dimension, if the angle index is in the first half,
        # create a flipped version by reflecting across the midpoint
        for flip_dimension_index in range(parameters.dim - 2, -1, -1):
            for function_index in range(angle_indices_2d.shape[0]):
                if (
                    2 * angle_indices_2d[function_index, flip_dimension_index]
                    <= max_angles_per_dim[flip_dimension_index]
                ):
                    # Compute reflected angle index
                    flipped_angle_indices = angle_indices_2d[
                        function_index : function_index + 1, :
                    ].copy()
                    flipped_angle_indices[0, flip_dimension_index] = (
                        max_angles_per_dim[flip_dimension_index]
                        + 1
                        - angle_indices_2d[function_index, flip_dimension_index]
                    )
                    angle_indices_2d = np.r_[angle_indices_2d, flipped_angle_indices]
                    # Flip the window function along the appropriate axis
                    flip_axis_dimension = int(
                        direction_mappings[scale_idx - 1][
                            dimension_idx, flip_dimension_index
                        ]
                    )
                    window = UDCTWindow._flip_with_fft_shift(
                        window_functions[function_index],
                        flip_axis_dimension,
                    )
                    window_functions.append(window)
        angle_indices_2d -= 1
        window_functions = np.c_[window_functions]

        # Convert all window functions to sparse format
        window_tuples = []
        for function_index in range(window_functions.shape[0]):
            window_tuples.append(
                UDCTWindow._to_sparse(
                    window_functions[function_index],
                    parameters.window_threshold,
                )
            )

        return window_tuples, angle_indices_2d

    @staticmethod
    def compute(
        parameters: ParamUDCT,
    ) -> tuple[UDCTWindows, list[IntegerNDArray], dict[int, dict[int, IntegerNDArray]]]:
        """
        Compute curvelet windows in frequency domain for UDCT transform.

        This method generates the frequency-domain windows used in the Uniform
        Discrete Curvelet Transform (UDCT). It creates bandpass filters using
        Meyer wavelets for radial frequency decomposition and angular wedges for
        directional selectivity. The windows are stored in sparse format to
        optimize memory usage.

        Parameters
        ----------
        parameters : ParamUDCT
            UDCT parameters containing:
            - res : int
                Number of resolution scales
            - size : tuple[int, ...]
                Shape of the input data
            - dim : int
                Dimensionality of the transform
            - radial_frequency_params : tuple[float, float, float, float]
                Parameters defining radial frequency bands
            - angular_wedges_config : IntegerNDArray
                Configuration array specifying number of angular wedges per scale
                and dimension, shape (num_scales, dimension)
            - window_overlap : float
                Window overlap parameter controlling smoothness of transitions
            - window_threshold : float
                Threshold for sparse window storage

        Returns
        -------
        windows : UDCTWindows
            Curvelet windows in sparse format. Structure is:
            windows[scale][direction][wedge] = (indices, values) tuple
            where scale 0 is low-frequency, scales 1..res are high-frequency bands
        decimation_ratios : list[IntegerNDArray]
            Decimation ratios for each scale and direction. Structure:
            - decimation_ratios[0]: shape (1, dim) for low-frequency band
            - decimation_ratios[scale]: shape (dim, dim) for scale > 0
        indices : dict[int, dict[int, IntegerNDArray]]
            Angular indices for each window. Structure:
            indices[scale][direction] = array of shape (num_wedges, dim-1)
            containing angular indices for each wedge

        Notes
        -----
        The window computation process involves several steps:

        1. **Radial frequency decomposition**: Creates bandpass filters using
           Meyer wavelets to decompose the frequency domain into radial bands
           corresponding to different scales.

        2. **Angular wedge construction**: For each scale and direction, creates
           angular wedges that provide directional selectivity. The number of
           wedges is determined by `angular_wedges_config`.

        3. **Window generation**: Combines radial bandpass filters with angular
           wedges to create the final curvelet windows. Each window is shifted
           by size//4 in each dimension to center it in frequency space.

        4. **Sparse storage**: Windows are converted to sparse format using
           `window_threshold` to reduce memory usage. Only values above the
           threshold are stored.

        5. **Normalization**: Windows are normalized so that the sum of squares
           of all windows equals 1 at each frequency point, ensuring perfect
           reconstruction.

        6. **Sorting**: Windows are sorted by their angular indices for
           consistent ordering.

        The windows are designed to provide a tight frame, meaning they form
        a complete representation that allows perfect reconstruction of the
        original signal.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._utils import ParamUDCT
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>>
        >>> # Create parameters for 2D transform with 3 scales
        >>> params = ParamUDCT(
        ...     size=(64, 64),
        ...     res=3,
        ...     dim=2,
        ...     angular_wedges_config=np.array([[3], [6], [12]]),
        ...     window_overlap=0.15,
        ...     window_threshold=1e-5,
        ...     radial_frequency_params=(np.pi/3, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3)
        ... )
        >>>
        >>> # Compute windows
        >>> windows, decimation_ratios, indices = UDCTWindow.compute(params)
        >>>
        >>> # Check structure
        >>> len(windows)  # Number of scales (0 + res)
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
            num_scales=parameters.res,
            shape=parameters.size,
            radial_frequency_params=parameters.radial_frequency_params,
        )
        low_frequency_window = circshift(
            np.sqrt(bandpass_windows[0]), tuple(s // 4 for s in parameters.size)
        )

        # convert to sparse format
        windows: UDCTWindows = []
        windows.append([])
        windows[0].append([])
        windows[0][0] = [
            UDCTWindow._to_sparse(low_frequency_window, parameters.window_threshold)
        ]

        indices: dict[int, dict[int, IntegerNDArray]] = {}
        indices[0] = {}
        indices[0][0] = np.zeros((1, 1), dtype=int)
        direction_mappings = UDCTWindow._create_direction_mappings(
            dimension=parameters.dim, num_resolutions=parameters.res
        )
        angle_functions, angle_indices = UDCTWindow._create_angle_info(
            frequency_grid,
            dimension=parameters.dim,
            num_resolutions=parameters.res,
            angular_wedges_config=parameters.angular_wedges_config,
            window_overlap=parameters.window_overlap,
        )

        decimation_ratios = UDCTWindow._calculate_decimation_ratios_with_lowest(
            num_resolutions=parameters.res,
            dimension=parameters.dim,
            angular_wedges_config=parameters.angular_wedges_config,
            direction_mappings=direction_mappings,
        )

        # Generate windows for each high-frequency scale
        # For each scale, direction, and wedge combination:
        #   1. Build window by combining angle functions with bandpass filter
        #   2. Generate flipped versions for symmetry
        #   3. Convert to sparse format and store
        for scale_idx in range(1, parameters.res + 1):
            windows.append([])
            indices[scale_idx] = {}

            for dimension_idx in range(parameters.dim):
                # Build angle_indices_1d once per (scale_idx, dimension_idx)
                angle_indices_1d = UDCTWindow._build_angle_indices_1d(
                    scale_idx=scale_idx,
                    dimension_idx=dimension_idx,
                    angle_functions=angle_functions,
                    parameters=parameters,
                )
                num_windows = angle_indices_1d.shape[0]
                max_angles_per_dim = parameters.angular_wedges_config[
                    scale_idx - 1, direction_mappings[scale_idx - 1][dimension_idx, :]
                ]

                # Process each window_index independently
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
                        parameters=parameters,
                    )
                    for window_index in range(num_windows)
                ]

                # Flatten windows and concatenate angle indices
                all_windows = [
                    window
                    for window_list, _ in window_results
                    for window in window_list
                ]
                all_angle_indices = [angle_idx for _, angle_idx in window_results]
                angle_index_array = (
                    np.concatenate(all_angle_indices, axis=0)
                    if all_angle_indices
                    else np.zeros((0, parameters.dim - 1), dtype=int)
                )

                windows[scale_idx].append(all_windows)
                indices[scale_idx][dimension_idx] = angle_index_array

        UDCTWindow._inplace_normalize_windows(
            windows,
            size=parameters.size,
            dimension=parameters.dim,
            num_resolutions=parameters.res,
        )

        UDCTWindow._inplace_sort_windows(
            windows=windows,
            indices=indices,
            num_resolutions=parameters.res,
            dimension=parameters.dim,
        )

        return windows, decimation_ratios, indices
