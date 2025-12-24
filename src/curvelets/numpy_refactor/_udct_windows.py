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
           Transform": IEEE Transactions on Signal Processing, 58, 3618–3634.
           DOI: 10.1109/TSP.2010.2047666

        Parameters
        ----------
        scale_idx : int
            High-frequency scale index, ranging from 1 to res (inclusive).
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
        angle_functions : dict[int, dict[tuple[int, int], FloatingNDArray]]
            Dictionary of angle functions A_{j,l} by scale and dimension (Section IV).
        angle_indices : dict[int, dict[tuple[int, int], IntegerNDArray]]
            Dictionary of angle indices by scale and dimension.
        bandpass_windows : dict[int, FloatingNDArray]
            Dictionary of Meyer wavelet-based bandpass filters F_j by scale (Section IV).
        direction_mappings : list[IntegerNDArray]
            Direction mappings for each resolution, used to determine flip axes.
        max_angles_per_dim : IntegerNDArray
            Maximum angles per dimension, used to determine which indices need flipping.
        parameters : ParamUDCT
            UDCT parameters containing transform configuration.

        Returns
        -------
        tuple[list[tuple[IntpNDArray, FloatingNDArray]], IntegerNDArray]
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

        Examples
        --------
        This method is typically called internally by :py:meth:`UDCTWindow.compute`.
        For a given window_index, it processes one set of angle indices and generates
        all symmetric flipped versions:

        >>> import numpy as np
        >>> from curvelets.numpy_refactor._utils import ParamUDCT
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>>
        >>> # Example output structure:
        >>> # For a 2D transform (dim=2), angle_indices_2d has shape (num_windows, 1)
        >>> # since dim-1 = 1. The first row is the original, subsequent rows
        >>> # are flipped versions for symmetry.
        >>> #
        >>> # Example: if processing window_index=0 with angle_idx=[1], it might generate:
        >>> angle_indices_2d = np.array([[1], [2]])  # original and one flipped version
        >>> angle_indices_2d.shape
        (2, 1)
        >>>
        >>> # window_tuples is a list of (indices, values) tuples in sparse format:
        >>> # window_tuples = [
        >>> #     (np.array([10, 20, 30]), np.array([0.5, 0.8, 0.3])),  # first window
        >>> #     (np.array([15, 25, 35]), np.array([0.4, 0.7, 0.2]))   # flipped window
        >>> # ]
        >>> # Each tuple represents a sparse window: (non-zero indices, non-zero values)
        """
        # Step 1: Build base window W_{j,l} = F_j · A_{j,l} (Section IV, Nguyen & Chauris 2010)
        # Initialize with unity: W = 1
        window: FloatingNDArray = np.ones(parameters.size, dtype=float)

        # Multiply by angular functions A_{j,l} for each angular dimension
        # These provide directional selectivity via Kronecker products
        for angle_dim_idx in range(parameters.dim - 1):
            angle_idx = angle_indices_1d.reshape(len(angle_indices_1d), -1)[
                window_index, angle_dim_idx
            ]
            # Get 1D angle function for this dimension and angle index
            # Note: scale_idx - 1 because angle_functions is 0-indexed for high-frequency scales
            # (scale 0 is lowpass, handled separately; scales 1..res map to indices 0..res-1)
            angle_func = angle_functions[scale_idx - 1][(dimension_idx, angle_dim_idx)][
                angle_idx
            ]
            angle_idx_mapping = angle_indices[scale_idx - 1][
                (dimension_idx, angle_dim_idx)
            ]
            # Expand 1D angle function to N-D using Kronecker products
            # This creates the multi-dimensional angular wedge A_{j,l}
            kron_angle = UDCTWindow._compute_angle_kronecker_product(
                angle_func, angle_idx_mapping, parameters
            )
            window *= kron_angle

        # Multiply by bandpass filter F_j (Meyer wavelet-based, Section IV)
        # This provides scale selectivity
        # Note: bandpass_windows[scale_idx] directly because bandpass_windows[0] is lowpass,
        # and bandpass_windows[1..res] correspond to high-frequency scales 1..res
        window *= bandpass_windows[scale_idx]

        # Apply frequency shift (size//4 in each dimension) and square root
        # The shift centers the window in frequency space, and square root
        # ensures proper normalization for the partition of unity condition
        window = np.sqrt(circshift(window, tuple(s // 4 for s in parameters.size)))

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
        for flip_dimension_index in range(parameters.dim - 2, -1, -1):
            angle_indices_list = add_flipped_indices(
                angle_indices_list, flip_dimension_index
            )

        # Create arrays once with final size (pre-allocate for efficiency)
        num_windows = len(angle_indices_list)
        angle_indices_2d = np.array(angle_indices_list)
        window_functions = np.zeros((num_windows, *parameters.size), dtype=window.dtype)

        # Second pass: fill in windows, applying flips as needed
        # Build mapping from angle index to its source window and flip dimension
        # This allows us to efficiently generate flipped windows from their sources
        angle_to_source: dict[tuple[int, ...], tuple[int, int]] = {
            tuple(angle_indices_2d[0]): (0, -1)  # Original has no source
        }
        # Build the complete mapping by iterating through dimensions
        for flip_dimension_index in range(parameters.dim - 2, -1, -1):
            for source_idx_tuple, (source_window_idx, _) in list(
                angle_to_source.items()
            ):
                source_angle_idx = np.array(source_idx_tuple)
                if needs_flipping(source_angle_idx, flip_dimension_index):
                    flipped_angle_idx = flip_angle_idx(
                        source_angle_idx, flip_dimension_index
                    )
                    flipped_angle_idx_tuple = tuple(flipped_angle_idx)
                    if flipped_angle_idx_tuple not in angle_to_source:
                        angle_to_source[flipped_angle_idx_tuple] = (
                            source_window_idx,
                            flip_dimension_index,
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
            # (scale 0 is lowpass, handled separately; scales 1..res map to indices 0..res-1)
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
                parameters.window_threshold,
            )
            for function_index in range(window_functions.shape[0])
        ]

        return window_tuples, angle_indices_2d

    @staticmethod
    def compute(
        parameters: ParamUDCT,
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
           Transform": IEEE Transactions on Signal Processing, 58, 3618–3634.
           DOI: 10.1109/TSP.2010.2047666

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
        **Window Construction (Section IV, Nguyen & Chauris 2010)**:
        Constructs windows :math:`W_{j,l} = F_j \cdot A_{j,l}` where :math:`F_j` are
        Meyer wavelet-based bandpass filters (via :py:meth:`UDCTWindow._create_bandpass_windows`)
        and :math:`A_{j,l}` are angular functions (via :py:meth:`UDCTWindow._create_angle_info`).
        Low-frequency window (scale 0) is handled separately; high-frequency windows
        (scales 1..res) are generated via :py:meth:`UDCTWindow._process_single_window`.

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

        # Generate windows for each high-frequency scale (scales 1 to res)
        # Each scale contains multiple directions (one per dimension), and each
        # direction contains multiple wedges (windows with different angular orientations)
        for scale_idx in range(1, parameters.res + 1):
            windows.append([])
            indices[scale_idx] = {}

            # Process each direction (dimension) independently
            for dimension_idx in range(parameters.dim):
                # Build angle index combinations once per (scale_idx, dimension_idx)
                # This determines which angular wedges will be created
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
                        parameters=parameters,
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
                    else np.zeros((0, parameters.dim - 1), dtype=int)
                )

                # Store results for this (scale_idx, dimension_idx) combination
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
