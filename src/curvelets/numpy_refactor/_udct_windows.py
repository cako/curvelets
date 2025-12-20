from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from math import ceil
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .typing import UDCTWindows
from .utils import ParamUDCT, circshift, fun_meyer

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
    >>> from curvelets.numpy_refactor.utils import ParamUDCT
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
        x_primary: np.ndarray, x_secondary: np.ndarray
    ) -> np.ndarray:
        """
        Compute one angle component from meshgrid coordinates.

        Parameters
        ----------
        x_primary : np.ndarray
            Primary coordinate grid (used for conditions).
        x_secondary : np.ndarray
            Secondary coordinate grid.

        Returns
        -------
        np.ndarray
            Angle component array.
        """
        t1: npt.NDArray[np.floating] = np.zeros_like(x_primary, dtype=float)
        ind = (x_primary != 0) & (np.abs(x_secondary) <= np.abs(x_primary))
        t1[ind] = -x_secondary[ind] / x_primary[ind]

        t2: npt.NDArray[np.floating] = np.zeros_like(x_primary, dtype=float)
        ind = (x_secondary != 0) & (np.abs(x_primary) < np.abs(x_secondary))
        t2[ind] = x_primary[ind] / x_secondary[ind]

        t3 = t2.copy()
        t3[t2 < 0] = t2[t2 < 0] + 2
        t3[t2 > 0] = t2[t2 > 0] - 2

        result = t1 + t3
        result[x_primary >= 0] = -2
        return result

    @staticmethod
    def _adapt_grid(S1: np.ndarray, S2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Adapt frequency grids for angle computation.

        Parameters
        ----------
        S1 : np.ndarray
            First frequency grid.
        S2 : np.ndarray
            Second frequency grid.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Adapted grid arrays M2 and M1.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> S1 = np.linspace(-1.5*np.pi, 0.5*np.pi, 64, endpoint=False)
        >>> S2 = np.linspace(-1.5*np.pi, 0.5*np.pi, 64, endpoint=False)
        >>> M2, M1 = UDCTWindow._adapt_grid(S1, S2)
        >>> M2.shape
        (64, 64)
        """
        x1, x2 = np.meshgrid(S2, S1)

        # Compute M1 using x1 as primary, x2 as secondary
        M1 = UDCTWindow._compute_angle_component(x1, x2)

        # Compute M2 using x2 as primary, x1 as secondary (swapped)
        M2 = UDCTWindow._compute_angle_component(x2, x1)

        return M2, M1

    @staticmethod
    def _angle_fun(
        Mgrid: np.ndarray, direction: int, n: int, window_overlap: float
    ) -> np.ndarray:
        """
        Create angle functions using Meyer windows.

        Parameters
        ----------
        Mgrid : np.ndarray
            Angle grid.
        direction : int
            Direction index (1 or 2).
        n : int
            Number of angular wedges.
        window_overlap : float
            Window overlap parameter.

        Returns
        -------
        np.ndarray
            Array of angle functions.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> Mgrid = np.linspace(-2, 2, 64)
        >>> angle_funcs = UDCTWindow._angle_fun(Mgrid, 1, 3, 0.15)
        >>> angle_funcs.shape[0]
        2
        """
        # Compute angular window spacing and boundaries
        angd = 2 / n
        ang = angd * np.array(
            [-window_overlap, window_overlap, 1 - window_overlap, 1 + window_overlap]
        )

        # Generate angle functions using Meyer windows
        # Note: Both direction 1 and 2 use the same computation because the
        # angle function is symmetric with respect to direction. The direction
        # parameter is kept for API consistency and potential future extensions.
        Mang = []
        if direction in (1, 2):
            for jn in range(1, ceil(n / 2) + 1):
                ang2 = -1 + (jn - 1) * angd + ang
                fang = fun_meyer(Mgrid, *ang2)
                Mang.append(fang[None, :])
        else:
            error_msg = f"Unrecognized direction: {direction}. Must be 1 or 2."
            raise ValueError(error_msg)
        return np.concatenate(Mang, axis=0)

    @staticmethod
    def _angle_kron(
        angle_arr: np.ndarray, nper: np.ndarray, param_udct: ParamUDCT
    ) -> np.ndarray:
        """
        Compute Kronecker product for angle functions.

        Parameters
        ----------
        angle_arr : np.ndarray
            Angle function array.
        nper : np.ndarray
            Dimension permutation indices.
        param_udct : ParamUDCT
            UDCT parameters.

        Returns
        -------
        np.ndarray
            Kronecker product result with shape matching param_udct.size.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor.utils import ParamUDCT
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> params = ParamUDCT(
        ...     size=(64, 64), res=3, dim=2,
        ...     angular_wedges_config=np.array([[3], [6], [12]]),
        ...     window_overlap=0.15, window_threshold=1e-5,
        ...     radial_frequency_params=(np.pi/3, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3)
        ... )
        >>> angle_arr = np.ones(64)
        >>> nper = np.array([1, 2])
        >>> result = UDCTWindow._angle_kron(angle_arr, nper, params)
        >>> result.shape
        (64, 64)
        """
        # Pre-compute dimension sizes for Kronecker product
        krsz: npt.NDArray[np.int_] = np.array(
            [
                np.prod(param_udct.size[: nper[0] - 1]),
                np.prod(param_udct.size[nper[0] : nper[1] - 1]),
                np.prod(param_udct.size[nper[1] : param_udct.dim]),
            ],
            dtype=int,
        )

        # Optimize: cache ravel() result to avoid recomputation
        tmp1 = np.kron(np.ones((krsz[1], 1), dtype=int), angle_arr)
        tmp1_flat = tmp1.ravel()
        tmp2 = np.kron(np.ones((krsz[2], 1), dtype=int), tmp1_flat).ravel()
        tmp3 = np.kron(tmp2, np.ones((krsz[0], 1), dtype=int)).ravel()
        return tmp3.reshape(*param_udct.size)

    @staticmethod
    def _fftflip(F: np.ndarray, axis: int) -> np.ndarray:
        """
        Flip array along specified axis with frequency domain shift.

        Parameters
        ----------
        F : np.ndarray
            Input array.
        axis : int
            Axis along which to flip.

        Returns
        -------
        np.ndarray
            Flipped and shifted array.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> flipped = UDCTWindow._fftflip(arr, 0)
        >>> flipped.shape
        (2, 2)
        """
        dim = F.ndim
        shiftvec: npt.NDArray[np.int_] = np.zeros((dim,), dtype=int)
        shiftvec[axis] = 1
        Fc = np.flip(F, axis)
        return circshift(Fc, tuple(shiftvec))

    @staticmethod
    def _to_sparse(
        arr: npt.NDArray[D_T], thresh: float
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[D_T]]:
        """
        Convert array to sparse format.

        Parameters
        ----------
        arr : npt.NDArray[D_T]
            Input array.
        thresh : float
            Threshold for sparse storage (values above threshold are kept).

        Returns
        -------
        tuple[npt.NDArray[np.intp], npt.NDArray[D_T]]
            Tuple of (indices, values) where indices are positions and values
            are the array values at those positions.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> arr = np.array([0.1, 0.5, 0.01, 0.8])
        >>> idx, vals = UDCTWindow._to_sparse(arr, 0.2)
        >>> len(vals)
        2
        """
        arr_flat = arr.ravel()
        idx = np.argwhere(arr_flat > thresh)
        return (idx, arr_flat[idx])

    @staticmethod
    def _nchoosek(
        n: Iterable[int] | npt.NDArray[np.int_], k: int
    ) -> npt.NDArray[np.int_]:
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
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
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
        frequency_grid : dict[int, np.ndarray]
            Dictionary mapping dimension index to frequency grid array.
            Each grid spans [-1.5*pi, 0.5*pi) with size matching shape[dimension].
        bandpass_windows : dict[int, np.ndarray]
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
        frequency_grid: dict[int, np.ndarray] = {}
        meyer_windows: dict[tuple[int, int], np.ndarray] = {}
        for dimension_idx in range(dimension):
            # Don't take the np.pi out of the linspace
            frequency_grid[dimension_idx] = np.linspace(
                -1.5 * np.pi, 0.5 * np.pi, shape[dimension_idx], endpoint=False
            )

            params = np.array([-2, -1, *radial_frequency_params[:2]])
            abs_frequency_grid = np.abs(frequency_grid[dimension_idx])
            meyer_windows[(num_scales, dimension_idx)] = fun_meyer(
                abs_frequency_grid, *params
            )
            if num_scales == 1:
                meyer_windows[(num_scales, dimension_idx)] += fun_meyer(
                    np.abs(frequency_grid[dimension_idx] + 2 * np.pi), *params
                )
            params[2:] = radial_frequency_params[2:]
            meyer_windows[(num_scales + 1, dimension_idx)] = fun_meyer(
                abs_frequency_grid, *params
            )

            for scale_idx in range(num_scales - 1, 0, -1):
                params[2:] = radial_frequency_params[:2]
                params[2:] /= 2 ** (num_scales - scale_idx)
                meyer_windows[(scale_idx, dimension_idx)] = fun_meyer(
                    abs_frequency_grid, *params
                )

        bandpass_windows: dict[int, np.ndarray] = {}
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
    def _create_mdirs(dimension: int, num_resolutions: int) -> list[np.ndarray]:
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
        list[np.ndarray]
            List of arrays, one per resolution. Each array has shape (dimension, dimension-1)
            and contains indices of dimensions used for angle calculations on each hyperpyramid.

            Examples
            --------
            >>> import numpy as np
            >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
            >>> mappings = UDCTWindow._create_mdirs(2, 3)
            >>> len(mappings)
            3
            >>> mappings[0].shape
            (2, 1)
        """
        # Mdir is dimension of need to calculate angle function on each
        # hyperpyramid
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
        frequency_grid: dict[int, np.ndarray],
        dimension: int,
        num_resolutions: int,
        angular_wedges_config: np.ndarray,
        window_overlap: float,
    ) -> tuple[
        dict[int, dict[tuple[int, int], np.ndarray]],
        dict[int, dict[tuple[int, int], np.ndarray]],
    ]:
        """
        Create angle functions and indices for window computation.

        Parameters
        ----------
        frequency_grid : dict[int, np.ndarray]
            Dictionary mapping dimension index to frequency grid.
        dimension : int
            Dimensionality of the transform.
        num_resolutions : int
            Number of resolution scales.
        angular_wedges_config : np.ndarray
            Configuration array specifying number of angular wedges per scale and dimension.
        window_overlap : float
            Window overlap parameter.

        Returns
        -------
        tuple[dict, dict]
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
        # every combination of 2 dimension out of 1:dimension
        dimension_permutations = UDCTWindow._nchoosek(np.arange(dimension), 2)
        angle_grid: dict[tuple[int, int], np.ndarray] = {}
        for perm_idx, perm in enumerate(dimension_permutations):
            out = UDCTWindow._adapt_grid(
                frequency_grid[perm[0]], frequency_grid[perm[1]]
            )
            angle_grid[(perm_idx, 0)] = out[0]
            angle_grid[(perm_idx, 1)] = out[1]

        # gather angle function for each pyramid
        angle_functions: dict[int, dict[tuple[int, int], np.ndarray]] = {}
        angle_indices: dict[int, dict[tuple[int, int], np.ndarray]] = {}
        for scale_idx in range(num_resolutions):
            angle_functions[scale_idx] = {}
            angle_indices[scale_idx] = {}
            # for each resolution
            for dimension_idx in range(dimension):
                # for each pyramid in resolution res
                angle_count = 0
                # angle_count is number of angle function required for each pyramid
                # now loop through dimension_permutations
                for hyperpyramid_idx in range(dimension_permutations.shape[0]):
                    for direction_idx in range(dimension_permutations.shape[1]):
                        if (
                            dimension_permutations[hyperpyramid_idx, direction_idx]
                            == dimension_idx
                        ):
                            angle_functions[scale_idx][(dimension_idx, angle_count)] = (
                                UDCTWindow._angle_fun(
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
                            )
                            angle_indices[scale_idx][(dimension_idx, angle_count)] = (
                                dimension_permutations[hyperpyramid_idx, :] + 1
                            )
                            angle_count += 1
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
        >>> from curvelets.numpy_refactor.typing import UDCTWindows
        >>> windows: UDCTWindows = [[[(np.array([[0]]), np.array([1.0]))]]]
        >>> UDCTWindow._inplace_normalize_windows(windows, (64, 64), 2, 3)
        """
        sum_squared_windows = np.zeros(size)
        idx, val = windows[0][0][0]
        idx_flat = idx.ravel()
        val_flat = val.ravel()
        sum_squared_windows.flat[idx_flat] += val_flat**2
        for scale_idx in range(1, num_resolutions + 1):
            for direction_idx in range(dimension):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    idx, val = windows[scale_idx][direction_idx][wedge_idx]
                    idx_flat = idx.ravel()
                    val_flat = val.ravel()
                    # Accumulate directly from sparse format (no temp array needed)
                    sum_squared_windows.flat[idx_flat] += val_flat**2
                    # For flipped version, still need temp array but create it only once
                    temp_window = np.zeros(size)
                    temp_window.flat[idx_flat] = val_flat**2
                    temp_window = UDCTWindow._fftflip(temp_window, direction_idx)
                    sum_squared_windows += temp_window

        sum_squared_windows = np.sqrt(sum_squared_windows)
        sum_squared_windows_flat = sum_squared_windows.ravel()
        idx, val = windows[0][0][0]
        idx_flat = idx.ravel()
        val_flat = val.ravel()
        val_flat[:] /= sum_squared_windows_flat[idx_flat]
        for scale_idx in range(1, num_resolutions + 1):
            for direction_idx in range(dimension):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    idx, val = windows[scale_idx][direction_idx][wedge_idx]
                    idx_flat = idx.ravel()
                    val_flat = val.ravel()
                    val_flat[:] /= sum_squared_windows_flat[idx_flat]

    @staticmethod
    def _calculate_decimation_ratios_with_lowest(
        num_resolutions: int,
        dimension: int,
        angular_wedges_config: np.ndarray,
        direction_mappings: list[np.ndarray],
    ) -> list[npt.NDArray[np.int_]]:
        """
        Calculate decimation ratios for each scale and direction.

        Parameters
        ----------
        num_resolutions : int
            Number of resolution scales.
        dimension : int
            Dimensionality of the transform.
        angular_wedges_config : np.ndarray
            Configuration array specifying number of angular wedges.
        direction_mappings : list[np.ndarray]
            Direction mappings for each resolution.

        Returns
        -------
        list[npt.NDArray[np.int_]]
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
        decimation_ratios: list[npt.NDArray[np.int_]] = [
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
        indices: dict[int, dict[int, np.ndarray]],
        num_resolutions: int,
        dimension: int,
    ) -> None:
        """
        Sort windows in-place by their angular indices.

        Parameters
        ----------
        windows : UDCTWindows
            Windows to sort (modified in-place).
        indices : dict[int, dict[int, np.ndarray]]
            Angular indices dictionary (modified in-place).
        num_resolutions : int
            Number of resolution scales.
        dimension : int
            Dimensionality of the transform.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor._udct_windows import UDCTWindow
        >>> from curvelets.numpy_refactor.typing import UDCTWindows
        >>> windows: UDCTWindows = [[[(np.array([[0]]), np.array([1.0]))]]]
        >>> indices = {1: {0: np.array([[0]])}}
        >>> UDCTWindow._inplace_sort_windows(windows, indices, 3, 2)
        """
        for scale_idx in range(1, num_resolutions + 1):
            for dimension_idx in range(dimension):
                index_list = indices[scale_idx][dimension_idx]

                max_val = index_list.max() + 1
                sort_indices = np.argsort(
                    sum(
                        max_val**power * index_list[:, col_idx]
                        for col_idx, power in enumerate(
                            range(index_list.shape[1] - 1, -1, -1)
                        )
                    )
                )

                indices[scale_idx][dimension_idx] = index_list[sort_indices]
                windows[scale_idx][dimension_idx] = [
                    windows[scale_idx][dimension_idx][idx] for idx in sort_indices
                ]

    @staticmethod
    def compute(
        parameters: ParamUDCT,
    ) -> tuple[
        UDCTWindows, list[npt.NDArray[np.int_]], dict[int, dict[int, np.ndarray]]
    ]:
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
            - angular_wedges_config : np.ndarray
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
        decimation_ratios : list[npt.NDArray[np.int_]]
            Decimation ratios for each scale and direction. Structure:
            - decimation_ratios[0]: shape (1, dim) for low-frequency band
            - decimation_ratios[scale]: shape (dim, dim) for scale > 0
        indices : dict[int, dict[int, np.ndarray]]
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
        >>> from curvelets.numpy_refactor.utils import ParamUDCT
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

        # `indices` gets stored as `parameters.ind` in the original.
        indices: dict[int, dict[int, np.ndarray]] = {}
        indices[0] = {}
        indices[0][0] = np.zeros((1, 1), dtype=int)
        direction_mappings = UDCTWindow._create_mdirs(
            dimension=parameters.dim, num_resolutions=parameters.res
        )
        angle_functions, angle_indices = UDCTWindow._create_angle_info(
            frequency_grid,
            dimension=parameters.dim,
            num_resolutions=parameters.res,
            angular_wedges_config=parameters.angular_wedges_config,
            window_overlap=parameters.window_overlap,
        )

        # decimation ratio for each band
        decimation_ratios = UDCTWindow._calculate_decimation_ratios_with_lowest(
            num_resolutions=parameters.res,
            dimension=parameters.dim,
            angular_wedges_config=parameters.angular_wedges_config,
            direction_mappings=direction_mappings,
        )

        # angle_functions is 1-d angle function for each hyper pyramid (row) and each angle
        # dimension (column)
        for scale_idx in range(1, parameters.res + 1):  # pylint: disable=too-many-nested-blocks
            # for each resolution
            windows.append([])
            indices[scale_idx] = {}
            for dimension_idx in range(parameters.dim):
                windows[scale_idx].append([])
                # for each hyperpyramid
                angle_indices_1d = np.arange(
                    len(angle_functions[scale_idx - 1][(dimension_idx, 0)])
                )[:, None]
                for angle_dim_idx in range(1, parameters.dim - 1):
                    num_angles = len(
                        angle_functions[scale_idx - 1][(dimension_idx, angle_dim_idx)]
                    )
                    angle_indices_2d = np.arange(
                        len(
                            angle_functions[scale_idx - 1][
                                (dimension_idx, angle_dim_idx)
                            ]
                        )
                    )[:, None]
                    kron_1 = np.kron(
                        angle_indices_1d, np.ones((num_angles, 1), dtype=int)
                    )
                    kron_2 = np.kron(
                        np.ones((angle_indices_1d.shape[0], 1), dtype=int),
                        angle_indices_2d,
                    )
                    angle_indices_1d = np.c_[kron_1, kron_2]
                num_windows = angle_indices_1d.shape[0]
                max_angles_per_dim = parameters.angular_wedges_config[
                    scale_idx - 1, direction_mappings[scale_idx - 1][dimension_idx, :]
                ]
                # num_windows is the smallest number of windows need to calculated on each
                # pyramid
                # max_angles_per_dim is M-1 vector contain number of angle function per each
                # dimension of the hyperpyramid
                for window_idx in range(num_windows):
                    # for each calculated windows function, estimated all the other
                    # flipped window functions
                    window: npt.NDArray[np.floating] = np.ones(
                        parameters.size, dtype=float
                    )
                    for angle_dim_idx in range(parameters.dim - 1):
                        angle_idx = angle_indices_1d.reshape(len(angle_indices_1d), -1)[
                            window_idx, angle_dim_idx
                        ]
                        angle_func = angle_functions[scale_idx - 1][
                            (dimension_idx, angle_dim_idx)
                        ][angle_idx]
                        angle_idx_mapping = angle_indices[scale_idx - 1][
                            (dimension_idx, angle_dim_idx)
                        ]
                        kron_angle = UDCTWindow._angle_kron(
                            angle_func, angle_idx_mapping, parameters
                        )
                        window *= kron_angle
                    window *= bandpass_windows[scale_idx]
                    window = np.sqrt(
                        circshift(window, tuple(s // 4 for s in parameters.size))
                    )

                    # first windows function
                    window_functions = []
                    window_functions.append(window)

                    # index of current angle
                    angle_indices_2d = (
                        angle_indices_1d[window_idx : window_idx + 1, :] + 1
                    )

                    # all possible flip along different dimension
                    for flip_dim_idx in range(parameters.dim - 2, -1, -1):
                        for func_idx in range(angle_indices_2d.shape[0]):
                            if (
                                2 * angle_indices_2d[func_idx, flip_dim_idx]
                                <= max_angles_per_dim[flip_dim_idx]
                            ):
                                angle_indices_tmp = angle_indices_2d[
                                    func_idx : func_idx + 1, :
                                ].copy()
                                angle_indices_tmp[0, flip_dim_idx] = (
                                    max_angles_per_dim[flip_dim_idx]
                                    + 1
                                    - angle_indices_2d[func_idx, flip_dim_idx]
                                )
                                angle_indices_2d = np.r_[
                                    angle_indices_2d, angle_indices_tmp
                                ]
                                flip_axis = int(
                                    direction_mappings[scale_idx - 1][
                                        dimension_idx, flip_dim_idx
                                    ]
                                )
                                window = UDCTWindow._fftflip(
                                    window_functions[func_idx], flip_axis
                                )
                                window_functions.append(window)
                    angle_indices_2d -= 1  # Adjust so that `indices` is 0-based
                    window_functions = np.c_[window_functions]

                    if window_idx == 0:
                        angle_index_array = angle_indices_2d
                        for func_idx in range(angle_index_array.shape[0]):
                            windows[scale_idx][dimension_idx].append(
                                UDCTWindow._to_sparse(
                                    window_functions[func_idx],
                                    parameters.window_threshold,
                                )
                            )
                    else:
                        old_size = angle_index_array.shape[0]
                        angle_index_array = np.concatenate(
                            (angle_index_array, angle_indices_2d), axis=0
                        )
                        new_size = angle_index_array.shape[0]
                        for func_idx in range(old_size, new_size):
                            windows[scale_idx][dimension_idx].append(
                                UDCTWindow._to_sparse(
                                    window_functions[func_idx - old_size],
                                    parameters.window_threshold,
                                )
                            )
                        indices[scale_idx][dimension_idx] = angle_index_array.copy()

        # Normalization
        UDCTWindow._inplace_normalize_windows(
            windows,
            size=parameters.size,
            dimension=parameters.dim,
            num_resolutions=parameters.res,
        )

        # sort the window
        UDCTWindow._inplace_sort_windows(
            windows=windows,
            indices=indices,
            num_resolutions=parameters.res,
            dimension=parameters.dim,
        )

        return windows, decimation_ratios, indices
