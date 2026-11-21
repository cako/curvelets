# pylint: disable=too-many-lines,duplicate-code
# Duplicate code with torch implementation is expected
from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from math import ceil, lcm

import numpy as np
import numpy.typing as npt

from ._fft import _usable_cpu_count
from ._sparse_window import SparseWindow
from ._utils import ParamUDCT, circular_shift, meyer_window
from .typing import _F, UDCTWindows, _IntegerNDArray, _IntpNDArray


def _window_workers(workers: int | None = None) -> int:
    """Number of threads used to build windows: ``workers``, or every usable CPU."""
    return workers if workers is not None else _usable_cpu_count()


@dataclass
class _SortedGrid:
    """
    A grid whose flat indices are grouped by value relative to breakpoints.

    For sorted unique breakpoints ``B``, every grid value falls in one bucket:
    either exactly on a breakpoint ``B[k]`` (key ``2k + 1``) or strictly
    between ``B[k-1]`` and ``B[k]`` (key ``2k``). The flat indices are
    counting-sorted by key (stable, so ascending within a bucket). Every
    closed range ``[B[i], B[j]]`` is then a contiguous run of buckets, found
    without touching the grid.
    """

    flat: npt.NDArray[np.floating]
    breakpoints: npt.NDArray[np.floating]
    order: _IntpNDArray
    bucket_starts: _IntpNDArray
    shape: tuple[int, ...]

    @classmethod
    def from_grid(
        cls,
        grid: npt.NDArray[np.floating],
        breakpoints: npt.ArrayLike,
    ) -> _SortedGrid:
        flat = grid.reshape(-1)
        unique_breakpoints = np.unique(np.asarray(breakpoints, dtype=flat.dtype))
        left = np.searchsorted(unique_breakpoints, flat, side="left")
        right = np.searchsorted(unique_breakpoints, flat, side="right")
        num_keys = 2 * len(unique_breakpoints) + 1
        key_dtype = np.int16 if num_keys <= np.iinfo(np.int16).max else np.intp
        keys = (2 * left + (right != left)).astype(key_dtype)
        order = np.argsort(keys, kind="stable")  # radix sort for small ints
        bucket_starts: _IntpNDArray = np.zeros(num_keys + 1, dtype=np.intp)
        np.cumsum(np.bincount(keys, minlength=num_keys), out=bucket_starts[1:])
        return cls(
            flat=flat,
            breakpoints=unique_breakpoints,
            order=order,
            bucket_starts=bucket_starts,
            shape=grid.shape,
        )

    def indices_in_range(self, lo: float, hi: float) -> _IntpNDArray:
        """Ascending flat indices ``i`` with ``lo <= flat[i] <= hi``.

        ``lo`` and ``hi`` must be breakpoints the grid was built with.
        """
        first = int(np.searchsorted(self.breakpoints, lo))
        last = int(np.searchsorted(self.breakpoints, hi))
        if self.breakpoints[first] != lo or self.breakpoints[last] != hi:
            msg = "range bounds must be breakpoints of the grid"
            raise ValueError(msg)
        start = self.bucket_starts[2 * first + 1]
        stop = self.bucket_starts[2 * last + 2]
        return np.sort(self.order[start:stop])


@dataclass
class _SparseAngleFunction:
    """
    One wedge's angle function on a 2-D angle grid, stored sparsely.

    ``indices`` are ascending flat indices into a grid of shape
    ``grid_shape``; every entry not listed is exactly zero.
    """

    indices: _IntpNDArray
    values: npt.NDArray[np.floating]
    grid_shape: tuple[int, ...]
    _dense: npt.NDArray[np.floating] | None = field(default=None, repr=False)

    def dense(self) -> npt.NDArray[np.floating]:
        """Dense angle function (cached)."""
        if self._dense is None:
            dense = np.zeros(self.grid_shape, dtype=self.values.dtype)
            dense.reshape(-1)[self.indices] = self.values
            self._dense = dense
        return self._dense

    def axis_supports(self) -> tuple[npt.NDArray[np.bool_], ...]:
        """Per axis, the indices where this function has a nonzero entry."""
        nonzero = np.unravel_index(self.indices[self.values != 0], self.grid_shape)
        supports = []
        for axis, n in enumerate(self.grid_shape):
            support: npt.NDArray[np.bool_] = np.zeros(n, dtype=bool)
            support[nonzero[axis]] = True
            supports.append(support)
        return tuple(supports)


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
    workers : int | None, optional
        Number of threads used to build the windows. Default None (every CPU
        the process may use).

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
        self,
        parameters: ParamUDCT,
        high_frequency_mode: str = "curvelet",
        workers: int | None = None,
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
        self.workers = workers

    @staticmethod
    def _compute_angle_component(
        x_primary: npt.NDArray[_F], x_secondary: npt.NDArray[_F]
    ) -> npt.NDArray[_F]:
        """
        Compute one angle component from meshgrid coordinates.

        Parameters
        ----------
        x_primary : npt.NDArray[_F]
            Primary coordinate grid (used for conditions).
        x_secondary : npt.NDArray[_F]
            Secondary coordinate grid.

        Returns
        -------
        npt.NDArray[_F]
            Angle component array.
        """
        # Compute angle component using piecewise function:
        # When primary coordinate dominates (|x_secondary| <= |x_primary|), use -x_secondary/x_primary
        # (np.divide/np.where with masks: the same elementwise operations as
        # boolean-indexed assignment, without gathering and scattering)
        abs_primary = np.abs(x_primary)
        abs_secondary = np.abs(x_secondary)
        primary_ratio = np.divide(
            np.negative(x_secondary),
            x_primary,
            out=np.zeros_like(x_primary),
            where=(x_primary != 0) & (abs_secondary <= abs_primary),
        )

        # When secondary coordinate dominates (|x_primary| < |x_secondary|), use x_primary/x_secondary
        secondary_ratio = np.divide(
            x_primary,
            x_secondary,
            out=np.zeros_like(x_primary),
            where=(x_secondary != 0) & (abs_primary < abs_secondary),
        )

        # Wrap secondary_ratio to the correct range by adding/subtracting 2
        wrapped_ratio = np.where(
            secondary_ratio < 0,
            secondary_ratio + 2,
            np.where(secondary_ratio > 0, secondary_ratio - 2, secondary_ratio),
        )

        # Combine ratios and set special case for x_primary >= 0
        # Create result array with same dtype as input to preserve TypeVar _F
        angle_component = np.add(primary_ratio, wrapped_ratio, out=primary_ratio)
        angle_component[x_primary >= 0] = -2
        return angle_component

    @staticmethod
    def _create_angle_grids_from_frequency_grids(
        frequency_grid_1: npt.NDArray[_F], frequency_grid_2: npt.NDArray[_F]
    ) -> tuple[npt.NDArray[_F], npt.NDArray[_F]]:
        """
        Adapt frequency grids for angle computation.

        Parameters
        ----------
        frequency_grid_1 : npt.NDArray[_F]
            First frequency grid.
        frequency_grid_2 : npt.NDArray[_F]
            Second frequency grid.

        Returns
        -------
        tuple[npt.NDArray[_F], npt.NDArray[_F]]
            Adapted grid arrays angle_grid_2 and angle_grid_1.
        """
        meshgrid_dim1: npt.NDArray[np.float64]
        meshgrid_dim2: npt.NDArray[np.float64]
        meshgrid_dim1, meshgrid_dim2 = np.meshgrid(frequency_grid_2, frequency_grid_1)

        # Compute angle_grid_1 using meshgrid_dim1 as primary, meshgrid_dim2 as secondary
        angle_grid_1 = UDCTWindow._compute_angle_component(meshgrid_dim1, meshgrid_dim2)

        # Compute angle_grid_2 using meshgrid_dim2 as primary, meshgrid_dim1 as secondary (swapped)
        angle_grid_2 = UDCTWindow._compute_angle_component(meshgrid_dim2, meshgrid_dim1)

        return angle_grid_2, angle_grid_1

    @staticmethod
    def _create_angle_functions(
        angle_grid: npt.NDArray[_F],
        direction: int,
        num_angular_wedges: int,
        window_overlap: float,
    ) -> npt.NDArray[_F]:
        """
        Create angle functions using Meyer windows.

        Parameters
        ----------
        angle_grid : npt.NDArray[_F]
            Angle grid.
        direction : int
            Direction index (1 or 2).
        num_angular_wedges : int
            Number of angular wedges.
        window_overlap : float
            Window overlap parameter.

        Returns
        -------
        npt.NDArray[_F]
            Array of angle functions.
        """
        # Generate angle functions using Meyer windows
        # Note: Both direction 1 and 2 use the same computation because the
        # angle function is symmetric with respect to direction. The direction
        # parameter is kept for API consistency and potential future extensions.
        if direction not in (1, 2):
            error_msg = f"Unrecognized direction: {direction}. Must be 1 or 2."
            raise ValueError(error_msg)
        wedge_parameters = UDCTWindow._wedge_parameters(
            num_angular_wedges, window_overlap
        )
        sparse_functions = UDCTWindow._create_sparse_angle_functions(
            _SortedGrid.from_grid(angle_grid, np.concatenate(wedge_parameters)),
            num_angular_wedges,
            window_overlap,
        )
        angle_functions = np.zeros(
            (len(sparse_functions), *angle_grid.shape), dtype=angle_grid.dtype
        )
        for wedge, sparse_function in enumerate(sparse_functions):
            angle_functions[wedge].reshape(-1)[sparse_function.indices] = (
                sparse_function.values
            )
        return angle_functions

    @staticmethod
    def _wedge_parameters(
        num_angular_wedges: int, window_overlap: float
    ) -> list[npt.NDArray[np.float64]]:
        """
        Meyer window breakpoints of each angular wedge.

        Parameters
        ----------
        num_angular_wedges : int
            Number of angular wedges.
        window_overlap : float
            Window overlap parameter.

        Returns
        -------
        list[npt.NDArray[np.float64]]
            For each of the ``ceil(num_angular_wedges / 2)`` wedges, the four
            breakpoints passed to :py:func:`meyer_window`.
        """
        # Compute angular window spacing and boundaries
        angular_spacing = 2 / num_angular_wedges
        angular_boundaries = angular_spacing * np.array([
            -window_overlap,
            window_overlap,
            1 - window_overlap,
            1 + window_overlap,
        ])
        return [
            -1 + (wedge_index - 1) * angular_spacing + angular_boundaries
            for wedge_index in range(1, ceil(num_angular_wedges / 2) + 1)
        ]

    @staticmethod
    def _create_sparse_angle_functions(
        sorted_grid: _SortedGrid,
        num_angular_wedges: int,
        window_overlap: float,
    ) -> list[_SparseAngleFunction]:
        """
        Sparse version of :py:meth:`_create_angle_functions`.

        Each wedge's Meyer window is nonzero only where the angle grid lies in
        ``[lo, hi]`` (the window's breakpoint range). With the grid bucketed by
        those breakpoints once, that set is a contiguous run of buckets, so
        each wedge costs O(support) instead of several passes over the grid.

        Parameters
        ----------
        sorted_grid : _SortedGrid
            Angle grid, bucketed by (at least) the breakpoints of every wedge
            from :py:meth:`_wedge_parameters`.
        num_angular_wedges : int
            Number of angular wedges.
        window_overlap : float
            Window overlap parameter.

        Returns
        -------
        list[_SparseAngleFunction]
            One sparse angle function per wedge, with ascending flat indices
            into the grid. Values equal ``meyer_window`` of the grid exactly.
        """
        sparse_functions: list[_SparseAngleFunction] = []
        for ang2 in UDCTWindow._wedge_parameters(num_angular_wedges, window_overlap):
            indices = sorted_grid.indices_in_range(ang2.min(), ang2.max())
            values = meyer_window(sorted_grid.flat[indices], *ang2)
            sparse_functions.append(
                _SparseAngleFunction(
                    indices=indices, values=values, grid_shape=sorted_grid.shape
                )
            )
        return sparse_functions

    @staticmethod
    def _compute_angle_kronecker_product(
        angle_function_1d: npt.NDArray[_F],
        dimension_permutation: _IntegerNDArray,
        shape: tuple[int, ...],
        dimension: int,
    ) -> npt.NDArray[_F]:
        """
        Compute Kronecker product for angle functions.

        Expands a 1D angle function to an N-D array by replicating values along
        dimensions not in the permutation, using Kronecker products.

        Parameters
        ----------
        angle_function_1d : npt.NDArray[_F]
            Angle function array representing a 2D grid of values.
        dimension_permutation : _IntegerNDArray
            Dimension permutation indices (1-indexed) specifying which two
            dimensions the angle function spans.
        shape : tuple[int, ...]
            Shape of the output array.
        dimension : int
            Dimensionality of the transform.

        Returns
        -------
        npt.NDArray[_F]
            Read-only broadcast view with shape matching input shape. The dtype
            matches the input angle_function_1d dtype (preserves TypeVar _F).

        Notes
        -----
        The angle function spans axes ``dimension_permutation - 1`` and is
        constant along every other axis, so the Kronecker replication is
        exactly a broadcast: no N-D array is materialized.
        """
        broadcast_shape = [1] * dimension
        for axis in dimension_permutation:
            broadcast_shape[int(axis) - 1] = shape[int(axis) - 1]
        return np.broadcast_to(angle_function_1d.reshape(broadcast_shape), shape)

    @staticmethod
    def _flip_with_fft_shift(
        input_array: npt.NDArray[_F], axis: int
    ) -> npt.NDArray[_F]:
        """
        Flip array along specified axis with frequency domain shift.

        Parameters
        ----------
        input_array : npt.NDArray[_F]
            Input array.
        axis : int
            Axis along which to flip.

        Returns
        -------
        npt.NDArray[_F]
            Flipped and shifted array.
        """
        shift_vector = [0] * input_array.ndim
        shift_vector[axis] = 1
        flipped_array = np.flip(input_array, axis)
        return circular_shift(flipped_array, tuple(shift_vector))

    @staticmethod
    def _axis_supports(array: npt.NDArray[_F]) -> list[npt.NDArray[np.bool_]]:
        """
        Per-axis supports of an N-D array.

        Parameters
        ----------
        array : npt.NDArray[_F]
            Input array.

        Returns
        -------
        list[npt.NDArray[np.bool_]]
            For each axis ``a``, a boolean array of length ``array.shape[a]``
            that is True at index ``i`` iff ``array`` has a nonzero entry with
            coordinate ``i`` along ``a``.
        """
        nonzero = array != 0
        return [
            np.any(nonzero, axis=tuple(ax for ax in range(array.ndim) if ax != axis))
            for axis in range(array.ndim)
        ]

    @staticmethod
    def _flip_flat_indices(
        flat_indices: _IntpNDArray, shape: tuple[int, ...], axis: int
    ) -> _IntpNDArray:
        """
        Map flat indices through :py:meth:`_flip_with_fft_shift` along ``axis``.

        ``_flip_with_fft_shift`` sends element ``k`` along ``axis`` to
        ``(-k) mod n``; this applies that permutation to flat indices directly,
        so a sparse window can be flipped without densifying it.

        Parameters
        ----------
        flat_indices : _IntpNDArray
            Flat (C-order) indices into an array of shape ``shape``.
        shape : tuple[int, ...]
            Array shape.
        axis : int
            Axis along which to flip.

        Returns
        -------
        _IntpNDArray
            Flipped flat indices, in the same order as ``flat_indices``.
        """
        stride = int(np.prod(shape[axis + 1 :], dtype=np.int64))
        n = shape[axis]
        coord = (flat_indices // stride) % n
        return flat_indices + ((-coord) % n - coord) * stride

    @staticmethod
    def _flip_sparse_window(window: SparseWindow, axis: int) -> SparseWindow:
        """
        Sparse equivalent of ``from_dense(_flip_with_fft_shift(window.to_dense()))``.

        Parameters
        ----------
        window : SparseWindow
            Window to flip; its indices must be sorted ascending.
        axis : int
            Axis along which to flip.

        Returns
        -------
        SparseWindow
            Flipped window with ascending indices.
        """
        flipped = UDCTWindow._flip_flat_indices(window.indices, window.shape, axis)
        # Flipping one axis reorders whole blocks of an ascending sequence, so
        # the result is a few ascending runs: cheap for a stable (merge) sort
        order = np.argsort(flipped, kind="stable")
        return SparseWindow(
            indices=flipped[order], values=window.values[order], shape=window.shape
        )

    @staticmethod
    def _nchoosek(n: Iterable[int] | _IntegerNDArray, k: int) -> _IntegerNDArray:
        """
        Generate all combinations of k elements from n.

        Parameters
        ----------
        n : Iterable[int] | _IntegerNDArray
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
        frequency_grid : dict[int, npt.NDArray[_F]]
            Dictionary mapping dimension index to frequency grid array.
            Each grid spans [-1.5*pi, 0.5*pi) with size matching shape[dimension].
        bandpass_windows : dict[int, npt.NDArray[_F]]
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

        def outer_product(scale: int) -> npt.NDArray[np.float64]:
            # Same values (and multiplication order) as the Kronecker product
            # kron(m_0, kron(m_1, ... kron(m_{d-1}, [1.0]))), via broadcasting
            result: npt.NDArray[np.float64] = np.ones(
                (1,) * dimension, dtype=np.float64
            )
            for dimension_idx in range(dimension - 1, -1, -1):
                axis_shape = [1] * dimension
                axis_shape[dimension_idx] = shape[dimension_idx]
                result = (
                    meyer_windows[(scale, dimension_idx)]
                    .astype(np.float64)
                    .reshape(axis_shape)
                    * result
                )
            return result

        bandpass_windows: dict[int, npt.NDArray[np.float64]] = {}
        for scale_idx in range(num_scales - 1, 0, -1):
            low_freq_nd = outer_product(scale_idx)
            high_freq_nd = outer_product(scale_idx + 1)
            bandpass_nd = high_freq_nd - low_freq_nd
            bandpass_nd[bandpass_nd < 0] = 0
            bandpass_windows[scale_idx] = bandpass_nd
        bandpass_windows[0] = low_freq_nd
        return frequency_grid, bandpass_windows

    @staticmethod
    def _create_direction_mappings(
        dimension: int, num_scales: int
    ) -> list[_IntegerNDArray]:
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
        list[_IntegerNDArray]
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
        frequency_grid: dict[int, npt.NDArray[_F]],
        dimension: int,
        num_scales: int,
        angular_wedges_config: _IntegerNDArray,
        window_overlap: float,
        high_frequency_mode: str = "curvelet",
        grid_shift: tuple[int, ...] | None = None,
        workers: int | None = None,
    ) -> tuple[
        dict[int, dict[tuple[int, int], list[_SparseAngleFunction]]],
        dict[int, dict[tuple[int, int], _IntegerNDArray]],
    ]:
        """
        Create angle functions and indices for window computation.

        Parameters
        ----------
        frequency_grid : dict[int, npt.NDArray[_F]]
            Dictionary mapping dimension index to frequency grid.
        dimension : int
            Dimensionality of the transform.
        num_scales : int
            Total number of scales (including lowpass scale).
        angular_wedges_config : _IntegerNDArray
            Configuration array specifying number of angular wedges per scale and dimension.
        window_overlap : float
            Window overlap parameter.
        high_frequency_mode : str, optional
            High frequency mode. When "wavelet", overlap is set to 0 for the highest scale.
            Default is "curvelet".
        grid_shift : tuple[int, ...] | None, optional
            Per-axis circular shift applied to the 2-D angle grids before the
            (elementwise) angle functions are evaluated, so the returned angle
            functions are circularly shifted by the same amount. Default None
            (no shift).
        workers : int | None, optional
            Number of threads. Default None (every usable CPU).

        Returns
        -------
        tuple[dict[int, dict[tuple[int, int], list[_SparseAngleFunction]]], dict[int, dict[tuple[int, int], _IntegerNDArray]]]
            Tuple of (angle_functions, angle_indices) dictionaries. Each angle
            function entry is a list with one sparse function per wedge.
        """
        dimension_permutations = UDCTWindow._nchoosek(np.arange(dimension), 2)

        def scale_overlap(scale_idx: int) -> float:
            # Use overlap=0 for highest scale when mode="wavelet"
            # Note: scale_idx is 0-indexed for high-frequency scales (0 to num_scales-2)
            # In compute(), actual scale indices are 1 to num_scales-1, and they access
            # angle_functions[scale_idx - 1]. So the highest scale (num_scales - 1) maps
            # to angle_functions[num_scales - 2] here. Therefore, we check scale_idx == num_scales - 2
            # to identify the highest scale.
            is_highest_scale = scale_idx == num_scales - 2
            return (
                0.0
                if (high_frequency_mode == "wavelet" and is_highest_scale)
                else window_overlap
            )

        def grid_breakpoints(
            hyperpyramid_idx: int, direction_idx: int
        ) -> npt.NDArray[np.float64]:
            # Breakpoints of every wedge, at every scale, evaluated on this grid
            return np.concatenate([
                np.concatenate(
                    UDCTWindow._wedge_parameters(
                        angular_wedges_config[
                            scale_idx,
                            dimension_permutations[hyperpyramid_idx, 1 - direction_idx],
                        ],
                        scale_overlap(scale_idx),
                    )
                )
                for scale_idx in range(num_scales - 1)
            ])

        with ThreadPoolExecutor(max_workers=_window_workers(workers)) as pool:
            grid_futures: dict[tuple[int, int], Future[_SortedGrid]] = {}
            for pair_index, dimension_pair in enumerate(dimension_permutations):
                angle_grids = UDCTWindow._create_angle_grids_from_frequency_grids(
                    frequency_grid[dimension_pair[0]], frequency_grid[dimension_pair[1]]
                )
                if grid_shift is not None:
                    pair_shift = (
                        grid_shift[dimension_pair[0]],
                        grid_shift[dimension_pair[1]],
                    )
                    angle_grids = tuple(
                        circular_shift(grid, pair_shift) for grid in angle_grids
                    )
                # Bucketed once and shared by every scale's wedges on this grid
                for direction_idx in (0, 1):
                    grid_futures[(pair_index, direction_idx)] = pool.submit(
                        _SortedGrid.from_grid,
                        angle_grids[direction_idx],
                        grid_breakpoints(pair_index, direction_idx),
                    )
            angle_grid = {key: future.result() for key, future in grid_futures.items()}

            function_futures: dict[
                int, dict[tuple[int, int], Future[list[_SparseAngleFunction]]]
            ] = {}
            angle_indices: dict[int, dict[tuple[int, int], _IntegerNDArray]] = {}
            for scale_idx in range(num_scales - 1):
                function_futures[scale_idx] = {}
                angle_indices[scale_idx] = {}
                overlap_to_use = scale_overlap(scale_idx)
                for dimension_idx in range(dimension):
                    angle_function_index = 0
                    for hyperpyramid_idx in range(dimension_permutations.shape[0]):
                        for direction_idx in range(dimension_permutations.shape[1]):
                            if (
                                dimension_permutations[hyperpyramid_idx, direction_idx]
                                == dimension_idx
                            ):
                                function_futures[scale_idx][
                                    (dimension_idx, angle_function_index)
                                ] = pool.submit(
                                    UDCTWindow._create_sparse_angle_functions,
                                    angle_grid[(hyperpyramid_idx, direction_idx)],
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
            angle_functions = {
                scale_idx: {key: future.result() for key, future in futures.items()}
                for scale_idx, futures in function_futures.items()
            }
        return angle_functions, angle_indices

    @staticmethod
    def _inplace_normalize_windows(
        windows: UDCTWindows[np.floating],
        size: tuple[int, ...],
        dimension: int,
        num_scales: int,
        workers: int | None = None,
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
        workers : int | None, optional
            Number of threads. Default None (every usable CPU).
        """
        # Phase 1: Compute sum of squares of all windows (including flipped versions)
        # This ensures the tight frame property: sum of squares equals 1 at each frequency
        # Each window contributes its squares, and (when all directions are
        # present) the squares of its flip along its direction axis. Also
        # accumulate flipped version for symmetry: only flip if we have the
        # full number of directions (not for "wavelet" mode at highest scale);
        # for "wavelet" mode, the ring window is already symmetric.
        tasks: list[tuple[SparseWindow, int | None]] = [(windows[0][0][0], None)]
        for scale_idx in range(1, num_scales):
            # Iterate over actual number of directions (may be less than dimension for "wavelet" mode)
            flip = len(windows[scale_idx]) == dimension
            for direction_idx in range(len(windows[scale_idx])):
                tasks.extend(
                    (window, direction_idx if flip else None)
                    for window in windows[scale_idx][direction_idx]
                )

        def contributions(
            task: tuple[SparseWindow, int | None],
        ) -> tuple[_IntpNDArray, npt.NDArray[np.floating], _IntpNDArray | None]:
            window, flip_axis = task
            squares = window.values**2
            flipped = (
                None
                if flip_axis is None
                else UDCTWindow._flip_flat_indices(window.indices, size, flip_axis)
            )
            return window.indices, squares, flipped

        # The index maps and squares are computed in parallel; the scatter-adds
        # stay sequential, in the original order, so the sums are identical
        sum_squared_windows = np.zeros(size)
        sum_flat = sum_squared_windows.reshape(-1)
        with ThreadPoolExecutor(max_workers=_window_workers(workers)) as pool:
            for idx_flat, squares, flipped in pool.map(contributions, tasks):
                sum_flat[idx_flat] += squares
                if flipped is not None:
                    # Flipped indices are a permutation, so they are unique
                    sum_flat[flipped] += squares

        # Phase 2: Normalize each window by dividing by sqrt(sum of squares)
        # This ensures perfect reconstruction (tight frame property)
        sum_squared_windows = np.sqrt(sum_squared_windows)
        sum_squared_windows_flat = sum_squared_windows.ravel()
        window = windows[0][0][0]
        idx_flat = window.indices
        val_flat = window.values
        val_flat[:] /= sum_squared_windows_flat[idx_flat]
        for scale_idx in range(1, num_scales):
            # Iterate over actual number of directions (may be less than dimension for "wavelet" mode)
            for direction_idx in range(len(windows[scale_idx])):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    window = windows[scale_idx][direction_idx][wedge_idx]
                    idx_flat = window.indices
                    val_flat = window.values
                    val_flat[:] /= sum_squared_windows_flat[idx_flat]

    @staticmethod
    def _calculate_decimation_ratios_with_lowest(
        num_scales: int,
        dimension: int,
        angular_wedges_config: _IntegerNDArray,
        direction_mappings: list[_IntegerNDArray],
        high_frequency_mode: str = "curvelet",
    ) -> list[_IntegerNDArray]:
        """
        Calculate decimation ratios for each scale and direction.

        Parameters
        ----------
        num_scales : int
            Total number of scales (including lowpass scale).
        dimension : int
            Dimensionality of the transform.
        angular_wedges_config : _IntegerNDArray
            Configuration array specifying number of angular wedges.
        direction_mappings : list[_IntegerNDArray]
            Direction mappings for each resolution.
        high_frequency_mode : str, optional
            High frequency mode. For "wavelet" mode, highest scale has decimation=1.
            Default is "curvelet".

        Returns
        -------
        list[_IntegerNDArray]
            List of decimation ratio arrays, one per scale.
        """
        decimation_ratios: list[_IntegerNDArray] = [
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
        windows: UDCTWindows[np.floating],
        indices: dict[int, dict[int, _IntegerNDArray]],
        num_scales: int,
    ) -> None:
        """
        Sort windows in-place by their angular indices.

        Parameters
        ----------
        windows : UDCTWindows
            Windows to sort (modified in-place).
        indices : dict[int, dict[int, _IntegerNDArray]]
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
        angle_functions: dict[int, dict[tuple[int, int], list[_SparseAngleFunction]]],
        dimension: int,
    ) -> _IntegerNDArray:
        """
        Build multi-dimensional angle index combinations using Kronecker products.

        Parameters
        ----------
        scale_idx : int
            Scale index (1-based).
        dimension_idx : int
            Dimension index (0-based).
        angle_functions : dict[int, dict[tuple[int, int], list[_SparseAngleFunction]]]
            Dictionary of angle functions by scale and dimension.
        dimension : int
            Dimensionality of the transform.

        Returns
        -------
        _IntegerNDArray
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
    def _build_base_window(
        angle_terms: list[tuple[_SparseAngleFunction, _IntegerNDArray]],
        bandpass: npt.NDArray[_F],
        bandpass_support: list[npt.NDArray[np.bool_]],
        shape: tuple[int, ...],
        dimension: int,
        window_threshold: float,
    ) -> SparseWindow:
        r"""
        Build a base window :math:`\sqrt{A_1 \cdots A_{d-1} \cdot F_j}` in sparse form.

        Equivalent to ``SparseWindow.from_dense`` of the dense product, but
        evaluated only where it can be nonzero:

        - 2-D: the single angle function spans the whole grid, so the product
          is taken over that function's sparse support only.
        - N-D: each factor is zero on whole hyperplanes. Index ``i`` along axis
          ``a`` can only carry a nonzero value if every factor has a nonzero
          entry at ``i`` along ``a``; the product is evaluated on the Cartesian
          product of those per-axis index sets.

        Factors are multiplied in the same order as the dense product
        (angle functions, then bandpass), so values are identical.

        Parameters
        ----------
        angle_terms : list[tuple[_SparseAngleFunction, _IntegerNDArray]]
            Per angular dimension, the angle function and the (1-based) pair
            of axes it spans.
        bandpass : npt.NDArray[_F]
            Bandpass filter for the scale, shape ``shape``.
        bandpass_support : list[npt.NDArray[np.bool_]]
            Per-axis supports of ``bandpass``.
        shape : tuple[int, ...]
            Window shape.
        dimension : int
            Dimensionality of the transform.
        window_threshold : float
            Threshold for sparse window storage.

        Returns
        -------
        SparseWindow
            Base window with ascending indices.
        """
        if window_threshold < 0:
            # Zeros pass a negative threshold, so every grid point is kept
            axis_indices = [np.arange(n) for n in shape]
        elif dimension == 2:
            # The angle grid is the full 2-D grid, so its flat indices are the
            # window's flat indices
            ((angle_func, _),) = angle_terms
            indices = angle_func.indices
            values = np.multiply(
                angle_func.values, bandpass.reshape(-1)[indices], dtype=float
            )
            np.sqrt(values, out=values)
            keep = values > window_threshold
            return SparseWindow(
                indices=indices[keep], values=values[keep], shape=tuple(shape)
            )
        else:
            axis_keep = [support.copy() for support in bandpass_support]
            for angle_func, axes in angle_terms:
                support_0, support_1 = angle_func.axis_supports()
                axis_keep[int(axes[0]) - 1] &= support_0
                axis_keep[int(axes[1]) - 1] &= support_1
            axis_indices = [np.flatnonzero(keep) for keep in axis_keep]
        sub_shape = tuple(len(idx) for idx in axis_indices)

        # Factors are multiplied left to right, as W = 1 · A_1 · ... · F_j
        factors: list[npt.NDArray[_F]] = []
        for angle_func, axes in angle_terms:
            # Expand the 2-D angle function over the box (Kronecker product)
            sub_angle = angle_func.dense()[
                np.ix_(axis_indices[int(axes[0]) - 1], axis_indices[int(axes[1]) - 1])
            ]
            factors.append(
                UDCTWindow._compute_angle_kronecker_product(
                    sub_angle, axes, sub_shape, dimension
                )
            )
        factors.append(bandpass[np.ix_(*axis_indices)])

        # The first product allocates the window (1.0 · x == x exactly, so
        # this matches starting from ones()); the rest multiply in place
        window: npt.NDArray[_F] = np.multiply(factors[0], factors[1], dtype=float)
        for factor in factors[2:]:
            window *= factor
        np.sqrt(window, out=window)

        # Box-local thresholded entries, mapped to ascending global flat indices
        # (axis_indices are ascending, so box C-order is global C-order)
        window_flat = window.reshape(-1)
        local = np.flatnonzero(window_flat > window_threshold)
        global_indices = np.ravel_multi_index(
            tuple(
                idx[coord]
                for idx, coord in zip(
                    axis_indices, np.unravel_index(local, sub_shape), strict=True
                )
            ),
            shape,
        )
        return SparseWindow(
            indices=global_indices, values=window_flat[local], shape=tuple(shape)
        )

    @staticmethod
    def _process_single_window(
        scale_idx: int,
        dimension_idx: int,
        window_index: int,
        angle_indices_1d: _IntegerNDArray,
        angle_functions: dict[int, dict[tuple[int, int], list[_SparseAngleFunction]]],
        angle_indices: dict[int, dict[tuple[int, int], _IntegerNDArray]],
        bandpass_windows: dict[int, npt.NDArray[_F]],
        direction_mappings: list[_IntegerNDArray],
        max_angles_per_dim: _IntegerNDArray,
        shape: tuple[int, ...],
        dimension: int,
        window_threshold: float,
        bandpass_support: dict[int, list[npt.NDArray[np.bool_]]],
    ) -> tuple[list[SparseWindow], _IntegerNDArray]:
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
        angle_indices_1d : _IntegerNDArray
            Pre-computed angle index combinations from Kronecker products.
        angle_functions : dict[int, dict[tuple[int, int], list[_SparseAngleFunction]]]
            Dictionary of angle functions A_{j,l} by scale and dimension (Section IV).
        angle_indices : dict[int, dict[tuple[int, int], _IntegerNDArray]]
            Dictionary of angle indices by scale and dimension.
        bandpass_windows : dict[int, npt.NDArray[_F]]
            Dictionary of Meyer wavelet-based bandpass filters F_j by scale (Section IV).
        direction_mappings : list[_IntegerNDArray]
            Direction mappings for each resolution, used to determine flip axes.
        max_angles_per_dim : _IntegerNDArray
            Maximum angles per dimension, used to determine which indices need flipping.
        shape : tuple[int, ...]
            Shape of the input data.
        dimension : int
            Dimensionality of the transform.
        window_threshold : float
            Threshold for sparse window storage.
        bandpass_support : dict[int, list[npt.NDArray[np.bool_]]]
            Per scale, one boolean array per axis marking the indices where
            ``bandpass_windows[scale]`` has a nonzero entry.

        Returns
        -------
        tuple[list[tuple[_IntpNDArray, npt.NDArray[_F]]], _IntegerNDArray]
            Tuple containing:
            - List of window tuples (indices, values) for this window_index
              (including original and flipped versions) in sparse format
            - angle_indices_2d : _IntegerNDArray, shape (num_windows, dim-1)
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
        # angle_functions and bandpass_windows arrive already circularly shifted
        # by size//4 (see compute()); shifting commutes with the elementwise
        # products below, so the product is the shifted window directly.
        angle_terms: list[tuple[_SparseAngleFunction, _IntegerNDArray]] = []
        for angle_dim_idx in range(dimension - 1):
            angle_idx = angle_indices_1d.reshape(len(angle_indices_1d), -1)[
                window_index, angle_dim_idx
            ]
            # Get the angle function for this dimension and angle index
            # Note: scale_idx - 1 because angle_functions is 0-indexed for high-frequency scales
            # (scale 0 is lowpass, handled separately; scales 1..(num_scales-1) map to indices 0..(num_scales-2))
            key = (dimension_idx, angle_dim_idx)
            angle_terms.append((
                angle_functions[scale_idx - 1][key][angle_idx],
                angle_indices[scale_idx - 1][key],
            ))
        # Multiply by bandpass filter F_j (Meyer wavelet-based, Section IV)
        # Note: bandpass_windows[scale_idx] directly because bandpass_windows[0] is lowpass,
        # and bandpass_windows[1..(num_scales-1)] correspond to high-frequency scales 1..(num_scales-1)
        base_window = UDCTWindow._build_base_window(
            angle_terms,
            bandpass_windows[scale_idx],
            bandpass_support[scale_idx],
            shape,
            dimension,
            window_threshold,
        )

        # Step 2: Generate symmetric flipped versions for partition of unity
        # The partition of unity requires: |W_0(ω)|² + ∑|W_{j,l}(ω)|² + ∑|W_{j,l}(-ω)|² = 1
        # (Section IV, Nguyen & Chauris 2010)
        # We need symmetric windows W_{j,l}(-ω) to cover negative frequencies

        def needs_flipping(
            angle_idx: _IntegerNDArray, flip_dimension_index: int
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
            angle_idx: _IntegerNDArray, flip_dimension_index: int
        ) -> _IntegerNDArray:
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
            indices_list: list[_IntegerNDArray], flip_dimension_index: int
        ) -> list[_IntegerNDArray]:
            """
            Add flipped indices for a given dimension.

            For each index in the list that needs flipping, generate its
            symmetric counterpart. This builds the complete set of angle
            indices needed for the partition of unity.
            """
            needs_flip_mask = np.array([
                needs_flipping(idx, flip_dimension_index) for idx in indices_list
            ])
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

        num_windows = len(angle_indices_list)
        angle_indices_2d = np.array(angle_indices_list)

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
        # Store the original window (computed in Step 1) in sparse format
        window_tuples = [base_window]

        # Generate flipped windows W_{j,l}(-ω) by applying frequency-domain flips
        # These symmetric windows are needed for the partition of unity condition.
        # A flip only permutes entries, so thresholding commutes with it and the
        # flip is applied to the sparse indices instead of a dense copy.
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
            window_tuples.append(
                UDCTWindow._flip_sparse_window(
                    window_tuples[source_window_idx], flip_axis_dimension
                )
            )

        return window_tuples, angle_indices_2d

    def compute(
        self,
    ) -> tuple[
        UDCTWindows[np.floating],
        list[_IntegerNDArray],
        dict[int, dict[int, _IntegerNDArray]],
    ]:
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
            windows[scale][direction][wedge] = SparseWindow
            where scale 0 is low-frequency, scales 1..(num_scales-1) are high-frequency bands.
            For "wavelet" mode at highest scale, windows[highest_scale] has 1 direction with 1 wedge
            (ring-shaped window encompassing the entire frequency ring, no angular components).
        decimation_ratios : list[_IntegerNDArray]
            Decimation ratios for each scale and direction. Structure:
            - decimation_ratios[0]: shape (1, dim) for low-frequency band
            - decimation_ratios[scale]: shape (dim, dim) for scale > 0 (normal modes)
            - decimation_ratios[highest_scale]: shape (1, dim) with values=1 for "wavelet" mode
        indices : dict[int, dict[int, _IntegerNDArray]]
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
        windows: UDCTWindows[np.floating] = []
        windows.append([])
        windows[0].append([])
        windows[0][0] = [
            SparseWindow.from_dense(low_frequency_window, self.window_threshold)
        ]

        indices: dict[int, dict[int, _IntegerNDArray]] = {}
        indices[0] = {}
        indices[0][0] = np.zeros((1, 1), dtype=int)
        direction_mappings = UDCTWindow._create_direction_mappings(
            dimension=self.dimension, num_scales=self.num_scales
        )
        # Every high-frequency window is circularly shifted by size//4 along each
        # axis. A circular shift commutes with the elementwise angle functions
        # and with the elementwise products forming each window, so shift the
        # (lower-dimensional) angle grids and bandpass filters once here
        # instead of every N-D window.
        quarter_shift = tuple(s // 4 for s in self.shape)
        shifted_angle_functions, angle_indices = UDCTWindow._create_angle_info(
            frequency_grid,
            dimension=self.dimension,
            num_scales=self.num_scales,
            angular_wedges_config=self.angular_wedges_config,
            window_overlap=self.window_overlap,
            high_frequency_mode=self.high_frequency_mode,
            grid_shift=quarter_shift,
            workers=self.workers,
        )
        shifted_bandpass_windows = {
            scale_key: circular_shift(bandpass, quarter_shift)
            for scale_key, bandpass in bandpass_windows.items()
            if scale_key > 0
        }

        # Per-axis supports of each factor, used to evaluate each window only on
        # the box where all of its factors can be nonzero
        bandpass_support = {
            scale_key: UDCTWindow._axis_supports(bandpass)
            for scale_key, bandpass in shifted_bandpass_windows.items()
        }

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
        with ThreadPoolExecutor(max_workers=_window_workers(self.workers)) as pool:
            pending: list[
                tuple[
                    int,
                    int,
                    list[Future[tuple[list[SparseWindow], _IntegerNDArray]]],
                ]
            ] = []
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
                    ring_window_sparse = SparseWindow.from_dense(
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
                            angle_functions=shifted_angle_functions,
                            dimension=self.dimension,
                        )
                        num_windows = angle_indices_1d.shape[0]
                        max_angles_per_dim = self.angular_wedges_config[
                            scale_idx - 1,
                            direction_mappings[scale_idx - 1][dimension_idx, :],
                        ]

                        # Process each window_index independently (in parallel: the
                        # work is NumPy calls that release the GIL). Each call returns
                        # (list of window tuples, angle_indices_2d array); windows
                        # include original and flipped versions for symmetry.
                        futures = [
                            pool.submit(
                                UDCTWindow._process_single_window,
                                scale_idx=scale_idx,
                                dimension_idx=dimension_idx,
                                window_index=window_index,
                                angle_indices_1d=angle_indices_1d,
                                angle_functions=shifted_angle_functions,
                                angle_indices=angle_indices,
                                bandpass_windows=shifted_bandpass_windows,
                                direction_mappings=direction_mappings,
                                max_angles_per_dim=max_angles_per_dim,
                                shape=self.shape,
                                dimension=self.dimension,
                                window_threshold=self.window_threshold,
                                bandpass_support=bandpass_support,
                            )
                            for window_index in range(num_windows)
                        ]
                        windows[scale_idx].append([])  # filled in below
                        pending.append((scale_idx, dimension_idx, futures))

            # Combine results for each (scale_idx, dimension_idx), in order:
            # - Flatten nested window lists into a single list
            # - Concatenate all angle_indices_2d arrays into one array
            for scale_idx, dimension_idx, futures in pending:
                window_results = [future.result() for future in futures]
                windows[scale_idx][dimension_idx] = [
                    window
                    for window_list, _ in window_results
                    for window in window_list
                ]
                all_angle_indices = [angle_idx for _, angle_idx in window_results]
                indices[scale_idx][dimension_idx] = (
                    np.concatenate(all_angle_indices, axis=0)
                    if all_angle_indices
                    else np.zeros((0, self.dimension - 1), dtype=int)
                )

        UDCTWindow._inplace_normalize_windows(
            windows,
            size=self.shape,
            dimension=self.dimension,
            num_scales=self.num_scales,
            workers=self.workers,
        )

        UDCTWindow._inplace_sort_windows(
            windows=windows,
            indices=indices,
            num_scales=self.num_scales,
        )

        self._check_shape_compatible(decimation_ratios)

        return windows, decimation_ratios, indices

    def _check_shape_compatible(self, decimation_ratios: list[_IntegerNDArray]) -> None:
        """
        Warn when an axis length is not a multiple of its decimation ratios.

        The periodized fold maps frequency ``k`` to ``k mod (n // d)``, which is
        a true periodization only when ``d`` divides ``n``. In practice the
        axis length must also be a multiple of 4, otherwise the radial and
        angular windows no longer form a partition of unity on the grid. Either
        way the round trip loses ~1e-2 or more. Forward-only use still works,
        so this is a warning rather than an error.
        """
        for axis, length in enumerate(self.shape):
            divisor = lcm(
                4, *(int(r[axis]) for ratios in decimation_ratios for r in ratios)
            )
            if length % divisor:
                below = length - length % divisor
                logging.warning(
                    "shape[%d]=%d is not a multiple of %d (needed for "
                    "num_scales=%d and these wedges), so the transform will not "
                    "reconstruct to machine precision. Nearest valid sizes: %d "
                    "or %d.",
                    axis,
                    length,
                    divisor,
                    self.num_scales,
                    max(below, divisor),
                    below + divisor,
                )
