"""Window computation for PyTorch UDCT implementation."""

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from math import ceil
from typing import Union

import numpy as np
import torch

from ._typing import UDCTWindows
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
    >>> import torch
    >>> from curvelets.torch._utils import ParamUDCT
    >>> from curvelets.torch._udct_windows import UDCTWindow
    >>>
    >>> # Create parameters for 2D transform with 4 scales total (1 lowpass + 3 high-frequency)
    >>> params = ParamUDCT(
    ...     shape=(64, 64),
    ...     angular_wedges_config=torch.tensor([[3], [6], [12]]),
    ...     window_overlap=0.15,
    ...     window_threshold=1e-5,
    ...     radial_frequency_params=(torch.pi/3, 2*torch.pi/3, 2*torch.pi/3, 4*torch.pi/3)
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
        x_primary: torch.Tensor, x_secondary: torch.Tensor
    ) -> torch.Tensor:
        """Compute one angle component from meshgrid coordinates."""
        # Compute angle component using piecewise function
        primary_ratio = torch.zeros_like(x_primary)
        mask = (x_primary != 0) & (torch.abs(x_secondary) <= torch.abs(x_primary))
        primary_ratio[mask] = -x_secondary[mask] / x_primary[mask]

        secondary_ratio = torch.zeros_like(x_primary)
        mask = (x_secondary != 0) & (torch.abs(x_primary) < torch.abs(x_secondary))
        secondary_ratio[mask] = x_primary[mask] / x_secondary[mask]

        # Wrap secondary_ratio to the correct range
        wrapped_ratio = secondary_ratio.clone()
        wrapped_ratio[secondary_ratio < 0] = secondary_ratio[secondary_ratio < 0] + 2
        wrapped_ratio[secondary_ratio > 0] = secondary_ratio[secondary_ratio > 0] - 2

        angle_component = torch.zeros_like(x_primary)
        angle_component[:] = primary_ratio + wrapped_ratio
        angle_component[x_primary >= 0] = -2
        return angle_component

    @staticmethod
    def _create_angle_grids_from_frequency_grids(
        frequency_grid_1: torch.Tensor, frequency_grid_2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adapt frequency grids for angle computation."""
        meshgrid_dim1, meshgrid_dim2 = torch.meshgrid(
            frequency_grid_2, frequency_grid_1, indexing="xy"
        )

        angle_grid_1 = UDCTWindow._compute_angle_component(meshgrid_dim1, meshgrid_dim2)
        angle_grid_2 = UDCTWindow._compute_angle_component(meshgrid_dim2, meshgrid_dim1)

        return angle_grid_2, angle_grid_1

    @staticmethod
    def _create_angle_functions(
        angle_grid: torch.Tensor,
        direction: int,
        num_angular_wedges: int,
        window_overlap: float,
    ) -> torch.Tensor:
        """Create angle functions using Meyer windows."""
        angular_spacing = 2 / num_angular_wedges
        angular_boundaries = angular_spacing * torch.tensor(
            [-window_overlap, window_overlap, 1 - window_overlap, 1 + window_overlap],
            dtype=angle_grid.dtype,
            device=angle_grid.device,
        )

        angle_functions_list: list[torch.Tensor] = []
        if direction in (1, 2):
            for wedge_index in range(1, ceil(num_angular_wedges / 2) + 1):
                ang2 = -1 + (wedge_index - 1) * angular_spacing + angular_boundaries
                window_values = meyer_window(angle_grid, *ang2.tolist())
                angle_functions_list.append(window_values.unsqueeze(0))
        else:
            error_msg = f"Unrecognized direction: {direction}. Must be 1 or 2."
            raise ValueError(error_msg)
        return torch.cat(angle_functions_list, dim=0)

    @staticmethod
    def _compute_angle_kronecker_product(
        angle_function_1d: torch.Tensor,
        dimension_permutation: torch.Tensor,
        shape: tuple[int, ...],
        dimension: int,
    ) -> torch.Tensor:
        """Compute Kronecker product for angle functions."""
        dim_perm = dimension_permutation

        # Compute dimension sizes
        kronecker_dimension_sizes = torch.tensor(
            [
                int(torch.prod(torch.tensor(shape[: int(dim_perm[0]) - 1]))),
                int(
                    torch.prod(
                        torch.tensor(shape[int(dim_perm[0]) : int(dim_perm[1]) - 1])
                    )
                ),
                int(torch.prod(torch.tensor(shape[int(dim_perm[1]) : dimension]))),
            ],
            dtype=torch.int64,
            device=angle_function_1d.device,
        )

        # Step 1: kron with ones, then transpose and flatten (matching NumPy's T.ravel())
        # Ensure angle_function_1d is at least 2D for torch.kron
        if angle_function_1d.ndim == 1:
            angle_2d = angle_function_1d.unsqueeze(0)
        else:
            angle_2d = angle_function_1d

        ones_step1 = torch.ones(
            (int(kronecker_dimension_sizes[1]), 1),
            dtype=angle_2d.dtype,
            device=angle_2d.device,
        )
        kron_step1 = torch.kron(ones_step1, angle_2d)
        # T.ravel() equivalent: transpose 2D then flatten
        kron_step1_travel = kron_step1.T.flatten()

        # Step 2: kron with ones again, then flatten
        ones_step2 = torch.ones(
            (int(kronecker_dimension_sizes[2]), 1),
            dtype=angle_2d.dtype,
            device=angle_2d.device,
        )
        kron_step2 = torch.kron(ones_step2, kron_step1_travel.unsqueeze(0)).flatten()

        # Step 3: final kron, then transpose and flatten (matching NumPy's T.ravel())
        ones_step3 = torch.ones(
            (int(kronecker_dimension_sizes[0]), 1),
            dtype=angle_2d.dtype,
            device=angle_2d.device,
        )
        kron_step3 = torch.kron(kron_step2.unsqueeze(0), ones_step3).T.flatten()

        # Reshape to reversed shape, then transpose (matching NumPy)
        # For N-D tensors, use permute to reverse dimensions instead of deprecated .T
        result = kron_step3.reshape(*shape[::-1])
        # Transpose: reverse all dimensions
        return result.permute(*reversed(range(result.ndim)))

    @staticmethod
    def _flip_with_fft_shift(input_tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """Flip tensor along specified axis with frequency domain shift."""
        shift_vector = [0] * input_tensor.ndim
        shift_vector[axis] = 1
        flipped_tensor = torch.flip(input_tensor, dims=(axis,))
        return circular_shift(flipped_tensor, tuple(shift_vector))

    @staticmethod
    def _to_sparse(
        arr: torch.Tensor, threshold: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert tensor to sparse format."""
        arr_flat = arr.flatten()
        indices = torch.argwhere(arr_flat > threshold)
        return (indices, arr_flat[indices])

    @staticmethod
    def _nchoosek(n: Union[Iterable[int], torch.Tensor], k: int) -> torch.Tensor:
        """Generate all combinations of k elements from n."""
        if isinstance(n, torch.Tensor):
            n = n.tolist()
        return torch.tensor(list(combinations(n, k)), dtype=torch.int64)

    @staticmethod
    def _create_bandpass_windows(
        num_scales: int,
        shape: tuple[int, ...],
        radial_frequency_params: tuple[float, float, float, float],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Create bandpass windows using Meyer wavelets."""
        dimension = len(shape)
        frequency_grid: dict[int, torch.Tensor] = {}
        meyer_windows: dict[tuple[int, int], torch.Tensor] = {}

        for dimension_idx in range(dimension):
            frequency_grid[dimension_idx] = torch.linspace(
                -1.5 * np.pi, 0.5 * np.pi, shape[dimension_idx]
            )[:-1]
            # Append last value to match NumPy's behavior
            frequency_grid[dimension_idx] = torch.cat(
                [
                    frequency_grid[dimension_idx],
                    torch.tensor([0.5 * np.pi - (2 * np.pi / shape[dimension_idx])]),
                ]
            )
            # Actually recreate properly
            frequency_grid[dimension_idx] = torch.linspace(
                -1.5 * np.pi, 0.5 * np.pi, shape[dimension_idx] + 1
            )[:-1]

            meyer_params = torch.tensor(
                [-2, -1, radial_frequency_params[0], radial_frequency_params[1]],
                device=frequency_grid[dimension_idx].device,
            )
            abs_frequency_grid = torch.abs(frequency_grid[dimension_idx])
            meyer_windows[(num_scales - 1, dimension_idx)] = meyer_window(
                abs_frequency_grid, *meyer_params.tolist()
            )
            if num_scales == 2:
                meyer_windows[(num_scales - 1, dimension_idx)] = meyer_windows[
                    (num_scales - 1, dimension_idx)
                ] + meyer_window(
                    torch.abs(frequency_grid[dimension_idx] + 2 * np.pi),
                    *meyer_params.tolist(),
                )
            meyer_params[2] = radial_frequency_params[2]
            meyer_params[3] = radial_frequency_params[3]
            meyer_windows[(num_scales, dimension_idx)] = meyer_window(
                abs_frequency_grid, *meyer_params.tolist()
            )

            for scale_idx in range(num_scales - 2, 0, -1):
                meyer_params[2] = radial_frequency_params[0]
                meyer_params[3] = radial_frequency_params[1]
                meyer_params[2] /= 2 ** (num_scales - 1 - scale_idx)
                meyer_params[3] /= 2 ** (num_scales - 1 - scale_idx)
                meyer_windows[(scale_idx, dimension_idx)] = meyer_window(
                    abs_frequency_grid, *meyer_params.tolist()
                )

        bandpass_windows: dict[int, torch.Tensor] = {}
        # Get device from first frequency grid
        device = frequency_grid[0].device if frequency_grid else None
        for scale_idx in range(num_scales - 1, 0, -1):
            low_freq = torch.tensor([1.0], dtype=torch.float64, device=device)
            high_freq = torch.tensor([1.0], dtype=torch.float64, device=device)
            for dimension_idx in range(dimension - 1, -1, -1):
                low_freq = torch.kron(
                    meyer_windows[(scale_idx, dimension_idx)].to(torch.float64),
                    low_freq,
                )
                high_freq = torch.kron(
                    meyer_windows[(scale_idx + 1, dimension_idx)].to(torch.float64),
                    high_freq,
                )
            low_freq_nd = low_freq.reshape(*shape)
            high_freq_nd = high_freq.reshape(*shape)
            bandpass_nd = high_freq_nd - low_freq_nd
            bandpass_nd = torch.clamp(bandpass_nd, min=0)
            bandpass_windows[scale_idx] = bandpass_nd
        bandpass_windows[0] = low_freq_nd
        return frequency_grid, bandpass_windows

    @staticmethod
    def _create_direction_mappings(
        dimension: int, num_scales: int, device: torch.device | None = None
    ) -> list[torch.Tensor]:
        """Create direction mappings for each resolution scale."""
        result = []
        for _ in range(num_scales - 1):
            mapping = []
            for dimension_idx in range(dimension):
                row = list(range(dimension_idx)) + list(
                    range(dimension_idx + 1, dimension)
                )
                mapping.append(row)
            # Shape is (dimension, dimension-1)
            result.append(torch.tensor(mapping, dtype=torch.int64, device=device))
        return result

    @staticmethod
    def _create_angle_info(
        frequency_grid: dict[int, torch.Tensor],
        dimension: int,
        num_scales: int,
        angular_wedges_config: torch.Tensor,
        window_overlap: float,
        high_frequency_mode: str = "curvelet",
    ) -> tuple[
        dict[int, dict[tuple[int, int], torch.Tensor]],
        dict[int, dict[tuple[int, int], torch.Tensor]],
    ]:
        """Create angle functions and indices for window computation."""
        # Get device from frequency_grid
        device = frequency_grid[0].device if frequency_grid else None
        dimension_permutations = UDCTWindow._nchoosek(
            torch.arange(dimension, device=device), 2
        )
        angle_grid: dict[tuple[int, int], torch.Tensor] = {}
        for pair_index, dimension_pair in enumerate(dimension_permutations):
            angle_grids = UDCTWindow._create_angle_grids_from_frequency_grids(
                frequency_grid[int(dimension_pair[0])],
                frequency_grid[int(dimension_pair[1])],
            )
            angle_grid[(pair_index, 0)] = angle_grids[0]
            angle_grid[(pair_index, 1)] = angle_grids[1]

        angle_functions: dict[int, dict[tuple[int, int], torch.Tensor]] = {}
        angle_indices: dict[int, dict[tuple[int, int], torch.Tensor]] = {}
        for scale_idx in range(num_scales - 1):
            angle_functions[scale_idx] = {}
            angle_indices[scale_idx] = {}
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
                                int(
                                    angular_wedges_config[
                                        scale_idx,
                                        int(
                                            dimension_permutations[
                                                hyperpyramid_idx, 1 - direction_idx
                                            ]
                                        ),
                                    ]
                                ),
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
        """Normalize windows in-place to ensure tight frame property."""
        # Get device from windows
        _, val = windows[0][0][0]
        device = val.device
        sum_squared_windows = torch.zeros(size, dtype=torch.float64, device=device)
        indices, values = windows[0][0][0]
        idx_flat = indices.flatten()
        val_flat = values.flatten()
        sum_squared_windows.flatten()[idx_flat] += val_flat**2

        for scale_idx in range(1, num_scales):
            for direction_idx in range(len(windows[scale_idx])):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    indices, values = windows[scale_idx][direction_idx][wedge_idx]
                    idx_flat = indices.flatten()
                    val_flat = values.flatten()
                    sum_squared_windows.flatten()[idx_flat] += val_flat**2
                    if len(windows[scale_idx]) == dimension:
                        temp_window = torch.zeros(size, dtype=torch.float64)
                        temp_window.flatten()[idx_flat] = val_flat**2
                        temp_window = UDCTWindow._flip_with_fft_shift(
                            temp_window, direction_idx
                        )
                        sum_squared_windows += temp_window

        sum_squared_windows = torch.sqrt(sum_squared_windows)
        sum_squared_windows_flat = sum_squared_windows.flatten()

        indices, values = windows[0][0][0]
        idx_flat = indices.flatten()
        val_flat = values.flatten()
        val_flat /= sum_squared_windows_flat[idx_flat]

        for scale_idx in range(1, num_scales):
            for direction_idx in range(len(windows[scale_idx])):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    indices, values = windows[scale_idx][direction_idx][wedge_idx]
                    idx_flat = indices.flatten()
                    val_flat = values.flatten()
                    val_flat /= sum_squared_windows_flat[idx_flat]

    @staticmethod
    def _calculate_decimation_ratios_with_lowest(
        num_scales: int,
        dimension: int,
        angular_wedges_config: torch.Tensor,
        direction_mappings: list[torch.Tensor],
        high_frequency_mode: str = "curvelet",
    ) -> list[torch.Tensor]:
        """Calculate decimation ratios for each scale and direction."""
        device = angular_wedges_config.device
        decimation_ratios: list[torch.Tensor] = [
            torch.full(
                (1, dimension),
                fill_value=2 ** (num_scales - 2),
                dtype=torch.int64,
                device=device,
            )
        ]
        for scale_idx in range(1, num_scales):
            if high_frequency_mode == "wavelet" and scale_idx == num_scales - 1:
                decimation_ratios.append(
                    torch.ones((1, dimension), dtype=torch.int64, device=device)
                )
            else:
                dec_ratio = torch.full(
                    (dimension, dimension),
                    fill_value=2 ** (num_scales - scale_idx),
                    dtype=torch.int64,
                    device=device,
                )
                for direction_idx in range(dimension):
                    other_directions = direction_mappings[scale_idx - 1][
                        direction_idx, :
                    ]
                    for od in other_directions:
                        od_int = int(od)
                        dec_ratio[direction_idx, od_int] = (
                            2
                            * int(angular_wedges_config[scale_idx - 1, od_int])
                            * 2 ** (num_scales - 1 - scale_idx)
                            // 3
                        )
                decimation_ratios.append(dec_ratio)
        return decimation_ratios

    @staticmethod
    def _inplace_sort_windows(
        windows: UDCTWindows,
        indices: dict[int, dict[int, torch.Tensor]],
        num_scales: int,
    ) -> None:
        """Sort windows in-place by their angular indices."""
        for scale_idx in range(1, num_scales):
            for dimension_idx in indices[scale_idx]:
                angular_index_array = indices[scale_idx][dimension_idx]
                max_index_value = int(angular_index_array.max()) + 1

                sort_keys = torch.zeros(
                    angular_index_array.shape[0],
                    dtype=torch.int64,
                    device=angular_index_array.device,
                )
                for column_index in range(angular_index_array.shape[1]):
                    position_weight = angular_index_array.shape[1] - 1 - column_index
                    sort_keys += (
                        max_index_value**position_weight
                    ) * angular_index_array[:, column_index]

                sorted_indices = torch.argsort(sort_keys)
                indices[scale_idx][dimension_idx] = angular_index_array[sorted_indices]
                windows[scale_idx][dimension_idx] = [
                    windows[scale_idx][dimension_idx][int(idx)]
                    for idx in sorted_indices
                ]

    @staticmethod
    def _build_angle_indices_1d(
        scale_idx: int,
        dimension_idx: int,
        angle_functions: dict[int, dict[tuple[int, int], torch.Tensor]],
        dimension: int,
    ) -> torch.Tensor:
        """Build multi-dimensional angle index combinations using Kronecker products."""
        # Get device from angle_functions
        device = (
            angle_functions[scale_idx - 1][(dimension_idx, 0)].device
            if angle_functions
            else None
        )
        angle_indices_1d = torch.arange(
            len(angle_functions[scale_idx - 1][(dimension_idx, 0)]), device=device
        ).unsqueeze(1)

        for angle_dim_idx in range(1, dimension - 1):
            num_angles = len(
                angle_functions[scale_idx - 1][(dimension_idx, angle_dim_idx)]
            )
            angle_indices_2d = torch.arange(
                num_angles, device=angle_indices_1d.device
            ).unsqueeze(1)
            kron_1 = torch.kron(
                angle_indices_1d,
                torch.ones(
                    (num_angles, 1), dtype=torch.int64, device=angle_indices_1d.device
                ),
            )
            kron_2 = torch.kron(
                torch.ones(
                    (angle_indices_1d.shape[0], 1),
                    dtype=torch.int64,
                    device=angle_indices_1d.device,
                ),
                angle_indices_2d,
            )
            angle_indices_1d = torch.cat([kron_1, kron_2], dim=1)
        return angle_indices_1d

    @staticmethod
    def _process_single_window(
        scale_idx: int,
        dimension_idx: int,
        window_index: int,
        angle_indices_1d: torch.Tensor,
        angle_functions: dict[int, dict[tuple[int, int], torch.Tensor]],
        angle_indices: dict[int, dict[tuple[int, int], torch.Tensor]],
        bandpass_windows: dict[int, torch.Tensor],
        direction_mappings: list[torch.Tensor],
        max_angles_per_dim: torch.Tensor,
        shape: tuple[int, ...],
        dimension: int,
        window_threshold: float,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Process a single window_index value, constructing curvelet windows."""
        # Get device from bandpass_windows
        device = bandpass_windows[scale_idx].device
        window = torch.ones(shape, dtype=torch.float64, device=device)

        for angle_dim_idx in range(dimension - 1):
            angle_idx = int(
                angle_indices_1d.reshape(len(angle_indices_1d), -1)[
                    window_index, angle_dim_idx
                ]
            )
            angle_func = angle_functions[scale_idx - 1][(dimension_idx, angle_dim_idx)][
                angle_idx
            ]
            angle_idx_mapping = angle_indices[scale_idx - 1][
                (dimension_idx, angle_dim_idx)
            ]
            kron_angle = UDCTWindow._compute_angle_kronecker_product(
                angle_func, angle_idx_mapping, shape, dimension
            )
            window *= kron_angle

        window *= bandpass_windows[scale_idx]
        window = torch.sqrt(circular_shift(window, tuple(s // 4 for s in shape)))

        def needs_flipping(angle_idx: torch.Tensor, flip_dimension_index: int) -> bool:
            return bool(
                2 * (angle_idx[flip_dimension_index] + 1)
                <= max_angles_per_dim[flip_dimension_index]
            )

        def flip_angle_idx(
            angle_idx: torch.Tensor, flip_dimension_index: int
        ) -> torch.Tensor:
            flipped = angle_idx.clone()
            flipped[flip_dimension_index] = (
                max_angles_per_dim[flip_dimension_index]
                - 1
                - angle_idx[flip_dimension_index]
            )
            return flipped

        def add_flipped_indices(
            indices_list: list[torch.Tensor], flip_dimension_index: int
        ) -> list[torch.Tensor]:
            needs_flip_mask = [
                needs_flipping(idx, flip_dimension_index) for idx in indices_list
            ]
            flipped_indices = [
                flip_angle_idx(indices_list[i], flip_dimension_index)
                for i in range(len(indices_list))
                if needs_flip_mask[i]
            ]
            return indices_list + flipped_indices

        angle_indices_list = [angle_indices_1d[window_index, :].clone()]
        for flip_dimension_index in range(dimension - 2, -1, -1):
            angle_indices_list = add_flipped_indices(
                angle_indices_list, flip_dimension_index
            )

        num_windows = len(angle_indices_list)
        angle_indices_2d = torch.stack(angle_indices_list)
        window_functions = torch.zeros((num_windows, *shape), dtype=window.dtype)

        angle_to_source: dict[tuple[int, ...], tuple[int, int]] = {
            tuple(angle_indices_2d[0].tolist()): (0, -1)
        }
        angle_idx_to_window_idx: dict[tuple[int, ...], int] = {
            tuple(angle_indices_2d[0].tolist()): 0
        }
        for flip_dimension_index in range(dimension - 2, -1, -1):
            for source_idx_tuple, source_window_idx in list(
                angle_idx_to_window_idx.items()
            ):
                source_angle_idx = torch.tensor(
                    source_idx_tuple, device=angle_indices_2d.device
                )
                if needs_flipping(source_angle_idx, flip_dimension_index):
                    flipped_angle_idx = flip_angle_idx(
                        source_angle_idx, flip_dimension_index
                    )
                    flipped_angle_idx_tuple = tuple(flipped_angle_idx.tolist())
                    if flipped_angle_idx_tuple not in angle_to_source:
                        angle_to_source[flipped_angle_idx_tuple] = (
                            source_window_idx,
                            flip_dimension_index,
                        )
                        angle_idx_to_window_idx[flipped_angle_idx_tuple] = len(
                            angle_idx_to_window_idx
                        )

        window_functions[0] = window

        for window_idx in range(1, num_windows):
            angle_idx_tuple = tuple(angle_indices_2d[window_idx].tolist())
            source_window_idx, flip_dimension_index = angle_to_source[angle_idx_tuple]
            flip_axis_dimension = int(
                direction_mappings[scale_idx - 1][dimension_idx, flip_dimension_index]
            )
            window_functions[window_idx] = UDCTWindow._flip_with_fft_shift(
                window_functions[source_window_idx], flip_axis_dimension
            )

        window_tuples = [
            UDCTWindow._to_sparse(window_functions[i], window_threshold)
            for i in range(num_windows)
        ]

        return window_tuples, angle_indices_2d

    def compute(
        self,
    ) -> tuple[UDCTWindows, list[torch.Tensor], dict[int, dict[int, torch.Tensor]]]:
        """
        Compute curvelet windows in frequency domain for UDCT transform.

        Returns
        -------
        windows : UDCTWindows
            Curvelet windows in sparse format.
        decimation_ratios : list[torch.Tensor]
            Decimation ratios for each scale and direction.
        indices : dict[int, dict[int, torch.Tensor]]
            Angular indices for each window.
        """
        frequency_grid, bandpass_windows = UDCTWindow._create_bandpass_windows(
            num_scales=self.num_scales,
            shape=self.shape,
            radial_frequency_params=self.radial_frequency_params,
        )
        low_frequency_window = circular_shift(
            torch.sqrt(bandpass_windows[0]), tuple(s // 4 for s in self.shape)
        )

        windows: UDCTWindows = []
        windows.append([])
        windows[0].append([])
        windows[0][0] = [
            UDCTWindow._to_sparse(low_frequency_window, self.window_threshold)
        ]

        indices: dict[int, dict[int, torch.Tensor]] = {}
        indices[0] = {}
        indices[0][0] = torch.zeros((1, 1), dtype=torch.int64)
        direction_mappings = UDCTWindow._create_direction_mappings(
            dimension=self.dimension,
            num_scales=self.num_scales,
            device=self.angular_wedges_config.device,
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

        for scale_idx in range(1, self.num_scales):
            windows.append([])
            indices[scale_idx] = {}

            if (
                self.high_frequency_mode == "wavelet"
                and scale_idx == self.num_scales - 1
            ):
                lowpass_filter = bandpass_windows[0].clone()
                ring_window = 1.0 - lowpass_filter
                ring_window = torch.clamp(ring_window, min=0.0)
                ring_window = torch.sqrt(
                    circular_shift(ring_window, tuple(s // 4 for s in self.shape))
                )
                ring_window_sparse = UDCTWindow._to_sparse(
                    ring_window, self.window_threshold
                )
                windows[scale_idx].append([ring_window_sparse])
                indices[scale_idx][0] = torch.zeros(
                    (1, self.dimension - 1),
                    dtype=torch.int64,
                    device=self.angular_wedges_config.device,
                )
            else:
                for dimension_idx in range(self.dimension):
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

                    all_windows = [
                        window
                        for window_list, _ in window_results
                        for window in window_list
                    ]
                    all_angle_indices = [angle_idx for _, angle_idx in window_results]
                    angle_index_array = (
                        torch.cat(all_angle_indices, dim=0)
                        if all_angle_indices
                        else torch.zeros(
                            (0, self.dimension - 1),
                            dtype=torch.int64,
                            device=self.angular_wedges_config.device,
                        )
                    )

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
