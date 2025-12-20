from __future__ import annotations

__all__ = ["udctmdwin"]
from itertools import combinations
from typing import Any

# import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .typing import UDCTWindows
from .utils import (
    ParamUDCT,
    adapt_grid,
    angle_fun,
    angle_kron,
    circshift,
    fftflip,
    from_sparse_new,
    fun_meyer,
    to_sparse_new,
)


def _create_bandpass_windows(
    num_scales: int,
    shape: tuple[int, ...],
    radial_frequency_params: tuple[float, float, float, float],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
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


def _nchoosek(n: Any, k: Any) -> np.ndarray:
    return np.asarray(list(combinations(n, k)))


def _create_mdirs(dimension: int, num_resolutions: int) -> list[np.ndarray]:
    # Mdir is dimension of need to calculate angle function on each
    # hyperpyramid
    return [
        np.c_[
            [
                np.r_[np.arange(dimension_idx), np.arange(dimension_idx + 1, dimension)]
                for dimension_idx in range(dimension)
            ]
        ]
        for scale_idx in range(num_resolutions)
    ]
    # Mdirs: dict[int, np.ndarray] = {}
    # for ires in range(res):
    #     Mdirs[ires] = np.zeros((dim, dim - 1), dtype=int)
    #     for idim in range(dim):
    #         Mdirs[ires][idim, :] = np.r_[range(idim), range(idim + 1, dim)]

    # return Mdirs


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
    # every combination of 2 dimension out of 1:dimension
    dimension_permutations = _nchoosek(np.arange(dimension), 2)
    angle_grid: dict[tuple[int, int], np.ndarray] = {}
    for perm_idx, perm in enumerate(dimension_permutations):
        out = adapt_grid(frequency_grid[perm[0]], frequency_grid[perm[1]])
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
                            angle_fun(
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


def _inplace_normalize_windows(
    windows: UDCTWindows, size: tuple[int, ...], dimension: int, num_resolutions: int
) -> None:
    sum_squared_windows = np.zeros(size)
    idx, val = from_sparse_new(windows[0][0][0])
    sum_squared_windows.flat[idx] += val**2
    for scale_idx in range(1, num_resolutions + 1):
        for direction_idx in range(dimension):
            for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                temp_window = np.zeros(size)
                idx, val = from_sparse_new(windows[scale_idx][direction_idx][wedge_idx])
                temp_window.flat[idx] += val**2
                sum_squared_windows += temp_window
                temp_window = fftflip(temp_window, direction_idx)
                sum_squared_windows += temp_window

    sum_squared_windows = np.sqrt(sum_squared_windows)
    idx, val = from_sparse_new(windows[0][0][0])
    val /= sum_squared_windows.ravel()[idx]
    for scale_idx in range(1, num_resolutions + 1):
        for direction_idx in range(dimension):
            for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                idx, val = from_sparse_new(windows[scale_idx][direction_idx][wedge_idx])
                val /= sum_squared_windows.ravel()[idx]


def _calculate_decimation_ratios_with_lowest(
    num_resolutions: int,
    dimension: int,
    angular_wedges_config: np.ndarray,
    direction_mappings: list[np.ndarray],
) -> list[npt.NDArray[np.int_]]:
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


def _inplace_sort_windows(
    windows: UDCTWindows,
    indices: dict[int, dict[int, np.ndarray]],
    num_resolutions: int,
    dimension: int,
) -> None:
    for scale_idx in range(1, num_resolutions + 1):
        for dimension_idx in range(dimension):
            index_list = indices[scale_idx][dimension_idx]

            # # Approach 1: Create a structured array and then sort by fields
            # struct = [(f"x{i}", "<i8") for i in range(index_list.shape[1])]
            # sort_indices = np.argsort(
            #     np.array([tuple(m) for m in index_list], dtype=struct),
            #     order=tuple(t[0] for t in struct),
            # )
            #
            # Approach 2: Create a 1D array then sort that array
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


def udctmdwin(
    parameters: ParamUDCT,
) -> tuple[UDCTWindows, list[npt.NDArray[np.int_]], dict[int, dict[int, np.ndarray]]]:
    frequency_grid, bandpass_windows = _create_bandpass_windows(
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
    windows[0][0] = [to_sparse_new(low_frequency_window, parameters.window_threshold)]

    # `indices` gets stored as `parameters.ind` in the original.
    indices: dict[int, dict[int, np.ndarray]] = {}
    indices[0] = {}
    indices[0][0] = np.zeros((1, 1), dtype=int)
    direction_mappings = _create_mdirs(
        dimension=parameters.dim, num_resolutions=parameters.res
    )
    angle_functions, angle_indices = _create_angle_info(
        frequency_grid,
        dimension=parameters.dim,
        num_resolutions=parameters.res,
        angular_wedges_config=parameters.angular_wedges_config,
        window_overlap=parameters.window_overlap,
    )

    # decimation ratio for each band
    decimation_ratios = _calculate_decimation_ratios_with_lowest(
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
                    len(angle_functions[scale_idx - 1][(dimension_idx, angle_dim_idx)])
                )[:, None]
                kron_1 = np.kron(angle_indices_1d, np.ones((num_angles, 1), dtype=int))
                kron_2 = np.kron(
                    np.ones((angle_indices_1d.shape[0], 1), dtype=int), angle_indices_2d
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
                window: npt.NDArray[np.floating] = np.ones(parameters.size, dtype=float)
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
                    kron_angle = angle_kron(angle_func, angle_idx_mapping, parameters)
                    window *= kron_angle
                window *= bandpass_windows[scale_idx]
                window = np.sqrt(
                    circshift(window, tuple(s // 4 for s in parameters.size))
                )

                # first windows function
                window_functions = []
                window_functions.append(window)

                # index of current angle
                angle_indices_2d = angle_indices_1d[window_idx : window_idx + 1, :] + 1

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
                            window = fftflip(window_functions[func_idx], flip_axis)
                            window_functions.append(window)
                angle_indices_2d -= 1  # Adjust so that `indices` is 0-based
                window_functions = np.c_[window_functions]

                if window_idx == 0:
                    angle_index_array = angle_indices_2d
                    for func_idx in range(angle_index_array.shape[0]):
                        windows[scale_idx][dimension_idx].append(
                            to_sparse_new(
                                window_functions[func_idx], parameters.window_threshold
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
                            to_sparse_new(
                                window_functions[func_idx - old_size],
                                parameters.window_threshold,
                            )
                        )
                    indices[scale_idx][dimension_idx] = angle_index_array.copy()

    # Normalization
    _inplace_normalize_windows(
        windows,
        size=parameters.size,
        dimension=parameters.dim,
        num_resolutions=parameters.res,
    )

    # sort the window
    _inplace_sort_windows(
        windows=windows,
        indices=indices,
        num_resolutions=parameters.res,
        dimension=parameters.dim,
    )

    return windows, decimation_ratios, indices
