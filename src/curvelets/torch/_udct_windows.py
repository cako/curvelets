"""Window computation for PyTorch UDCT implementation.

The windows are built once, on the CPU, when a transform is created. They are
computed by the NumPy window builder and converted to tensors, so both
backends share a single (optimized) implementation and produce identical
windows.
"""

from __future__ import annotations

import numpy as np
import torch

from ..numpy._udct_windows import UDCTWindow as _NpUDCTWindow
from ..numpy._utils import ParamUDCT as _NpParamUDCT
from ._sparse_window import SparseWindow
from ._utils import ParamUDCT
from .typing import UDCTWindows


class UDCTWindow:
    """
    Window computation for Uniform Discrete Curvelet Transform.

    Thin wrapper around :class:`curvelets.numpy._udct_windows.UDCTWindow`:
    windows are computed with NumPy (in float64) and returned as tensors on
    the CPU, with the same structure as the NumPy windows.

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

    def compute(
        self,
    ) -> tuple[UDCTWindows, list[torch.Tensor], dict[int, dict[int, torch.Tensor]]]:
        """
        Compute curvelet windows in frequency domain for UDCT transform.

        Returns
        -------
        windows : UDCTWindows
            Curvelet windows in sparse format (int64 indices, float64 values,
            on the CPU). Structure is ``windows[scale][direction][wedge]``.
        decimation_ratios : list[torch.Tensor]
            Decimation ratios (int64) for each scale and direction.
        indices : dict[int, dict[int, torch.Tensor]]
            Angular indices (int64) for each window.
        """
        device = self.angular_wedges_config.device
        numpy_parameters = _NpParamUDCT(
            shape=tuple(int(s) for s in self.shape),
            angular_wedges_config=self.angular_wedges_config.cpu().numpy(),
            window_overlap=float(self.window_overlap),
            radial_frequency_params=(
                float(self.radial_frequency_params[0]),
                float(self.radial_frequency_params[1]),
                float(self.radial_frequency_params[2]),
                float(self.radial_frequency_params[3]),
            ),
            window_threshold=float(self.window_threshold),
        )
        np_windows, np_decimation_ratios, np_indices = _NpUDCTWindow(
            numpy_parameters, self.high_frequency_mode
        ).compute()

        windows: UDCTWindows = [
            [
                [
                    SparseWindow(
                        indices=torch.from_numpy(
                            np.asarray(window.indices, dtype=np.int64)
                        ),
                        values=torch.from_numpy(
                            np.asarray(window.values, dtype=np.float64)
                        ),
                        shape=tuple(window.shape),
                    )
                    for window in direction_windows
                ]
                for direction_windows in scale_windows
            ]
            for scale_windows in np_windows
        ]
        decimation_ratios = [
            torch.from_numpy(np.asarray(ratios, dtype=np.int64)).to(device)
            for ratios in np_decimation_ratios
        ]
        indices = {
            scale_idx: {
                direction_idx: torch.from_numpy(
                    np.asarray(angle_indices, dtype=np.int64)
                ).to(device)
                for direction_idx, angle_indices in scale_indices.items()
            }
            for scale_idx, scale_indices in np_indices.items()
        }
        return windows, decimation_ratios, indices
