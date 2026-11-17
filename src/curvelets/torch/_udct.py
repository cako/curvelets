"""Main UDCT class for PyTorch implementation."""

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from __future__ import annotations

import logging
from collections.abc import Callable
from math import prod, sqrt
from typing import Any, Literal

import numpy as np
import torch

from ._autograd import BackwardWithAdjoint, ForwardWithAdjoint
from ._cuda_graphs import CudaGraphCache, graphable, graphable_with_adjoint
from ._dtypes import check_dtype, resolve_dtype
from ._plan import UDCTPlan
from ._udct_windows import UDCTWindow
from ._utils import ParamUDCT
from .typing import UDCTCoefficients, UDCTWindows


class UDCT:
    """
    Uniform Discrete Curvelet Transform (UDCT) implementation.

    This class provides forward and backward curvelet transforms with support
    for both real and complex transforms.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the input data.
    angular_wedges_config : torch.Tensor, optional
        Configuration tensor specifying the number of angular wedges per scale
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
        Default is (:math:`\\pi/3`, :math:`2\\pi/3`, :math:`2\\pi/3`, :math:`4\\pi/3`).
    window_threshold : float, optional
        Threshold for sparse window storage (values below this are stored as sparse).
        Default is 1e-5.
    high_frequency_mode : {"curvelet", "wavelet"}, optional
        High frequency mode. "curvelet" uses curvelets at all scales,
        "wavelet" creates a single ring-shaped window (bandpass filter only,
        no angular components) at the highest scale with decimation=1.
        Default is "curvelet".
    transform_kind : {"real", "complex", "monogenic"}, optional
        Type of transform to use:

        - "real" (default): Real transform where each band captures both
          positive and negative frequencies combined.
        - "complex": Complex transform which separates positive and negative
          frequency components into different bands. Each band is scaled by
          :math:`\\sqrt{0.5}`.
        - "monogenic": Monogenic transform that extends the curvelet transform
          by applying Riesz transforms, producing ndim+1 components per band
          (scalar plus all Riesz components).

    dtype : torch.dtype, optional
        Precision of the transform: ``torch.float32`` or ``torch.float64``
        (``torch.complex64`` / ``torch.complex128`` are accepted and mean the
        same precision). The windows are computed in double precision and
        stored once in this dtype. Inputs must have this precision (or its
        complex counterpart for the complex transform), and coefficients come
        out as :attr:`coefficient_dtype`. Default is
        :func:`torch.get_default_dtype` (normally ``torch.float32``).
    cuda_graphs : bool, optional
        Replay the real and complex transforms on CUDA from CUDA graphs. Each
        transform launches a few dozen small kernels, and for small and medium
        inputs launching them from Python takes longer than running them; a
        graph replay launches them all at once. The first call for each input
        shape and dtype captures a graph (a few warmup calls, plus a memory
        pool of about one call's intermediates); later calls replay it. The
        last :data:`cuda_graph_cache_size` graphs are kept.

        It applies to every flat or nested forward/backward call, to
        :meth:`filter` and to :class:`UDCTModule`. Calls that need gradients
        use the adjoint transform (the other direction) as their gradient, so
        both run from graphs; under ``torch.vmap``, ``torch.compile`` or
        another stream capture, and for CPU tensors, calls run as usual.

        On an RTX 4090, a 512² forward drops from 0.37 to 0.065 ms and the
        backward from 0.55 to 0.087 ms. For large inputs the GPU work
        dominates, and copying in and out of the graph's buffers makes replay
        slower than plain calls (6.7 vs 5.5 ms for a 2048² forward of a batch
        of 8), so leave it off there. Default is False.

    Attributes
    ----------
    dtype : torch.dtype
        Real precision of the transform.
    cuda_graphs : bool
        Whether CUDA calls replay CUDA graphs (can be switched at any time).
    cuda_graph_cache_size : int
        Number of captured graphs kept (least recently used dropped first).
    coefficient_dtype : torch.dtype
        Dtype of the coefficients (complex for the real and complex transforms).
    high_frequency_mode : str
        High frequency mode.
    transform_kind : str
        Type of transform being used ("real", "complex", or "monogenic").
    parameters : ParamUDCT
        Internal UDCT parameters.

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch import UDCT
    >>> # Create a 2D transform using num_scales (simplified interface)
    >>> transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
    >>> data = torch.randn(64, 64)
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> torch.allclose(data, recon, atol=1e-4)
    True
    >>> # Create using angular_wedges_config (advanced interface)
    >>> cfg = torch.tensor([[3, 3], [6, 6]], dtype=torch.int64)
    >>> transform2 = UDCT(shape=(64, 64), angular_wedges_config=cfg)
    >>> coeffs2 = transform2.forward(data)
    >>> recon2 = transform2.backward(coeffs2)
    >>> torch.allclose(data, recon2, atol=1e-4)
    True
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        angular_wedges_config: torch.Tensor | None = None,
        num_scales: int | None = None,
        wedges_per_direction: int | None = None,
        window_overlap: float | None = None,
        radial_frequency_params: tuple[float, float, float, float] | None = None,
        window_threshold: float = 1e-5,
        high_frequency_mode: Literal["curvelet", "wavelet"] = "curvelet",
        transform_kind: Literal["real", "complex", "monogenic"] = "real",
        dtype: torch.dtype | None = None,
        cuda_graphs: bool = False,
    ) -> None:
        dtype = resolve_dtype(dtype)

        # Validate transform_kind
        if transform_kind not in ("real", "complex", "monogenic"):
            msg = f"transform_kind must be 'real', 'complex', or 'monogenic', got {transform_kind!r}"
            raise ValueError(msg)

        # Store basic attributes
        self._high_frequency_mode = high_frequency_mode
        self._transform_kind = transform_kind

        # Calculate necessary parameters
        params_dict = self._initialize_parameters(
            shape=shape,
            angular_wedges_config=angular_wedges_config,
            num_scales=num_scales,
            wedges_per_direction=wedges_per_direction,
            window_overlap=window_overlap,
            radial_frequency_params=radial_frequency_params,
            window_threshold=window_threshold,
        )

        # Create ParamUDCT object
        self._parameters = ParamUDCT(
            shape=params_dict["internal_shape"],
            angular_wedges_config=params_dict["angular_wedges_config"],
            window_overlap=params_dict["window_overlap"],
            radial_frequency_params=params_dict["radial_frequency_params"],
            window_threshold=params_dict["window_threshold"],
        )

        # Calculate windows
        self._windows, self._decimation_ratios, self._indices = (
            self._initialize_windows(dtype)
        )
        # Integer metadata used to compute scale factors and coefficient shapes.
        # It stays on the CPU so the transform never syncs with the device for it,
        # even after the public ``decimation_ratios`` are moved with ``.to()``.
        self._decimation_ratios_cpu: list[torch.Tensor] = [
            d.detach().cpu().clone() for d in self._decimation_ratios
        ]
        # Fused execution plan, built lazily from the windows on first use and
        # dropped whenever the windows move (see apply_to_tensors).
        self._plan: UDCTPlan | None = None
        # Captured CUDA graphs of the plan's flat transforms (cuda_graphs=True).
        self._cuda_graphs = cuda_graphs
        self._graphs = CudaGraphCache()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the transform."""
        return self._parameters.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._parameters.ndim

    @property
    def num_scales(self) -> int:
        """Number of scales."""
        return self._parameters.num_scales

    @property
    def dtype(self) -> torch.dtype:
        """Real precision of the transform (follows ``UDCTModule.to(dtype)``)."""
        return self._windows[0][0][0].values.dtype

    @property
    def coefficient_dtype(self) -> torch.dtype:
        """Dtype of the coefficients: complex, or real for the monogenic transform."""
        if self._transform_kind == "monogenic":
            return self.dtype
        return self.dtype.to_complex()

    @property
    def cuda_graphs(self) -> bool:
        """Whether CUDA calls replay CUDA graphs (see the class docstring)."""
        return self._cuda_graphs

    @cuda_graphs.setter
    def cuda_graphs(self, enabled: bool) -> None:
        self._cuda_graphs = enabled
        if not enabled:
            self._graphs.clear()

    @property
    def cuda_graph_cache_size(self) -> int:
        """Number of captured CUDA graphs kept (least recently used dropped)."""
        return self._graphs.max_entries

    @cuda_graph_cache_size.setter
    def cuda_graph_cache_size(self, size: int) -> None:
        self._graphs.max_entries = size
        self._graphs.clear()

    def clear_cuda_graphs(self) -> None:
        """Drop all captured CUDA graphs and their memory."""
        self._graphs.clear()

    @property
    def windows(self) -> UDCTWindows:
        """Curvelet windows in sparse format."""
        return self._windows

    @property
    def decimation_ratios(self) -> list[torch.Tensor]:
        """Decimation ratios for each scale."""
        return self._decimation_ratios

    @staticmethod
    def _compute_optimal_window_overlap(
        wedges_per_scale: torch.Tensor,
    ) -> float:
        """
        Compute optimal window_overlap from Nguyen & Chauris (2010) constraint.

        The constraint :math:`(2^{s/N})(1+2a)(1+a) < N` must hold for all scales.
        This method solves for the theoretical maximum at each scale,
        takes the minimum across all scales, and returns 10% of this value.

        The 10% factor was empirically determined to give better reconstruction
        quality than both the theoretical maximum and previous hardcoded defaults.

        Parameters
        ----------
        wedges_per_scale : torch.Tensor
            Tensor of wedge counts per scale.

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
        >>> import torch
        >>> from curvelets.torch import UDCT
        >>> # For wedges_per_direction=3, num_scales=3
        >>> wedges_per_scale = torch.tensor([3, 6], dtype=torch.int64)
        >>> alpha = UDCT._compute_optimal_window_overlap(wedges_per_scale)
        >>> 0.1 < alpha < 0.2  # Expected around 0.11
        True
        """
        min_overlap = float("inf")

        for scale_idx, num_wedges in enumerate(wedges_per_scale, start=1):
            num_wedges_float = float(num_wedges.item())
            k = 2 ** (scale_idx / num_wedges_float)
            # Solve: 2k*a^2 + 3k*a + (k - N) = 0
            # a = (-3 + sqrt(1 + 8*N/k)) / 4
            discriminant = 1 + 8 * num_wedges_float / k
            a_max = (-3 + sqrt(discriminant)) / 4
            min_overlap = min(min_overlap, a_max)

        # Use 10% of theoretical max for optimal reconstruction quality
        return 0.1 * min_overlap

    @staticmethod
    def _compute_from_angular_wedges_config(
        angular_wedges_config: torch.Tensor,
        window_overlap: float | None,
    ) -> tuple[torch.Tensor, float]:
        """
        Compute angular wedges configuration and window overlap from provided config.

        Parameters
        ----------
        angular_wedges_config : torch.Tensor
            Configuration tensor specifying the number of angular wedges per scale
            and dimension. Shape is (num_scales - 1, dimension), where num_scales
            includes the lowpass scale.
        window_overlap : float | None
            Window overlap parameter. If None, automatically computed using
            the Nguyen & Chauris (2010) constraint formula.

        Returns
        -------
        tuple[torch.Tensor, float]
            Tuple of (computed_angular_wedges_config, computed_window_overlap).
        """
        # Use provided angular_wedges_config directly
        computed_angular_wedges_config = angular_wedges_config

        # Validate that all wedge counts are divisible by 3
        # According to Nguyen & Chauris (2010), the decimation ratio formula
        # uses integer division by 3, so wedges must be divisible by 3
        invalid_mask = computed_angular_wedges_config % 3 != 0
        if invalid_mask.any():
            invalid_values = torch.unique(
                computed_angular_wedges_config[invalid_mask]
            ).tolist()
            msg = (
                f"All values in angular_wedges_config must be divisible by 3. "
                f"Found invalid values: {invalid_values}. "
                "According to the Nguyen & Chauris (2010) paper specification, "
                "the decimation ratio formula requires integer division by 3. "
                "Recommended values are: 3, 6, 12"
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
    ) -> tuple[torch.Tensor, float]:
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
        tuple[torch.Tensor, float]
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
                "Recommended values are: 3, 6, 12"
            )
            raise ValueError(msg)

        # Convert to angular_wedges_config
        wedges_per_scale = wedges_per_direction * 2 ** torch.arange(
            num_scales - 1, dtype=torch.int64
        )
        computed_angular_wedges_config = wedges_per_scale.unsqueeze(1).repeat(
            1, dimension
        )

        # Auto-select window_overlap if not provided
        if window_overlap is None:
            computed_window_overlap = UDCT._compute_optimal_window_overlap(
                wedges_per_scale
            )
        else:
            computed_window_overlap = window_overlap

        # Validate window_overlap
        for scale_idx, num_wedges in enumerate(wedges_per_scale, start=1):
            num_wedges_float = float(num_wedges.item())
            const = (
                2 ** (scale_idx / num_wedges_float)
                * (1 + 2 * computed_window_overlap)
                * (1 + computed_window_overlap)
            )
            if const >= num_wedges_float:
                msg = (
                    f"window_overlap={computed_window_overlap:.3f} does not respect the relationship "
                    f"(2^{scale_idx}/{num_wedges_float})(1+2a)(1+a) = {const:.3f} < {num_wedges_float} for scale {scale_idx + 1}"
                )
                logging.warning(msg)

        return computed_angular_wedges_config, computed_window_overlap

    def _initialize_parameters(
        self,
        shape: tuple[int, ...],
        angular_wedges_config: torch.Tensor | None,
        num_scales: int | None,
        wedges_per_direction: int | None,
        window_overlap: float | None,
        radial_frequency_params: tuple[float, float, float, float] | None,
        window_threshold: float,
    ) -> dict[str, Any]:
        """
        Calculate all necessary parameters for UDCT initialization.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the input data.
        angular_wedges_config : torch.Tensor | None
            Configuration tensor, or None to use num_scales/wedges_per_direction.
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
            computed_radial_frequency_params: tuple[float, float, float, float] = (
                np.pi / 3,
                2 * np.pi / 3,
                2 * np.pi / 3,
                4 * np.pi / 3,
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
        self, dtype: torch.dtype
    ) -> tuple[UDCTWindows, list[torch.Tensor], dict[int, dict[int, torch.Tensor]]]:
        """
        Calculate curvelet windows (stored in ``dtype``), decimation ratios, and indices.

        Returns
        -------
        tuple
            (windows, decimation_ratios, indices)
        """
        window_computer = UDCTWindow(self._parameters, self._high_frequency_mode)
        windows, decimation_ratios, indices = window_computer.compute()
        need_flip = self._transform_kind == "complex"
        for scale_idx, scale_windows in enumerate(windows):
            for direction_idx, direction_windows in enumerate(scale_windows):
                if decimation_ratios[scale_idx].shape[0] == 1:
                    dec = decimation_ratios[scale_idx][0]
                else:
                    dec = decimation_ratios[scale_idx][
                        min(direction_idx, decimation_ratios[scale_idx].shape[0] - 1)
                    ]
                for window in direction_windows:
                    window.attach_periodized(dec, with_flip=need_flip)
                    # Computed in double precision, stored once in the transform's.
                    window.values = window.values.to(dtype)
        return windows, decimation_ratios, indices

    def coefficient_shapes(self) -> list[list[list[tuple[int, ...]]]]:
        """
        Calculate shapes of all curvelet-domain coefficient arrays.

        Computes the expected shape of every coefficient wedge across all
        scales, directions, and angular wedges without executing a forward
        transform.

        Returns
        -------
        list[list[list[tuple[int, ...]]]]
            Nested list containing the shape of each coefficient wedge.
            Structure is `shapes[scale_idx][direction_idx][wedge_idx]`.
            For monogenic transforms, each shape includes the channel dimension
            as the last axis ``(*wedge_shape, ndim + 2)``.

        Examples
        --------
        >>> from curvelets.torch import UDCT
        >>> transform = UDCT(shape=(64, 64), num_scales=3)
        >>> shapes = transform.coefficient_shapes()
        >>> len(shapes) == transform.num_scales
        True
        >>> shapes[0][0][0]  # Low-frequency band shape
        (16, 16)
        """
        if self._transform_kind == "monogenic":
            return self._coefficient_shapes_monogenic()
        if self._transform_kind == "complex":
            return self._coefficient_shapes_complex()
        return self._coefficient_shapes_real()

    def _coefficient_shapes_real(self) -> list[list[list[tuple[int, ...]]]]:
        """Private method for real transform coefficient shapes calculation."""
        shapes: list[list[list[tuple[int, ...]]]] = []
        internal_shape = torch.tensor(self._parameters.shape, dtype=torch.int64)
        for scale_idx, decimation_ratios_scale in enumerate(
            self._decimation_ratios_cpu
        ):
            shapes.append([])
            num_directions = len(decimation_ratios_scale)
            for direction_idx in range(num_directions):
                shapes[scale_idx].append([])
                window_direction_idx = min(
                    direction_idx, len(self._windows[scale_idx]) - 1
                )
                decimation_ratio_dir = decimation_ratios_scale[
                    min(direction_idx, len(decimation_ratios_scale) - 1), :
                ]
                for _ in self._windows[scale_idx][window_direction_idx]:
                    shape_dec = tuple(
                        int(x)
                        for x in (internal_shape // decimation_ratio_dir).tolist()
                    )
                    shapes[scale_idx][direction_idx].append(shape_dec)
        return shapes

    def _coefficient_shapes_complex(self) -> list[list[list[tuple[int, ...]]]]:
        """Private method for complex transform coefficient shapes calculation."""
        shapes: list[list[list[tuple[int, ...]]]] = []
        internal_shape = torch.tensor(self._parameters.shape, dtype=torch.int64)
        for scale_idx, decimation_ratios_scale in enumerate(
            self._decimation_ratios_cpu
        ):
            shapes.append([])
            if scale_idx > 0:
                num_directions = 2 * self._parameters.ndim
            else:
                num_directions = len(decimation_ratios_scale)
            for direction_idx in range(num_directions):
                shapes[scale_idx].append([])
                if scale_idx > 0 and direction_idx >= self._parameters.ndim:
                    window_direction_idx = direction_idx % self._parameters.ndim
                    window_direction_idx = min(
                        window_direction_idx, len(self._windows[scale_idx]) - 1
                    )
                    decimation_ratio_dir = decimation_ratios_scale[
                        min(window_direction_idx, len(decimation_ratios_scale) - 1), :
                    ]
                else:
                    window_direction_idx = min(
                        direction_idx, len(self._windows[scale_idx]) - 1
                    )
                    decimation_ratio_dir = decimation_ratios_scale[
                        min(direction_idx, len(decimation_ratios_scale) - 1), :
                    ]
                for _ in self._windows[scale_idx][window_direction_idx]:
                    shape_dec = tuple(
                        int(x)
                        for x in (internal_shape // decimation_ratio_dir).tolist()
                    )
                    shapes[scale_idx][direction_idx].append(shape_dec)
        return shapes

    def _coefficient_shapes_monogenic(self) -> list[list[list[tuple[int, ...]]]]:
        """Private method for monogenic transform coefficient shapes calculation."""
        shapes: list[list[list[tuple[int, ...]]]] = []
        internal_shape = torch.tensor(self._parameters.shape, dtype=torch.int64)
        num_channels = self._parameters.ndim + 2
        for scale_idx, decimation_ratios_scale in enumerate(
            self._decimation_ratios_cpu
        ):
            shapes.append([])
            num_directions = len(decimation_ratios_scale)
            for direction_idx in range(num_directions):
                shapes[scale_idx].append([])
                window_direction_idx = min(
                    direction_idx, len(self._windows[scale_idx]) - 1
                )
                decimation_ratio_dir = decimation_ratios_scale[
                    min(direction_idx, len(decimation_ratios_scale) - 1), :
                ]
                for _ in self._windows[scale_idx][window_direction_idx]:
                    shape_dec = (
                        *(
                            int(x)
                            for x in (internal_shape // decimation_ratio_dir).tolist()
                        ),
                        num_channels,
                    )
                    shapes[scale_idx][direction_idx].append(shape_dec)
        return shapes

    def vect(self, coefficients: UDCTCoefficients) -> torch.Tensor:
        """
        Vectorize curvelet coefficients.

        Parameters
        ----------
        coefficients : UDCTCoefficients
            Curvelet coefficients.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(*batch, ncoef)`` containing all coefficients,
            where ``batch`` are the leading dimensions of the coefficients
            (1D for unbatched coefficients).
        """
        # Dispatch based on transform_kind
        if self._transform_kind == "monogenic":
            return self._vect_monogenic(coefficients)
        return self._get_plan().flatten(coefficients)

    def _vect_monogenic(self, coefficients: UDCTCoefficients) -> torch.Tensor:
        """Private method for monogenic coefficient vectorization (no input validation)."""
        parts: list[torch.Tensor] = []
        for scale_coeffs in coefficients:
            for direction_coeffs in scale_coeffs:
                for wedge_coeffs in direction_coeffs:
                    # wedge_coeffs has shape (*wedge_shape, ndim+2)
                    # Flatten each channel separately to maintain order
                    num_channels = wedge_coeffs.shape[-1]
                    for ch_idx in range(num_channels):
                        parts.append(wedge_coeffs[..., ch_idx].flatten())
        return torch.cat(parts)

    def struct(self, vector: torch.Tensor) -> UDCTCoefficients:
        """
        Restructure vectorized coefficients to nested list format.

        Parameters
        ----------
        vector : torch.Tensor
            Coefficients of shape ``(*batch, ncoef)``.

        Returns
        -------
        UDCTCoefficients
            Restructured coefficients, each of shape ``(*batch, *wedge_shape)``,
            as views into ``vector`` (same dtype).

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import UDCT
        >>> transform = UDCT(shape=(64, 64), angular_wedges_config=torch.tensor([[3, 3]]))
        >>> data = torch.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> vec = transform.vect(coeffs)
        >>> coeffs_recon = transform.struct(vec)
        >>> len(coeffs_recon) == len(coeffs)
        True
        """
        # Dispatch based on transform_kind
        if self._transform_kind == "monogenic":
            return self._struct_monogenic(vector)
        return self._get_plan().unflatten(vector)

    def _struct_monogenic(self, vector: torch.Tensor) -> UDCTCoefficients:
        """Private method for monogenic coefficient restructuring (no input validation)."""
        begin_idx = 0
        coefficients: UDCTCoefficients = []
        internal_shape = torch.tensor(self._parameters.shape, dtype=torch.int64)
        # ndim+2 channels: [scalar.real, scalar.imag, riesz_1, ..., riesz_ndim]
        # Matches numpy's structure
        num_channels = self._parameters.ndim + 2

        for scale_idx, decimation_ratios_scale in enumerate(
            self._decimation_ratios_cpu
        ):
            coefficients.append([])
            num_directions = len(decimation_ratios_scale)

            for direction_idx in range(num_directions):
                coefficients[scale_idx].append([])
                window_direction_idx = min(
                    direction_idx, len(self._windows[scale_idx]) - 1
                )
                decimation_ratio_dir = decimation_ratios_scale[
                    min(direction_idx, len(decimation_ratios_scale) - 1), :
                ]

                for _ in self._windows[scale_idx][window_direction_idx]:
                    shape_decimated = (internal_shape // decimation_ratio_dir).tolist()
                    # For monogenic, each wedge has shape (*wedge_shape, ndim+2)
                    # Stack channels along last axis to match numpy structure
                    wedge_channels: list[torch.Tensor] = []
                    for _ in range(num_channels):
                        size = prod(shape_decimated)
                        end_idx = begin_idx + size
                        channel = vector[begin_idx:end_idx].reshape(shape_decimated)
                        wedge_channels.append(channel)
                        begin_idx = end_idx
                    # Stack channels along last axis: shape (*wedge_shape, num_channels)
                    wedge_array = torch.stack(wedge_channels, dim=-1)
                    coefficients[scale_idx][direction_idx].append(wedge_array)
        return coefficients

    def forward(self, image: torch.Tensor) -> UDCTCoefficients:
        """
        Apply forward curvelet transform.

        Parameters
        ----------
        image : torch.Tensor
            Input of shape ``(*batch, *self.shape)``: any number of leading batch
            dimensions (none for a single image) followed by the transform shape.

            - For transform_kind="real" or "monogenic": must be real-valued
            - For transform_kind="complex": can be real-valued or complex-valued

        Returns
        -------
        UDCTCoefficients
            Curvelet coefficients organized by scale, direction, and wedge, each
            of shape ``(*batch, *wedge_shape)``.
            For monogenic transforms, each coefficient tensor has shape
            ``(*wedge_shape, ndim+1)`` with channels stacked along the last axis.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import UDCT
        >>> transform = UDCT(shape=(32, 32), num_scales=3, wedges_per_direction=3)
        >>> batch = torch.randn(8, 32, 32)
        >>> coeffs = transform.forward(batch)
        >>> coeffs[0][0][0].shape[0]
        8
        >>> transform.backward(coeffs).shape
        torch.Size([8, 32, 32])
        """
        self._check_input(image)
        if self._transform_kind == "monogenic":
            return self._forward_monogenic(image)
        if self._cuda_graphs:
            return self._get_plan().unflatten(self._forward_flat(image))
        return self._get_plan().forward_nested(image)

    def _check_input(self, image: torch.Tensor) -> None:
        """Validate the shape, kind and precision of a forward-transform input."""
        self._check_shape(image)
        if self._transform_kind in ("real", "monogenic") and image.is_complex():
            msg = (
                f"{self._transform_kind.capitalize()} transform requires real-valued input. "
                "Got complex tensor. Use transform_kind='complex' for complex inputs."
            )
            raise ValueError(msg)
        check_dtype(image, self.dtype, "input")

    def _check_shape(self, image: torch.Tensor) -> None:
        shape = tuple(self._parameters.shape)
        if tuple(image.shape[image.ndim - len(shape) :]) != shape or image.ndim < len(
            shape
        ):
            msg = (
                f"Expected input of shape (*batch, {', '.join(map(str, shape))}), "
                f"got {tuple(image.shape)}."
            )
            raise ValueError(msg)

    def filter(
        self,
        image: torch.Tensor,
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward transform, apply ``fn`` to the coefficients, backward transform.

        The coefficients reach ``fn`` as one tensor of shape ``(*batch, ncoef)`` in
        :meth:`vect` order (use :meth:`struct` inside ``fn`` for per-wedge
        access), and never go through the nested list representation.

        Parameters
        ----------
        image : torch.Tensor
            Input of shape ``(*batch, *self.shape)``.
        fn : Callable[[torch.Tensor], torch.Tensor]
            Map from flattened coefficients to flattened coefficients of the same
            shape, e.g. a thresholding function.

        Returns
        -------
        torch.Tensor
            Filtered result of shape ``(*batch, *self.shape)``.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import UDCT
        >>> transform = UDCT(shape=(32, 32), num_scales=3, wedges_per_direction=3)
        >>> noisy = torch.randn(4, 32, 32)
        >>> def hard_threshold(c):
        ...     return c * (c.abs() > 0.5)
        >>> transform.filter(noisy, hard_threshold).shape
        torch.Size([4, 32, 32])
        """
        return self._backward_vect(fn(self._forward_vect(image)))

    def _get_plan(self) -> UDCTPlan:
        """Fused plan for the real/complex transform (built on first use)."""
        if self._plan is None:
            self._plan = UDCTPlan.build(
                self._parameters,
                self._windows,
                self._decimation_ratios_cpu,
                complex_mode=self._transform_kind == "complex",
            )
        return self._plan

    def _forward_vect(self, image: torch.Tensor) -> torch.Tensor:
        """Forward transform straight to the ``vect`` layout (used by UDCTModule)."""
        if self._transform_kind == "monogenic":
            return self.vect(self.forward(image))
        self._check_input(image)
        return self._forward_flat(image)

    def _backward_vect(self, vector: torch.Tensor) -> torch.Tensor:
        """Backward transform from the ``vect`` layout (used by UDCTModule)."""
        if self._transform_kind == "monogenic":
            return self.backward(self.struct(vector))
        check_dtype(vector, self.dtype, "coefficients")
        return self._backward_flat(vector)

    def _forward_flat(self, image: torch.Tensor) -> torch.Tensor:
        """Flat forward transform of checked input, from a CUDA graph when enabled."""
        plan = self._get_plan()
        if self._cuda_graphs:
            if graphable(image):
                return self._graphs.run("forward", plan.forward_flat, image)
            if graphable_with_adjoint(image):
                return ForwardWithAdjoint.apply(image, self)  # type: ignore[no-any-return]
        return plan.forward_flat(image)

    def _backward_flat(self, vector: torch.Tensor) -> torch.Tensor:
        """Flat backward transform of checked input, from a CUDA graph when enabled."""
        plan = self._get_plan()
        if self._cuda_graphs:
            if graphable(vector):
                return self._graphs.run("backward", plan.backward_flat, vector)
            if graphable_with_adjoint(vector):
                return BackwardWithAdjoint.apply(vector, self)  # type: ignore[no-any-return]
        return plan.backward_flat(vector)

    def _forward_monogenic(self, image: torch.Tensor) -> UDCTCoefficients:  # pylint: disable=unused-argument
        """Private method for monogenic forward transform (no input validation)."""
        # TODO: Implement monogenic forward transform for PyTorch
        # For now, raise NotImplementedError
        msg = "Monogenic transform not yet implemented for PyTorch"
        raise NotImplementedError(msg)

    def backward(self, coefficients: UDCTCoefficients) -> torch.Tensor:
        """
        Apply backward (inverse) curvelet transform.

        Parameters
        ----------
        coefficients : UDCTCoefficients
            Curvelet coefficients from forward transform, each of shape
            ``(*batch, *wedge_shape)``.

        Returns
        -------
        torch.Tensor
            Reconstruction of shape ``(*batch, *self.shape)``.
        """
        if self._transform_kind == "monogenic":
            return self._backward_monogenic(coefficients)
        plan = self._get_plan()
        # Checked after flattening, so mixed-precision wedges are caught too.
        flat = plan.flatten(coefficients)
        check_dtype(flat, self.dtype, "coefficients")
        return self._backward_flat(flat)

    def _backward_monogenic(self, coefficients: UDCTCoefficients) -> torch.Tensor:  # pylint: disable=unused-argument
        """Private method for monogenic backward transform (no input validation)."""
        # TODO: Implement monogenic backward transform for PyTorch
        # For now, raise NotImplementedError
        msg = "Monogenic transform not yet implemented for PyTorch"
        raise NotImplementedError(msg)

    def apply_to_tensors(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Apply a function to all internal tensors.

        This method is used for device and dtype transfers (e.g., when calling
        `model.to(device)` on a module containing this UDCT instance). It applies
        the given function to all internal tensor attributes including windows
        and decimation ratios. If ``fn`` changes the floating-point dtype of the
        windows, :attr:`dtype` follows it; a complex target dtype (e.g.
        ``module.to(torch.complex64)``) sets the matching real precision.

        Parameters
        ----------
        fn : Callable[[torch.Tensor], torch.Tensor]
            Function to apply to each tensor. Typically this is a function like
            `lambda t: t.cuda()` or `lambda t: t.to(device)`.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import UDCT
        >>> transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
        >>> # Move all internal tensors to GPU (if available)
        >>> if torch.cuda.is_available():
        ...     transform.apply_to_tensors(lambda t: t.cuda())  # doctest: +SKIP
        """
        # Apply to windows (nested structure of SparseWindow objects)
        for scale_idx, scale_windows in enumerate(self._windows):
            for dir_idx, dir_windows in enumerate(scale_windows):
                for wedge_idx, window in enumerate(dir_windows):
                    # apply() keeps the periodized index maps; rebuilding the
                    # window from indices/values would drop them and force a
                    # recomputation on every forward/backward call.
                    self._windows[scale_idx][dir_idx][wedge_idx] = window.apply(fn)

        # The fused plan holds copies of the window tensors; rebuild it lazily.
        # Captured graphs refer to the old plan's tensors.
        self._plan = None
        self._graphs.clear()

        # Apply to decimation_ratios (list of tensors)
        for i, ratio in enumerate(self._decimation_ratios):
            self._decimation_ratios[i] = fn(ratio)
