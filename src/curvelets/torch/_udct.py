"""Main UDCT class for PyTorch implementation."""

# pylint: disable=duplicate-code,too-many-lines
# Duplicate code with numpy implementation is expected
from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from math import sqrt
from typing import Any, Literal, Self, overload

import numpy as np
import torch
from torch import nn

from ._cuda_graphs import CudaGraphCache, CudaGraphsMixin, _may_need_grad
from ._dtypes import check_dtype, resolve_dtype
from ._monogenic import (
    MonogenicLayout,
    channels_from_real,
    real_from_channels,
    riesz_images,
    riesz_spectra,
)
from ._plan import UDCTPlan
from ._udct_windows import UDCTWindow
from ._utils import ParamUDCT
from .typing import UDCTCoefficients, UDCTWindows

# Largest ``window_overlap`` whose wedges still fit their decimated grids.
# A wedge at scale s (N_s wedges, radial outer edge 4*pi/3 / 2**(S-1-s) with the
# default radial parameters) spans (2/N_s)(1+2a) of its outer radius along the
# transverse axis, i.e. 8/9 (1+2a) decimated bins (d = 2 N_s 2**(S-1-s) / 3).
# That is at most one bin, so the periodized fold is injective, iff a <= 1/16,
# independent of shape, scales and wedge count. Beyond it tail samples alias and
# the round trip loses precision (~1e-9 for ``wedges_per_direction=6`` at the
# former default of 0.09).
_MAX_ALIASING_FREE_OVERLAP = 1 / 16


class UDCT(CudaGraphsMixin, nn.Module):  # type: ignore[misc]
    """
    Uniform Discrete Curvelet Transform (UDCT) implementation.

    Forward and backward curvelet transforms (real, complex or monogenic) as
    a :class:`torch.nn.Module` with no learnable parameters. Calling the module
    (``udct(x)``) is :meth:`forward`; ``forward(x, structured=False)`` and
    :meth:`backward` of a flat tensor work on the :meth:`vect` layout instead
    of nested lists, which suits networks and optimizers.

    Gradients of every transform come from the adjoint transform: the UDCT is
    a tight frame, so the gradient of the forward transform is the backward
    transform and vice versa, with no autograd graph through the
    computation. Second derivatives, ``torch.func`` transforms and
    ``torch.compile`` are supported. ``.to(device)``, ``.cuda()``,
    ``.double()`` and ``.float()`` move or recast the windows; they are not
    buffers, so :meth:`state_dict` stays empty (they are rebuilt from the
    constructor arguments).

    .. versionchanged:: 1.3
       ``UDCT`` is a :class:`torch.nn.Module`, and its gradients come from the
       adjoint transform in both layouts. It replaces the deprecated
       :class:`UDCTModule`.

    .. versionchanged:: 1.3
       Every transform accepts leading batch dimensions, so
       :func:`torch.vmap` is no longer needed.

    .. versionchanged:: 1.3
       The transform has a fixed precision (``dtype``). Inputs and
       coefficients of another precision raise a :class:`TypeError` instead
       of being cast.

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
        Window overlap parameter :math:`\\alpha` controlling the smoothness of
        window transitions. If None, automatically chosen as 10% of the
        Nguyen & Chauris (2010) upper bound, capped at :math:`1/16` (0.0625).
        With the default ``radial_frequency_params``, a wedge at scale
        :math:`s` spans :math:`\\frac{8}{9}(1 + 2\\alpha)` decimated bins along
        the transverse axis, so its periodized fold stays injective
        (aliasing-free) iff :math:`\\alpha \\le 1/16`, independent of shape,
        scales, and wedge count. Passing a larger value logs a warning when
        aliasing occurs. Default is None (auto).

        .. versionchanged:: 1.3
           Auto-selected ``window_overlap`` is now capped at :math:`1/16` so
           ``wedges_per_direction >= 6`` reconstructs to machine precision, and
           explicit overlaps that alias log a warning.
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
          by applying Riesz transforms. Each band has ndim+2 real channels
          along its last axis: the complex scalar coefficient (real and
          imaginary parts) and one Riesz component per dimension. The
          backward transform returns ``ndim + 1`` components stacked along a
          new first axis: :math:`f`, then approximations of :math:`-R_k f`
          (see :meth:`backward`). The results match
          :class:`curvelets.numpy.UDCT`.

        .. versionchanged:: 1.3
           ``"monogenic"`` is implemented (it raised ``NotImplementedError``
           before).

    dtype : torch.dtype, optional
        Precision of the transform: ``torch.float32`` or ``torch.float64``
        (``torch.complex64`` / ``torch.complex128`` are accepted and mean the
        same precision). The windows are computed in double precision and
        stored once in this dtype. Inputs must have this precision (or its
        complex counterpart for the complex transform), and coefficients come
        out as :attr:`coefficient_dtype`. Default is
        :func:`torch.get_default_dtype` (normally ``torch.float32``).

        .. versionadded:: 1.3
    cuda_graphs : bool, optional
        Replay the transforms on CUDA from CUDA graphs: one
        launch per call instead of a few dozen, captured on the first call for
        each input shape and dtype (the last :attr:`cuda_graph_cache_size` are
        kept). Covers every forward/backward entry point, :meth:`filter`
        and gradients (through the adjoint transform).
        Worthwhile for small and medium inputs, where launches dominate (a
        512² transform gets ~6x faster on an RTX 4090); slower for large
        inputs, where copying in and out of the graph buffers costs more.
        Default is False.

        .. versionadded:: 1.3

    Attributes
    ----------
    dtype : torch.dtype
        Real precision of the transform.
    cuda_graphs : bool
        Whether CUDA calls replay CUDA graphs (can be switched at any time).
    cuda_graph_cache_size : int
        Number of captured graphs kept (least recently used dropped first).
    coefficient_dtype : torch.dtype
        Dtype of the coefficients (complex for the real and complex transforms,
        real for the monogenic transform).
    high_frequency_mode : str
        High frequency mode.
    transform_kind : str
        Type of transform being used ("real", "complex", or "monogenic").

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
    >>> # Flat coefficients and gradients, e.g. inside a network
    >>> x = torch.randn(4, 64, 64, requires_grad=True)
    >>> features = transform.forward(x, structured=False)  # shape (4, ncoef)
    >>> features.abs().pow(2).sum().backward()
    >>> x.grad.shape
    torch.Size([4, 64, 64])
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
        super().__init__()
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
        self._params = ParamUDCT(
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
        # Monogenic coefficient layout and Riesz filters, built on first use.
        self._monogenic_layout: MonogenicLayout | None = None
        self._riesz: torch.Tensor | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the transform."""
        return self._params.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._params.ndim

    @property
    def num_scales(self) -> int:
        """Number of scales."""
        return self._params.num_scales

    @property
    def dtype(self) -> torch.dtype:
        """Real precision of the transform (follows ``.to(dtype)``).

        .. versionadded:: 1.3
        """
        return self._windows[0][0][0].values.dtype

    @property
    def coefficient_dtype(self) -> torch.dtype:
        """Dtype of the coefficients: complex, or real for the monogenic transform.

        .. versionadded:: 1.3
        """
        if self._transform_kind == "monogenic":
            return self.dtype
        return self.dtype.to_complex()

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
        The result is capped at ``_MAX_ALIASING_FREE_OVERLAP`` (:math:`1/16`):
        with the default ``radial_frequency_params``, a wedge spans
        :math:`\\frac{8}{9}(1 + 2\\alpha)` decimated bins along its transverse
        axis, which fits within one bin (keeping the periodized fold injective)
        iff :math:`\\alpha \\le 1/16`, independent of shape, scales, and wedge
        count. Above :math:`1/16`, tail samples alias and the transform stops
        reconstructing to machine precision.

        Parameters
        ----------
        wedges_per_scale : torch.Tensor
            Tensor of wedge counts per scale.

        Returns
        -------
        float
            Optimal window_overlap value (10% of theoretical maximum, capped
            at ``_MAX_ALIASING_FREE_OVERLAP``).

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
        >>> round(alpha, 4)
        0.0369
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

        # Use 10% of theoretical max for optimal reconstruction quality, but
        # never beyond the point where wedge supports outgrow their decimated
        # grids (see _MAX_ALIASING_FREE_OVERLAP)
        return min(0.1 * min_overlap, _MAX_ALIASING_FREE_OVERLAP)

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
        window_computer = UDCTWindow(self._params, self._high_frequency_mode)
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
        >>> shapes[0][0][0]  # Low-frequency band shape (decimated by 2)
        (32, 32)
        """
        if self._transform_kind == "monogenic":
            return self._coefficient_shapes_monogenic()
        if self._transform_kind == "complex":
            return self._coefficient_shapes_complex()
        return self._coefficient_shapes_real()

    def _coefficient_shapes_real(self) -> list[list[list[tuple[int, ...]]]]:
        """Private method for real transform coefficient shapes calculation."""
        shapes: list[list[list[tuple[int, ...]]]] = []
        internal_shape = torch.tensor(self._params.shape, dtype=torch.int64)
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
        internal_shape = torch.tensor(self._params.shape, dtype=torch.int64)
        for scale_idx, decimation_ratios_scale in enumerate(
            self._decimation_ratios_cpu
        ):
            shapes.append([])
            if scale_idx > 0 and len(self._windows[scale_idx]) > 1:
                num_directions = 2 * self._params.ndim
            else:
                num_directions = len(decimation_ratios_scale)
            for direction_idx in range(num_directions):
                shapes[scale_idx].append([])
                if scale_idx > 0 and direction_idx >= self._params.ndim:
                    window_direction_idx = direction_idx % self._params.ndim
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
        internal_shape = torch.tensor(self._params.shape, dtype=torch.int64)
        num_channels = self._params.ndim + 2
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

        .. versionchanged:: 1.3
           Coefficients with leading batch dimensions are supported.

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
        return self._get_monogenic_layout(coefficients[0][0][0].device).flatten(
            coefficients
        )

    def struct(self, vector: torch.Tensor) -> UDCTCoefficients:
        """
        Restructure vectorized coefficients to nested list format.

        .. versionchanged:: 1.3
           Leading batch dimensions are supported, and the wedges are views
           into ``vector`` in its dtype (previously the dtype of the last
           forward call).

        Parameters
        ----------
        vector : torch.Tensor
            Coefficients of shape ``(*batch, ncoef)``.

        Returns
        -------
        UDCTCoefficients
            Restructured coefficients, each of shape ``(*batch, *wedge_shape)``
            (``(*batch, *wedge_shape, ndim + 2)`` for the monogenic transform),
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
        return self._get_monogenic_layout(vector.device).unflatten(vector)

    @overload
    def forward(
        self, image: torch.Tensor, structured: Literal[True] = True
    ) -> UDCTCoefficients: ...

    @overload
    def forward(
        self, image: torch.Tensor, structured: Literal[False]
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self, image: torch.Tensor, structured: bool
    ) -> UDCTCoefficients | torch.Tensor: ...

    def forward(
        self, image: torch.Tensor, structured: bool = True
    ) -> UDCTCoefficients | torch.Tensor:
        """
        Apply forward curvelet transform.

        .. versionchanged:: 1.3
           Leading batch dimensions are supported, and ``structured=False``
           returns the flat :meth:`vect` layout directly.

        Parameters
        ----------
        image : torch.Tensor
            Input of shape ``(*batch, *self.shape)``: any number of leading batch
            dimensions (none for a single image) followed by the transform shape.

            - For transform_kind="real" or "monogenic": must be real-valued
            - For transform_kind="complex": can be real-valued or complex-valued
        structured : bool, default True
            If True, return the nested coefficients. If False, return the flat
            tensor that :meth:`vect` would produce, without building the nested
            list (``forward(x, structured=False)`` equals
            ``vect(forward(x))``); :meth:`backward` accepts it as is.

            .. versionadded:: 1.3

        Returns
        -------
        UDCTCoefficients | torch.Tensor
            With ``structured=False``, a tensor of shape ``(*batch, ncoef)``.
            Otherwise curvelet coefficients organized by scale, direction, and wedge, each
            of shape ``(*batch, *wedge_shape)``.
            For monogenic transforms, each coefficient tensor is real with shape
            ``(*batch, *wedge_shape, ndim+2)``, channels
            ``[scalar.real, scalar.imag, riesz_1, ..., riesz_ndim]`` along the
            last axis.

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
        >>> flat = transform.forward(batch, structured=False)
        >>> torch.allclose(flat, transform.vect(coeffs))
        True
        """
        self._check_input(image)
        if not structured:
            if self._transform_kind == "monogenic":
                return self._forward_monogenic_flat(image)
            return self._forward_flat(image)
        if self._transform_kind == "monogenic":
            return self._struct_monogenic(self._forward_monogenic_flat(image))
        if self._cuda_graphs or _may_need_grad(image):
            # Views into the flat coefficients; gradients through the adjoint.
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
        shape = tuple(self._params.shape)
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

        .. versionadded:: 1.3

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
            Filtered result of shape ``(*batch, *self.shape)``; for the
            monogenic transform, ``(ndim + 1, *batch, *self.shape)`` (see
            :meth:`backward`).

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
        return self.backward(fn(self.forward(image, structured=False)))

    def _get_plan(self) -> UDCTPlan:
        """Fused plan of the real/complex transform (built on first use).

        The monogenic transform runs on the real transform's plan.

        Also rebuilt when the windows are on another device than the plan was
        built for, e.g. after ``torch.load(..., map_location=...)``, which moves
        tensors without going through :meth:`apply_to_tensors`.
        """
        plan = self._plan
        if plan is None or plan.device != self._windows[0][0][0].values.device:
            self._graphs.clear()
            plan = self._plan = UDCTPlan.build(
                self._params,
                self._windows,
                self._decimation_ratios_cpu,
                complex_mode=self._transform_kind == "complex",
            )
        return plan

    def forward_vect(self, image: torch.Tensor) -> torch.Tensor:
        """
        Deprecated alias of ``forward(image, structured=False)``.

        .. deprecated:: 1.3
           Use ``forward(image, structured=False)``.
        """
        warnings.warn(
            "forward_vect is deprecated: use forward(image, structured=False).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.forward(image, structured=False)

    def backward_vect(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Deprecated alias of ``backward(vector)``.

        .. deprecated:: 1.3
           :meth:`backward` accepts the flat tensor directly.
        """
        warnings.warn(
            "backward_vect is deprecated: backward accepts the flat tensor.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.backward(vector)

    def _get_monogenic_layout(self, device: torch.device) -> MonogenicLayout:
        """Monogenic coefficient layout, with its index maps on ``device``."""
        if self._monogenic_layout is None:
            self._monogenic_layout = MonogenicLayout.build(
                self._get_plan(), self._params.ndim + 2
            )
        self._monogenic_layout = self._monogenic_layout.to(device)
        return self._monogenic_layout

    def _get_riesz(self, device: torch.device) -> torch.Tensor:
        """Riesz filters ``(ndim, *shape)`` on ``device`` (built on first use)."""
        if self._riesz is None or self._riesz.device != device:
            self._riesz = riesz_spectra(self._params.shape, self.dtype, device)
        return self._riesz

    def _forward_monogenic_flat(self, image: torch.Tensor) -> torch.Tensor:
        """Monogenic forward transform of checked input to the :meth:`vect` layout.

        One batched real transform of ``f`` and its Riesz images (see
        :mod:`curvelets.torch._monogenic`); gradients flow through the adjoint
        real transform.
        """
        images, even = riesz_images(image, self._get_riesz(image.device))
        coefficients = self._forward_flat(images)
        channels = channels_from_real(coefficients, self._params.ndim, even)
        layout = self._get_monogenic_layout(image.device)
        return channels.flatten(-2)[..., layout.perm]

    def _backward_monogenic_flat(self, vector: torch.Tensor) -> torch.Tensor:
        """Monogenic backward transform of :meth:`vect`-layout coefficients."""
        if vector.is_complex():
            msg = "Monogenic coefficients must be real-valued, got a complex tensor."
            raise ValueError(msg)
        layout = self._get_monogenic_layout(vector.device)
        if vector.shape[-1] != layout.ncoef:
            msg = (
                f"Expected monogenic coefficients of shape (*batch, {layout.ncoef}), "
                f"got {tuple(vector.shape)}."
            )
            raise ValueError(msg)
        channels = vector[..., layout.inverse].unflatten(
            -1, (layout.num_channels, self._get_plan().ncoef)
        )
        images = self._backward_flat(real_from_channels(channels))
        # (*batch, ndim + 1, *shape) -> (ndim + 1, *batch, *shape), Riesz negated.
        images = images.movedim(-self._params.ndim - 1, 0)
        return torch.cat([images[:1], -images[1:]])

    def backward(self, coefficients: UDCTCoefficients | torch.Tensor) -> torch.Tensor:
        """
        Apply backward (inverse) curvelet transform.

        .. versionchanged:: 1.3
           Leading batch dimensions are supported, and the flat tensor of
           ``forward(x, structured=False)`` is accepted.

        Parameters
        ----------
        coefficients : UDCTCoefficients | torch.Tensor
            Curvelet coefficients from forward transform, each of shape
            ``(*batch, *wedge_shape)``, or the flat tensor of shape
            ``(*batch, ncoef)`` from ``forward(x, structured=False)`` or
            :meth:`vect`.

        Returns
        -------
        torch.Tensor
            Reconstruction of shape ``(*batch, *self.shape)``.
            For the monogenic transform, a real tensor of shape
            ``(ndim + 1, *batch, *self.shape)``: the components that
            :class:`curvelets.numpy.UDCT` returns as a tuple, stacked along a
            new first axis (so ``f, *riesz = udct.backward(coeffs)`` works as
            in numpy). The first is the reconstruction of :math:`f`; the others
            are the synthesis of each Riesz channel, which approximates
            :math:`-R_k f` (see :meth:`monogenic` for the exact values). As in
            numpy, each Riesz channel only keeps the real part of its
            coefficients, so these components are not exact.
        """
        if isinstance(coefficients, torch.Tensor):
            check_dtype(coefficients, self.dtype, "coefficients")
            if self._transform_kind == "monogenic":
                return self._backward_monogenic_flat(coefficients)
            return self._backward_flat(coefficients)
        if self._transform_kind == "monogenic":
            flat = self._vect_monogenic(coefficients)
            check_dtype(flat, self.dtype, "coefficients")
            return self._backward_monogenic_flat(flat)
        plan = self._get_plan()
        # Checked after flattening, so mixed-precision wedges are caught too.
        flat = plan.flatten(coefficients)
        check_dtype(flat, self.dtype, "coefficients")
        return self._backward_flat(flat)

    def monogenic(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute the monogenic signal directly, without the curvelet transform.

        Returns :math:`(f, -R_1 f, \\ldots, -R_n f)`, where :math:`R_k` are the
        Riesz transforms, computed exactly in the Fourier domain. Available
        for every ``transform_kind``.

        .. versionadded:: 1.3

        Parameters
        ----------
        image : torch.Tensor
            Real input of shape ``(*batch, *self.shape)`` and precision
            :attr:`dtype`.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``(ndim + 1, *batch, *self.shape)``: ``image``,
            then :math:`-R_k f` for each dimension (the tuple that
            :meth:`curvelets.numpy.UDCT.monogenic` returns, stacked along a new
            first axis).

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import UDCT
        >>> transform = UDCT(shape=(64, 64), transform_kind="monogenic")
        >>> data = torch.randn(64, 64)
        >>> scalar, riesz1, riesz2 = transform.monogenic(data)  # shape (3, 64, 64)
        >>> torch.equal(scalar, data)
        True
        >>> recon = transform.backward(transform.forward(data))
        >>> torch.allclose(recon[0], data, atol=1e-4)
        True
        """
        self._check_shape(image)
        if image.is_complex():
            msg = (
                "Monogenic transform requires real-valued input. "
                "Got complex tensor. Use forward() for complex inputs."
            )
            raise ValueError(msg)
        check_dtype(image, self.dtype, "input")
        ndim = self._params.ndim
        dims = tuple(range(-ndim, 0))
        spectrum = torch.fft.fftn(image, dim=dims).unsqueeze(-ndim - 1)  # pylint: disable=not-callable
        riesz = torch.fft.ifftn(spectrum * self._get_riesz(image.device), dim=dims)  # pylint: disable=not-callable
        riesz = -riesz.real.movedim(-ndim - 1, 0)
        return torch.cat([image.unsqueeze(0), riesz])

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> Self:
        """Apply ``fn`` to the windows too (``.to()``, ``.cuda()``, ``.double()``)."""
        super()._apply(fn, recurse)
        self.apply_to_tensors(fn)
        return self

    def apply_to_tensors(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Apply a function to all internal tensors.

        Called by ``.to()``, ``.cuda()``, ``.double()`` and the like (prefer
        those). It applies
        the given function to all internal tensor attributes including windows
        and decimation ratios. If ``fn`` changes the floating-point dtype of the
        windows, :attr:`dtype` follows it; a complex target dtype (e.g.
        ``udct.to(torch.complex64)``) sets the matching real precision.

        .. versionchanged:: 1.3
           Dtype changes set the transform's precision, and the periodized
           index maps are kept instead of recomputed.

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
        # Captured graphs refer to the old plan's tensors. The Riesz filters
        # follow the device and dtype on their next use.
        self._plan = None
        self._graphs.clear()
        self._monogenic_layout = None
        self._riesz = None

        # Apply to decimation_ratios (list of tensors)
        for i, ratio in enumerate(self._decimation_ratios):
            self._decimation_ratios[i] = fn(ratio)
