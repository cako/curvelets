"""Deprecated :class:`UDCTModule`: :class:`UDCT` is an ``nn.Module`` itself."""

from __future__ import annotations

import warnings
from typing import Literal

import torch

from ._udct import UDCT


class UDCTModule(UDCT):
    """
    Deprecated: use :class:`UDCT`, which is a :class:`torch.nn.Module`.

    .. deprecated:: 1.3
        :class:`UDCT` now is an ``nn.Module`` with adjoint gradients.
        ``UDCTModule(shape, ..., transform_type=kind)`` returned flat
        coefficients from ``__call__``; the replacement is
        ``UDCT(shape, ..., transform_kind=kind).forward_vect(x)`` (or
        ``udct(x)`` for nested coefficients). ``UDCTModule`` will be removed
        in a future release.

    Parameters are those of :class:`UDCT`, with ``transform_type`` (``"real"``
    or ``"complex"``) in place of ``transform_kind``. Calling the module
    returns flattened coefficients of shape ``(*batch, ncoef)``.
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
        transform_type: Literal["real", "complex"] = "real",
        dtype: torch.dtype | None = None,
        cuda_graphs: bool = False,
    ) -> None:
        warnings.warn(
            "UDCTModule is deprecated: UDCT is an nn.Module now. Use "
            "UDCT(..., transform_kind=...) and udct.forward_vect(x) for flat "
            "coefficients.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            shape=shape,
            angular_wedges_config=angular_wedges_config,
            num_scales=num_scales,
            wedges_per_direction=wedges_per_direction,
            window_overlap=window_overlap,
            radial_frequency_params=radial_frequency_params,
            window_threshold=window_threshold,
            high_frequency_mode=high_frequency_mode,
            transform_kind=transform_type,
            dtype=dtype,
            cuda_graphs=cuda_graphs,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Flattened coefficients of shape ``(*batch, ncoef)`` (see :meth:`UDCT.forward_vect`)."""
        return self.forward_vect(image)
