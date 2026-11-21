"""Deprecated :class:`UDCTModule`: :class:`UDCT` is an ``nn.Module`` itself."""

from __future__ import annotations

import warnings
from typing import Any, Literal

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
        *args: Any,
        transform_type: Literal["real", "complex"] = "real",
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            "UDCTModule is deprecated: UDCT is an nn.Module now. Use "
            "UDCT(..., transform_kind=...) and udct.forward_vect(x) for flat "
            "coefficients.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs["transform_kind"] = transform_type
        super().__init__(*args, **kwargs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Flattened coefficients of shape ``(*batch, ncoef)`` (see :meth:`UDCT.forward_vect`)."""
        return self.forward_vect(image)
