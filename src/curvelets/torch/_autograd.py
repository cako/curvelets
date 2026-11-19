"""Autograd Functions that use the adjoint transform as the gradient.

The UDCT is a tight frame: the backward (inverse) transform is the adjoint of
the forward transform, and vice versa. So the gradient of either transform is
the other transform applied to the incoming gradient, with no autograd graph
through the fused plan. Both directions are then plain calls without
gradients, which :class:`~curvelets.torch.UDCTModule` relies on for speed
and ``UDCT(cuda_graphs=True)`` for replaying both from CUDA graphs.
"""

from __future__ import annotations

import typing

import torch

if typing.TYPE_CHECKING:
    from ._udct import UDCT


class ForwardWithAdjoint(torch.autograd.Function):  # type: ignore[misc]  # pylint: disable=abstract-method
    """Flat forward transform; its gradient is the backward transform."""

    # Automatic vmap rule, for functorch transforms (vmap, grad, jacrev).
    generate_vmap_rule = True

    @staticmethod
    def forward(image: torch.Tensor, udct: UDCT) -> torch.Tensor:  # pylint: disable=arguments-differ
        """
        Forward transform to the flat ``vect`` layout.

        Parameters
        ----------
        image : torch.Tensor
            Input of shape ``(*batch, *udct.shape)``.
        udct : UDCT
            The transform.

        Returns
        -------
        torch.Tensor
            Flattened coefficients.
        """
        return udct._forward_flat(image)  # pylint: disable=protected-access

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[torch.Tensor, UDCT],
        output: torch.Tensor,  # noqa: ARG004
    ) -> None:
        """Keep the transform for the backward pass."""
        ctx.udct = inputs[1]  # ty: ignore[unresolved-attribute]

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """Gradient w.r.t. the image: the backward transform of the gradient."""
        udct = ctx.udct  # ty: ignore[unresolved-attribute]
        return udct._backward_flat(grad_outputs[0]), None  # pylint: disable=protected-access


class BackwardWithAdjoint(torch.autograd.Function):  # type: ignore[misc]  # pylint: disable=abstract-method
    """Flat backward transform; its gradient is the forward transform."""

    generate_vmap_rule = True

    @staticmethod
    def forward(vector: torch.Tensor, udct: UDCT) -> torch.Tensor:  # pylint: disable=arguments-differ
        """
        Backward transform of flat ``vect``-layout coefficients.

        Parameters
        ----------
        vector : torch.Tensor
            Coefficients of shape ``(*batch, ncoef)``.
        udct : UDCT
            The transform.

        Returns
        -------
        torch.Tensor
            Reconstruction of shape ``(*batch, *udct.shape)``.
        """
        return udct._backward_flat(vector)  # pylint: disable=protected-access

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[torch.Tensor, UDCT],
        output: torch.Tensor,  # noqa: ARG004
    ) -> None:
        """Keep the transform for the backward pass."""
        ctx.udct = inputs[1]  # ty: ignore[unresolved-attribute]

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """Gradient w.r.t. the coefficients: the forward transform of the gradient."""
        udct = ctx.udct  # ty: ignore[unresolved-attribute]
        return udct._forward_flat(grad_outputs[0]), None  # pylint: disable=protected-access
