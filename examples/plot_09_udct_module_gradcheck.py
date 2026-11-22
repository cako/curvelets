"""
PyTorch UDCT Module
===================

``curvelets.torch.UDCT`` is a :class:`torch.nn.Module`: it moves with
``.to()``, nests inside networks, and is differentiable.

Its gradients come from the adjoint transform. The UDCT :cite:`Nguyen2010` is a
tight frame, so
the gradient of the forward transform is the backward transform (and vice
versa). Autograd never records the many small operations inside the
transform, which keeps gradients fast and memory-light.

Two layouts are available:

- **Nested** (``udct(x)`` / :meth:`~curvelets.torch.UDCT.forward` and
  :meth:`~curvelets.torch.UDCT.backward`): lists of tensors indexed by scale,
  direction and wedge, as in ``curvelets.numpy``.
- **Flat** (:meth:`~curvelets.torch.UDCT.forward` with ``structured=False``,
  and :meth:`~curvelets.torch.UDCT.backward` on the resulting tensor): one tensor of shape
  ``(*batch, ncoef)`` in :meth:`~curvelets.torch.UDCT.vect` order, which suits
  networks and optimizers.

Both layouts are differentiable.
"""

from __future__ import annotations

# %%
import torch

from curvelets.torch import UDCT

# %%
# Setup
# #####
shape = (32, 32)
udct = UDCT(
    shape=shape,
    angular_wedges_config=torch.tensor([[3, 3]]),
    dtype=torch.float64,  # gradcheck needs double precision
)
assert isinstance(udct, torch.nn.Module)

# %%
# Flat Coefficients and Gradients
# ###############################
# ``forward(x, structured=False)`` returns all coefficients as one tensor. A loss on them
# backpropagates to the input through the backward transform.
x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
coeffs = udct.forward(x, structured=False)
loss = coeffs.abs().sum()
loss.backward()
assert x.grad is not None

# %%
# Nested Coefficients
# ###################
# Calling the module returns the nested structure, which is differentiable as
# well. ``struct`` and ``vect`` convert between the two layouts.
nested = udct(x)
torch.testing.assert_close(udct.vect(nested), coeffs)
reconstructed = udct.backward(udct.struct(coeffs.detach()))
assert torch.allclose(x.detach(), reconstructed, atol=1e-4)

# %%
# Gradcheck Verification
# ######################
# :func:`torch.autograd.gradcheck` compares the adjoint-based gradients with
# finite differences.
x.grad = None
assert torch.autograd.gradcheck(
    (lambda t: udct.forward(t, structured=False)),
    x,
    fast_mode=True,
    atol=1e-4,
    rtol=1e-3,
)

# %%
# References
# ##########
#
# .. bibliography::
#    :filter: docname in docnames
