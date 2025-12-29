"""
PyTorch UDCT Module
===================

This example demonstrates the recommended way to use PyTorch bindings for the
Uniform Discrete Curvelet Transform (UDCT) via ``UDCTModule``.

PyTorch Bindings Architecture
------------------------------

The curvelets library provides two interfaces for PyTorch:

1. **UDCT**: Base class that provides forward and backward transforms but does
   not integrate with PyTorch's automatic differentiation system. It returns
   nested coefficient structures (lists of tensors organized by scale and
   direction).

2. **UDCTModule**: A ``nn.Module`` wrapper around ``UDCT`` that provides full
   autograd integration. It uses a custom ``torch.autograd.Function`` to enable
   automatic differentiation through the transform. When called, it returns
   flattened coefficients as a single tensor, making it PyTorch-friendly for
   use in neural networks and optimization workflows.

**Recommendation**: Use ``UDCTModule`` for all PyTorch workflows. It is the
recommended interface because it:
- Integrates seamlessly with PyTorch's autograd system
- Returns flattened tensors suitable for PyTorch operations
- Automatically uses the backward transform for gradient computation
- Can be used as a standard ``nn.Module`` in neural network architectures

How Autograd Integration Works
-------------------------------

``UDCTModule`` uses a custom autograd function that:

- **Forward pass**: Computes the forward curvelet transform and flattens the nested coefficients into a single tensor
- **Backward pass**: Automatically uses the backward transform to compute
  gradients with respect to the input image

This means you can use ``UDCTModule`` in any PyTorch computation graph, and
gradients will flow correctly through the transform.
"""

from __future__ import annotations

# %%
import torch

from curvelets.torch import UDCTModule

# %%
# PyTorch Bindings: UDCT vs UDCTModule
# ####################################
#
# The curvelets library provides two PyTorch interfaces:
#
# **UDCT** (not recommended for most PyTorch workflows):
#   - Base transform class without autograd integration
#   - Returns nested coefficient structures (lists of tensors)
#   - Requires manual gradient handling if used in optimization
#   - Useful for inspection/debugging or when you need nested structures
#
# **UDCTModule** (recommended for PyTorch workflows):
#   - ``nn.Module`` wrapper with full autograd support
#   - Returns flattened coefficients as a single tensor
#   - Automatically integrates with PyTorch's computation graph
#   - Uses backward transform for gradient computation
#   - Can be used directly in neural networks and optimization loops
#
# When to use each:
#   - Use **UDCTModule** for: neural networks, optimization, any workflow
#     requiring gradients, standard PyTorch tensor operations
#   - Use **UDCT** for: inspection of nested structures, debugging,
#     when you specifically need the nested coefficient format
#
# This example demonstrates ``UDCTModule``, which is the recommended interface
# for all PyTorch use cases.

# %%
# Setup
# #####
shape = (32, 32)
angular_wedges_config = torch.tensor([[3, 3]])
udct_module = UDCTModule(
    shape=shape,
    angular_wedges_config=angular_wedges_config,
)

# %%
# Forward Transform
# #################
# UDCTModule returns flattened coefficients as a single tensor, making it
# compatible with standard PyTorch operations. The nested structure is
# automatically flattened during the forward pass.
input_tensor = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
output = udct_module(input_tensor)
# Input shape: input_tensor.shape
# Output shape: output.shape
# Note: Output is a flattened tensor, not a nested structure

# %%
# Reconstruction via Autograd
# ############################
# UDCTModule's autograd integration works as follows:
#
# - Forward pass: Uses forward transform to compute coefficients
# - Backward pass: Automatically uses backward transform to compute gradients
#
# This happens transparently through PyTorch's autograd system.
#
# Compute a simple operation on the coefficients (use abs to ensure real scalar)
loss = torch.abs(output).sum()
# Backward pass: The custom autograd function automatically applies the
# backward transform to compute gradients w.r.t. the input
loss.backward()
# The gradients in input_tensor.grad demonstrate the backward transform is working
grad = input_tensor.grad
assert grad is not None
# Gradient shape: grad.shape
# Gradient norm: grad.norm().item()
# The backward transform is automatically used in the autograd graph!

# Verify reconstruction matches input
# Get nested coefficients and reconstruct using backward transform
coeffs_nested = udct_module.struct(output.detach())
reconstructed = udct_module._udct.backward(coeffs_nested)
reconstruction_error = torch.abs(input_tensor.detach() - reconstructed).max()
# Reconstruction error: reconstruction_error.item()
assert torch.allclose(input_tensor.detach(), reconstructed, atol=1e-4), (
    f"Reconstruction does not match input! Max error: {reconstruction_error.item():.2e}"
)
# Reconstruction matches input tensor!

# %%
# Using struct() Method
# #####################
# While UDCTModule returns flattened coefficients (PyTorch-friendly), you can
# convert them back to the nested structure when needed using struct().
# This is useful for inspection, visualization, or when working with code that
# expects nested coefficient structures.
#
# Convert flattened coefficients back to nested structure
coeffs_nested_from_struct = udct_module.struct(output.detach())
# Flattened coefficients shape: output.shape
# Restructured to nested format with len(coeffs_nested_from_struct) scales
# struct() converts flattened coefficients to nested structure using internal state

# %%
# Gradcheck Verification
# ######################
# PyTorch's gradcheck verifies that the backward pass correctly computes
# gradients. This confirms that UDCTModule's autograd integration is working
# correctly - the backward transform is properly used for gradient computation.
#
# Clear gradients for gradcheck
input_tensor.grad = None
result = torch.autograd.gradcheck(
    udct_module,
    input_tensor,
    fast_mode=True,
    atol=1e-4,
    rtol=1e-3,
)
assert result, "Gradcheck failed"
# This confirms the autograd integration is working correctly!
