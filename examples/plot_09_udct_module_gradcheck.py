"""
PyTorch UDCT Module
===================
"""

from __future__ import annotations

# %%
import torch

from curvelets.torch import UDCTModule

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
input_tensor = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
output = udct_module(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")

# %%
# Reconstruction via Autograd
# ############################
# The backward transform is automatically used when computing gradients
# Compute a simple operation on the coefficients (use abs to ensure real scalar)
loss = torch.abs(output).sum()
# Backward pass: gradients are computed using the backward transform
loss.backward()
# The gradients in input_tensor.grad demonstrate the backward transform is working
grad = input_tensor.grad
assert grad is not None
print(f"Gradient shape: {grad.shape}")
print(f"Gradient norm: {grad.norm().item():.2e}")
print("The backward transform is automatically used in the autograd graph!")

# Verify reconstruction matches input
# Get nested coefficients and reconstruct using backward transform
coeffs_nested = udct_module.struct(output.detach())
reconstructed = udct_module._udct.backward(coeffs_nested)
reconstruction_error = torch.abs(input_tensor.detach() - reconstructed).max()
print(f"Reconstruction error: {reconstruction_error.item():.2e}")
assert torch.allclose(input_tensor.detach(), reconstructed, atol=1e-4), (
    f"Reconstruction does not match input! Max error: {reconstruction_error.item():.2e}"
)
print("Reconstruction matches input tensor!")

# %%
# Using struct() Method
# #####################
# struct() can be used to convert flattened coefficients back to nested structure
# Use struct() to convert flattened coefficients to nested structure
coeffs_nested_from_struct = udct_module.struct(output.detach())
print(f"Flattened coefficients shape: {output.shape}")
print(f"Restructured to nested format with {len(coeffs_nested_from_struct)} scales")
print("struct() converts flattened coefficients to nested structure using internal state")

# %%
# Gradcheck Verification
# ######################
# Clear gradients for gradcheck
input_tensor.grad = None
result = torch.autograd.gradcheck(
    udct_module,
    input_tensor,
    fast_mode=True,
    atol=1e-4,
    rtol=1e-3,
)
print(f"Gradcheck passed: {result}")
