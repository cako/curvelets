r"""
Curvelet-Domain Hessian Action
==============================

This example showcases how the curvelet transform can be used to estimate and
compensate for the action of the Hessian in Least-Squares Migration (LSM), inspired by
:cite:t:`Wang2016,Wang2017`.

The Gauss-Newton Hessian :math:`J^\dagger J` acts as a spatially varying, directionally
dependent blurring operator (a dip filter). By analyzing the Hessian-vector product
on a point scatterer (which acts as a Point Spread Function), we can see that the
blurring is highly anisotropic.

The curvelet transform, which localizes signals in both position and direction,
is perfectly suited to diagonalize this operator. In the curvelet domain, the Hessian
acts approximately as a scalar multiplier on each coefficient, allowing us to build
a Curvelet-domain Hessian Filter (CHF) to invert it.
"""

# sphinx_gallery_thumbnail_number = 2
# (dummy comment to trigger sphinx-gallery rebuild after testdata update)

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from curvelets.numpy import UDCT
from curvelets.plot import create_colorbar, despine

# %%
# Load the Hessian-Vector Product Workspace
# -----------------------------------------
# We load a pre-computed workspace containing a point scatterer perturbation (:math:`\delta m`)
# and the Gauss-Newton Hessian action on it (:math:`h_{gn}`).

workspace_path = Path("testdata/hessian_workspace.npz")
if not workspace_path.exists():
    workspace_path = Path("../testdata/hessian_workspace.npz")
if not workspace_path.exists():
    raise FileNotFoundError(
        f"Could not find hessian_workspace.npz. Please generate it first."
    )

data = np.load(workspace_path)
dm = data["dm"]
h_gn = data["h_gn"]
h_full = data["h_full"]
grad = data["grad"]
h_gn_grad = data["h_gn_grad"]
h_full_grad = data["h_full_grad"]
dx, dy = data["dx"], data["dy"]

# Crop to even dimensions for the transform
dm = dm[:-1, :-1]
h_gn = h_gn[:-1, :-1]
h_full = h_full[:-1, :-1]
grad = grad[:-1, :-1]
h_gn_grad = h_gn_grad[:-1, :-1]
h_full_grad = h_full_grad[:-1, :-1]

print(f"Loaded workspace with shape {dm.shape}")

# %%
# Computing Curvelet Transforms
# -----------------------------
# We transform both the true model (for the filter design) and the blurred
# Hessian action into the curvelet domain.

dimsd = h_gn.shape
crop = int(0.15 * dimsd[0])
slice_crop = slice(crop, -crop)

dm_c = dm[slice_crop, slice_crop]
grad_c = grad[slice_crop, slice_crop]
h_gn_grad_c = h_gn_grad[slice_crop, slice_crop]
h_full_grad_c = h_full_grad[slice_crop, slice_crop]

fig1, axs1 = plt.subplots(2, 2, figsize=(10, 10))
opts = {"cmap": "RdBu_r", "aspect": "equal"}
pclip = 0.5

im0 = axs1[0, 0].imshow(
    dm_c.T, vmin=-pclip * np.abs(dm_c).max(), vmax=pclip * np.abs(dm_c).max(), **opts
)
axs1[0, 0].set_title("1) True Scatterer ($\\delta m$)")
create_colorbar(im0, ax=axs1[0, 0])

im1 = axs1[0, 1].imshow(
    grad_c.T, vmin=-pclip * np.abs(grad_c).max(), vmax=pclip * np.abs(grad_c).max(), **opts
)
axs1[0, 1].set_title("2) Gradient ($g = J^T \\delta d$)")
create_colorbar(im1, ax=axs1[0, 1])

im2 = axs1[1, 0].imshow(
    h_gn_grad_c.T,
    vmin=-pclip * np.abs(h_gn_grad_c).max(),
    vmax=pclip * np.abs(h_gn_grad_c).max(),
    **opts,
)
axs1[1, 0].set_title("3) GN Hessian on Gradient ($H^{gn} g$)")
create_colorbar(im2, ax=axs1[1, 0])

im3 = axs1[1, 1].imshow(
    h_full_grad_c.T,
    vmin=-pclip * np.abs(h_full_grad_c).max(),
    vmax=pclip * np.abs(h_full_grad_c).max(),
    **opts,
)
axs1[1, 1].set_title("4) Full Newton Hessian on Gradient ($H^{fn} g$)")
create_colorbar(im3, ax=axs1[1, 1])

for ax in axs1.flat:
    despine(ax)

fig1.tight_layout()

# Number of scales is 4 since our grid is around 300x300
C = UDCT(shape=h_gn.shape, num_scales=4, wedges_per_direction=3)
coeffs_dm = C.forward(dm)
coeffs_h = C.forward(h_gn)
coeffs_h_full = C.forward(h_full)

# %%
# Visualizing Curvelet Coefficients
# ---------------------------------
# Let's visualize the curvelet coefficients at scale 2 (a mid-frequency scale)
scale_idx = 2

num_dirs = len(coeffs_h[scale_idx])
num_wedges = len(coeffs_h[scale_idx][0])

fig, axs = plt.subplots(num_dirs, num_wedges, figsize=(3 * num_wedges, 3 * num_dirs))
fig.suptitle(
    f"Curvelet Coefficients of Hessian Action (Scale {scale_idx})", fontsize=14
)

vmax_coeff = max(
    np.abs(coeffs_h[scale_idx][d][w]).max()
    for d in range(num_dirs)
    for w in range(num_wedges)
)

for d in range(num_dirs):
    for w in range(num_wedges):
        ax = axs[d, w]
        coeff_real = np.real(coeffs_h[scale_idx][d][w])
        im = ax.imshow(
            coeff_real.T,
            vmin=-vmax_coeff,
            vmax=vmax_coeff,
            cmap="RdBu_r",
            aspect="equal",
            extent=[0, 1, 1, 0],
        )
        ax.set_title(f"Dir {d}, Wedge {w}")
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)

fig.tight_layout()

# %%
# Estimating the Curvelet-Domain Hessian Filter (CHF)
# By applying the curvelet-domain inverse filter in sliding windows, we can estimate :math:`H^{-1}` and
# recover the true grid of scatterers from the blurred Hessian-vector product.

import pylops
from scipy.ndimage import gaussian_filter

# Define sliding window parameters
nwin = (64, 64)
nover = (32, 32)
dimsd = h_gn.shape

# Use PyLops to design the patch geometry
nwins, dims, mwins, dwins = pylops.signalprocessing.patch2d_design(
    dimsd, nwin, nover, nwin
)

# Pad the input images to the dimensions required by the PyLops patching
grad_pad = np.pad(grad, ((0, dims[0] - dimsd[0]), (0, dims[1] - dimsd[1])))
h_gn_grad_pad = np.pad(h_gn_grad, ((0, dims[0] - dimsd[0]), (0, dims[1] - dimsd[1])))
h_full_grad_pad = np.pad(
    h_full_grad, ((0, dims[0] - dimsd[0]), (0, dims[1] - dimsd[1]))
)

# Initialize the Curvelet Transform for the patch size
C_patch = UDCT(shape=nwin, num_scales=3, wedges_per_direction=3)

# Initialize arrays for the reconstructed image and taper weights
dm_est_pad = np.zeros_like(grad_pad)
dm_est_full_pad = np.zeros_like(grad_pad)
weights = np.zeros_like(grad_pad)

# Create a 2D Hanning taper for smooth blending of overlapping patches
taper = np.outer(np.hanning(nwin[0]), np.hanning(nwin[1]))

for i in range(nwins[0]):
    for j in range(nwins[1]):
        # Extract the patch using PyLops window indices
        slice_i = slice(dwins[0][0][i], dwins[0][1][i])
        slice_j = slice(dwins[1][0][j], dwins[1][1][j])

        patch_grad = grad_pad[slice_i, slice_j]
        patch_h_gn_grad = h_gn_grad_pad[slice_i, slice_j]
        patch_h_full_grad = h_full_grad_pad[slice_i, slice_j]

        # Forward Curvelet Transform of the patch
        coeffs_grad_patch = C_patch.forward(patch_grad)
        coeffs_h_gn_grad_patch = C_patch.forward(patch_h_gn_grad)
        coeffs_h_full_grad_patch = C_patch.forward(patch_h_full_grad)

        # Compute global max for stabilization across the entire patch
        # This prevents amplifying noise in wedges that are in the null space
        max_h_gn = max(
            np.max(np.abs(coeffs_h_gn_grad_patch[s][d][w]))
            for s in range(len(coeffs_h_gn_grad_patch))
            for d in range(len(coeffs_h_gn_grad_patch[s]))
            for w in range(len(coeffs_h_gn_grad_patch[s][d]))
        )

        max_h_full = max(
            np.max(np.abs(coeffs_h_full_grad_patch[s][d][w]))
            for s in range(len(coeffs_h_full_grad_patch))
            for d in range(len(coeffs_h_full_grad_patch[s]))
            for w in range(len(coeffs_h_full_grad_patch[s][d]))
        )

        eps_gn = 1e-2 * max_h_gn
        eps_full = 1e-2 * max_h_full

        coeffs_inv = []
        coeffs_inv_full = []
        for s in range(len(coeffs_grad_patch)):
            scale_coeffs = []
            scale_coeffs_full = []
            for d in range(len(coeffs_grad_patch[s])):
                dir_coeffs = []
                dir_coeffs_full = []
                for w in range(len(coeffs_grad_patch[s][d])):
                    abs_grad = np.abs(coeffs_grad_patch[s][d][w])
                    smooth_grad = gaussian_filter(abs_grad, sigma=2.0)

                    # For GN
                    abs_h = np.abs(coeffs_h_gn_grad_patch[s][d][w])
                    smooth_h = gaussian_filter(abs_h, sigma=2.0)
                    filt = smooth_grad / (smooth_h + eps_gn)
                    dir_coeffs.append(coeffs_grad_patch[s][d][w] * filt)

                    # For Full Newton
                    abs_h_full = np.abs(coeffs_h_full_grad_patch[s][d][w])
                    smooth_h_full = gaussian_filter(abs_h_full, sigma=2.0)
                    filt_full = smooth_grad / (smooth_h_full + eps_full)
                    dir_coeffs_full.append(coeffs_grad_patch[s][d][w] * filt_full)

                scale_coeffs.append(dir_coeffs)
                scale_coeffs_full.append(dir_coeffs_full)
            coeffs_inv.append(scale_coeffs)
            coeffs_inv_full.append(scale_coeffs_full)

        # Inverse transform the filtered patch
        patch_est = np.real(C_patch.backward(coeffs_inv))
        patch_est_full = np.real(C_patch.backward(coeffs_inv_full))

        # Overlap-add accumulation
        dm_est_pad[slice_i, slice_j] += patch_est * taper
        dm_est_full_pad[slice_i, slice_j] += patch_est_full * taper
        weights[slice_i, slice_j] += taper

# Normalize by weights and crop back to original dimensions
dm_est = (dm_est_pad / (weights + 1e-10))[: dimsd[0], : dimsd[1]]
dm_est_full = (dm_est_full_pad / (weights + 1e-10))[: dimsd[0], : dimsd[1]]

# %%
# Plotting the Recovered Image
# ----------------------------

dm_est_c = dm_est[slice_crop, slice_crop]
dm_est_full_c = dm_est_full[slice_crop, slice_crop]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

opts = {"cmap": "RdBu_r", "aspect": "equal"}
pclip = 0.5
vmax_dm = pclip * np.abs(dm_c).max()
vmax_grad = pclip * np.abs(grad_c).max()
vmax_est = pclip * np.abs(dm_est_c).max()
vmax_est_full = pclip * np.abs(dm_est_full_c).max()

im0 = axs[0, 0].imshow(dm_c.T, vmin=-vmax_dm, vmax=vmax_dm, **opts)
axs[0, 0].set_title("1) True Scatterer ($\\delta m$)")
create_colorbar(im0, ax=axs[0, 0])

im1 = axs[0, 1].imshow(grad_c.T, vmin=-vmax_grad, vmax=vmax_grad, **opts)
axs[0, 1].set_title("2) Gradient ($g = J^T \\delta d$)")
create_colorbar(im1, ax=axs[0, 1])

im2 = axs[1, 0].imshow(dm_est_c.T, vmin=-vmax_est, vmax=vmax_est, **opts)
axs[1, 0].set_title("3) Recovered via GN CHF ($(H^{gn})^{-1} g$)")
create_colorbar(im2, ax=axs[1, 0])

im3 = axs[1, 1].imshow(dm_est_full_c.T, vmin=-vmax_est_full, vmax=vmax_est_full, **opts)
axs[1, 1].set_title("4) Recovered via Full Newton CHF ($(H^{fn})^{-1} g$)")
create_colorbar(im3, ax=axs[1, 1])

for ax in axs.flat:
    despine(ax)

fig.tight_layout()
plt.show()

# %%
# References
# ----------
#
# .. bibliography::
#    :filter: docname in docnames

