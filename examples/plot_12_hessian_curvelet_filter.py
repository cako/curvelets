r"""
Curvelet-Domain Hessian Action
==============================

This example showcases how the curvelet transform can be used to estimate and
compensate for the action of the Hessian in Least-Squares Migration (LSM), inspired by
Wang et al. (2016, 2017).

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

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import ticker

from curvelets.numpy import UDCT
from curvelets.plot import create_colorbar, despine

# %%
# Load the Hessian-Vector Product Workspace
# -----------------------------------------
# We load a pre-computed workspace containing a point scatterer perturbation (:math:`\delta m`) 
# and the Gauss-Newton Hessian action on it (:math:`h_{gn}`).

workspace_path = Path("../testdata/hessian_workspace.npz")
if not workspace_path.exists():
    raise FileNotFoundError(f"Could not find {workspace_path}. Please generate it first.")

data = np.load(workspace_path)
dm = data["dm"]
h_gn = data["h_gn"]

# Crop to even dimensions for the transform
dm = dm[:-1, :-1]
h_gn = h_gn[:-1, :-1]

print(f"Loaded workspace with shape {dm.shape}")

# %%
# Visualizing the Point Scatterer and its Hessian Action
# ------------------------------------------------------
# Notice how the true scatterer is an isotropic circle, but the Hessian action 
# smears it into a distinct "rabbit ears" or "hourglass" shape due to the 
# limited acquisition geometry (surface receivers only).

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

opts = {"cmap": "RdBu_r", "aspect": "equal"}
vmax_dm = np.abs(dm).max()
vmax_h = np.abs(h_gn).max()

im0 = axs[0].imshow(dm.T, vmin=-vmax_dm, vmax=vmax_dm, **opts)
axs[0].set_title("True Scatterer ($\\delta m$)")
create_colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(h_gn.T, vmin=-vmax_h, vmax=vmax_h, **opts)
axs[1].set_title("Gauss-Newton Hessian Action ($J^\\dagger J \\delta m$)")
create_colorbar(im1, ax=axs[1])

for ax in axs:
    despine(ax)

fig.tight_layout()

# %%
# Curvelet Transform of the Hessian Action
# ----------------------------------------
# We compute the curvelet transform of both the scatterer and the Hessian action.
# We'll use 4 scales and 3 wedges per direction.

C = UDCT(shape=dm.shape, num_scales=4, wedges_per_direction=3)

coeffs_dm = C.forward(dm)
coeffs_h = C.forward(h_gn)

# Let's visualize the curvelet coefficients at scale 2 (a mid-frequency scale)
scale_idx = 2

num_dirs = len(coeffs_h[scale_idx])
num_wedges = len(coeffs_h[scale_idx][0])

fig, axs = plt.subplots(num_dirs, num_wedges, figsize=(3 * num_wedges, 3 * num_dirs))
fig.suptitle(f"Curvelet Coefficients of Hessian Action (Scale {scale_idx})", fontsize=14)

vmax_coeff = max(np.abs(coeffs_h[scale_idx][d][w]).max() for d in range(num_dirs) for w in range(num_wedges))

for d in range(num_dirs):
    for w in range(num_wedges):
        ax = axs[d, w]
        coeff_real = np.real(coeffs_h[scale_idx][d][w])
        im = ax.imshow(coeff_real.T, vmin=-vmax_coeff, vmax=vmax_coeff, cmap="RdBu_r", aspect="equal")
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
h_gn_pad = np.pad(h_gn, ((0, dims[0] - dimsd[0]), (0, dims[1] - dimsd[1])))
dm_pad = np.pad(dm, ((0, dims[0] - dimsd[0]), (0, dims[1] - dimsd[1])))

# Initialize the Curvelet Transform for the patch size
C_patch = UDCT(shape=nwin, num_scales=3, wedges_per_direction=3)

# Initialize arrays for the reconstructed image and taper weights
dm_est_pad = np.zeros_like(h_gn_pad)
weights = np.zeros_like(h_gn_pad)

# Create a 2D Hanning taper for smooth blending of overlapping patches
taper = np.outer(np.hanning(nwin[0]), np.hanning(nwin[1]))

for i in range(nwins[0]):
    for j in range(nwins[1]):
        # Extract the patch using PyLops window indices
        slice_i = slice(dwins[0][0][i], dwins[0][1][i])
        slice_j = slice(dwins[1][0][j], dwins[1][1][j])
        
        patch_h = h_gn_pad[slice_i, slice_j]
        patch_dm = dm_pad[slice_i, slice_j]
        
        # Forward Curvelet Transform of the patch
        coeffs_h_patch = C_patch.forward(patch_h)
        coeffs_dm_patch = C_patch.forward(patch_dm)
        
        coeffs_inv = []
        for s in range(len(coeffs_h_patch)):
            scale_coeffs = []
            for d in range(len(coeffs_h_patch[s])):
                dir_coeffs = []
                for w in range(len(coeffs_h_patch[s][d])):
                    abs_dm = np.abs(coeffs_dm_patch[s][d][w])
                    abs_h = np.abs(coeffs_h_patch[s][d][w])
                    
                    # Spatially varying filter using smoothed envelopes (Wang et al., 2016)
                    smooth_dm = gaussian_filter(abs_dm, sigma=2.0)
                    smooth_h = gaussian_filter(abs_h, sigma=2.0)
                    
                    # Tikhonov regularized inverse filter
                    max_val = np.max(smooth_h)
                    filt = smooth_dm / (smooth_h + 1e-3 * max_val)
                    dir_coeffs.append(coeffs_h_patch[s][d][w] * filt)
                scale_coeffs.append(dir_coeffs)
            coeffs_inv.append(scale_coeffs)
        
        # Inverse transform the filtered patch
        patch_est = np.real(C_patch.backward(coeffs_inv))
        
        # Overlap-add accumulation
        dm_est_pad[slice_i, slice_j] += patch_est * taper
        weights[slice_i, slice_j] += taper

# Normalize by weights and crop back to original dimensions
dm_est = (dm_est_pad / (weights + 1e-10))[:dimsd[0], :dimsd[1]]

# %%
# Plotting the Recovered Image
# ----------------------------

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

opts = {"cmap": "RdBu_r", "aspect": "equal"}
vmax_dm = np.abs(dm).max()
vmax_h = np.abs(h_gn).max()
vmax_est = np.abs(dm_est).max()

im0 = axs[0].imshow(dm.T, vmin=-vmax_dm, vmax=vmax_dm, **opts)
axs[0].set_title("True Grid ($\\delta m$)")
create_colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(h_gn.T, vmin=-vmax_h, vmax=vmax_h, **opts)
axs[1].set_title("Blurred by Hessian ($J^\\dagger J \\delta m$)")
create_colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(dm_est.T, vmin=-vmax_est, vmax=vmax_est, **opts)
axs[2].set_title("Recovered via CHF ($H^{-1}_{est} J^\\dagger J \\delta m$)")
create_colorbar(im2, ax=axs[2])

for ax in axs:
    despine(ax)

fig.tight_layout()
plt.show()

