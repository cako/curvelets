"""
Curvelet-based Denoising
========================

This example shows how the UDCT curvelets transform can be used to denoise images. More
precisily, an the curvelet transform of a natual image tends to have strong, localized
coefficients; on the other hand, the UDCT transform of a (white Gaussian) noise
realization does not map into anything really consistent in the curvelets domain. As
such if we take the UDCT transform of a noisy image and apply a suitable thresholding
(e.g., by retaining a given percentage of the coefficients sorted in decreasing order),
the inverse UDCT transform produces a denoised version of the input image.
"""

from __future__ import annotations

# %%
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy as sp

from curvelets.numpy import UDCT

# %%
# Denoising by thresholding
# #########################
#
# Define function that retains only a percentage of the strongest coefficients of the
# UDCT transform


def udct_threshold(
    cwt_vect: npt.NDArray[np.complex64], perc: float
) -> npt.NDArray[np.complex64]:
    """
    Apply threshold to vectorized UDCT coefficients

    Parameters
    ----------
    cwt_vect : numpy.ndarray
        UDCT coefficients (vectorized version)
    perc : float
        Percentage of coefficients being retained

    Returns
    -------
    out : float
        Thresholded UDCT coefficients (vectorized version)

    """
    n = round(cwt_vect.size * perc)
    idxs = np.argsort(np.abs(cwt_vect))[::-1][:n]
    out = np.zeros_like(cwt_vect.ravel())
    out[idxs] = cwt_vect[idxs]
    return out.reshape(cwt_vect.shape)


# %%
# Noisy image
# ###########
#
# Create noisy image by adding a realization of white Gaussian noise to a
# natural image

# Load image
d = sp.datasets.face()
d = d / d.max()

# Add noise to image
dorig = d.copy()
rng = np.random.default_rng()
d = d + rng.normal(0, 3e-1, d.shape)
d = np.clip(d, 0, 1)

# %%
# Denoising
# #########
#
# Apply denoising by thresholding in the UDCT domain (applied individually to each RGB)
# component

# Defined UDCT transform
Cop = UDCT(shape=d.shape[:2], num_scales=4, transform_kind="real")

perc = 0.04
d_cwt = []
for i in range(3):
    cwt = Cop.vect(Cop.forward(d[..., i]))
    cwt_thresh = udct_threshold(cwt, perc)
    cwt_thresh = Cop.struct(cwt_thresh)
    d_cwt_raw = Cop.backward(cwt_thresh)
    d_cwt.append(np.clip(d_cwt_raw, 0, 1))
d_cwt = np.stack(d_cwt, axis=-1)

fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharey=True, sharex=True)
axs[0, 0].imshow(dorig)
axs[0, 0].set_title("Clean")
axs[0, 1].imshow(d)
axs[0, 1].set_title("Noisy")
axs[0, 2].imshow(d_cwt)
axs[0, 2].set_title("Denoised")
axs[1, 0].axis("off")
axs[1, 1].imshow(np.abs(d - d_cwt))
axs[1, 1].set_title("Removed noise")
axs[1, 2].imshow(np.abs(dorig - d_cwt))
axs[1, 2].set_title("Signal leakage")
for ax in axs.ravel():
    ax.axis("off")
    ax.axis("tight")
fig.tight_layout()
