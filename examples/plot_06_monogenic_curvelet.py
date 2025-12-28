"""
Monogenic Curvelet Transform
============================

This example reproduces Figure 2 from :cite:`Storath2010`, showing filters of usual
curvelets and monogenic curvelets for an isotropic scale.

The monogenic curvelet transform extends the standard curvelet transform by
applying Riesz transforms, producing three components per band that form a
quaternion-like structure. This enables meaningful amplitude/phase decomposition
over all scales, unlike the standard curvelet transform which only provides
this decomposition at the highest scale.

Mathematical Foundation
------------------------

The monogenic signal :math:`M_f` of a real-valued function :math:`f` is defined as:

.. math::
   M_f = f + i(-R_1f) + j(-R_2f)

where :math:`R_1` and :math:`R_2` are the first two Riesz transforms, defined in
the frequency domain as:

.. math::
   \\widehat{R_k(f)}(\\xi) = i \\frac{\\xi_k}{|\\xi|} \\widehat{f}(\\xi)

for :math:`k = 1, 2`, where :math:`\\widehat{f}` is the Fourier transform of :math:`f`,
:math:`\\xi = (\\xi_1, \\xi_2)` is the frequency vector, and :math:`|\\xi|` is its magnitude.

The monogenic curvelet transform applies this monogenic signal construction to
each curvelet band :math:`\\beta_{a\\theta}` (where :math:`a` is scale and :math:`\\theta`
is direction), producing three components:

- **Scalar component** :math:`\\beta_{a\\theta}`: Same as the standard curvelet coefficient
- **Riesz_1 component** :math:`\\mathcal{R}_1\\beta_{a\\theta}`: First Riesz transform applied to the curvelet
- **Riesz_2 component** :math:`\\mathcal{R}_2\\beta_{a\\theta}`: Second Riesz transform applied to the curvelet

The amplitude of the monogenic signal is computed as:

.. math::
   |M| = \\sqrt{|\\beta|^2 + (\\mathcal{R}_1\\beta)^2 + (\\mathcal{R}_2\\beta)^2}

This amplitude provides a scale-invariant measure of local structure, enabling
meaningful phase/amplitude analysis at all scales.

.. note::
   The monogenic curvelet transform is mathematically defined only for 2D signals
   according to :cite:`Storath2010`. While the implementation accepts arbitrary
   dimensions, only the first two Riesz components (R_1 and R_2) are computed,
   which is correct only for 2D inputs.
"""

from __future__ import annotations

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from curvelets.numpy import UDCT
from curvelets.numpy._riesz import riesz_filters
from curvelets.plot import create_colorbar, despine

# %%
# Setup: Create a curvelet at isotropic scale
# ############################################
#
# We'll extract the low-frequency band (isotropic scale) and visualize
# the scalar component and its Riesz transforms.

shape = (256, 256)
# Use smaller window overlap for better reconstruction accuracy
transform = UDCT(shape=shape, num_scales=3, wedges_per_direction=3, window_overlap=0.1)

# Create a delta function at the center to visualize the curvelet
image = np.zeros(shape)
image[shape[0] // 2, shape[1] // 2] = 1.0

# Get monogenic coefficients
coeffs_mono = transform.forward_monogenic(image)

# Extract the low-frequency band (scale 0, isotropic)
scalar_low = coeffs_mono[0][0][0][0]  # Scalar component
riesz1_low = coeffs_mono[0][0][0][1]  # Riesz_1 component
riesz2_low = coeffs_mono[0][0][0][2]  # Riesz_2 component

# %%
# Time Domain Visualization
# ##########################
#
# Figure 2a: From left to right: :math:`\beta_{a0\theta} = \gamma_{a0\theta}`,
# :math:`\mathcal{R}_1\beta_{a0\theta}`, and :math:`\mathcal{R}_2\beta_{a0\theta}`

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Scalar component (:math:`\beta = \gamma` at isotropic scale)
vmax = np.abs(scalar_low).max()
opts = {"aspect": "equal", "cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax}

im = axs[0].imshow(np.real(scalar_low.T), **opts)
_, cb = create_colorbar(im=im, ax=axs[0])
despine(axs[0])
axs[0].set(title=r"$\beta_{a0\theta} = \gamma_{a0\theta}$" + "\n" + "(Scalar)")

# Riesz_1 component
vmax = np.abs(riesz1_low).max()
opts["vmax"] = vmax
opts["vmin"] = -vmax

im = axs[1].imshow(riesz1_low.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[1])
despine(axs[1])
axs[1].set(title=r"$\mathcal{R}_1\beta_{a0\theta}$" + "\n" + "(Riesz_1)")

# Riesz_2 component
im = axs[2].imshow(riesz2_low.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[2])
despine(axs[2])
axs[2].set(title=r"$\mathcal{R}_2\beta_{a0\theta}$" + "\n" + "(Riesz_2)")

plt.tight_layout()

# %%
# Frequency Domain Visualization
# ###############################
#
# Figure 2b: From left to right: :math:`\widehat{\beta}_{a0\theta}`,
# :math:`i\widehat{\mathcal{R}_1\beta}_{a0\theta}`, and
# :math:`i\widehat{\mathcal{R}_2\beta}_{a0\theta}`
#
# To visualize in frequency domain, we compute the FFT of each component
# and apply the Riesz filters in frequency domain. The Riesz transforms are
# applied as :math:`i \cdot R_k \cdot \widehat{\beta}`, where :math:`R_k` is the
# Riesz filter in frequency domain.

# Get the low-frequency window to understand the frequency support
window = transform.windows[0][0][0]
idx, val = window

# Create frequency-domain representation
frequency_band = np.zeros(shape, dtype=np.complex128)
frequency_band.flat[idx] = val

# Apply Riesz filters in frequency domain
riesz_filters_list = riesz_filters(shape)
riesz1_filter = riesz_filters_list[0]
riesz2_filter = riesz_filters_list[1]

# Frequency domain: :math:`\widehat{\beta}`
freq_scalar = frequency_band.copy()

# Frequency domain: :math:`i\widehat{\mathcal{R}_1\beta} = i \cdot R_1 \cdot \widehat{\beta}`
freq_riesz1 = 1j * riesz1_filter * frequency_band

# Frequency domain: :math:`i\widehat{\mathcal{R}_2\beta} = i \cdot R_2 \cdot \widehat{\beta}`
freq_riesz2 = 1j * riesz2_filter * frequency_band

# Visualize frequency domain
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Shift zero frequency to center for visualization
freq_scalar_shifted = np.fft.fftshift(freq_scalar)
freq_riesz1_shifted = np.fft.fftshift(freq_riesz1)
freq_riesz2_shifted = np.fft.fftshift(freq_riesz2)

vmax = np.abs(freq_scalar_shifted).max()
opts = {"aspect": "equal", "cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax}

im = axs[0].imshow(np.real(freq_scalar_shifted).T, **opts)
_, cb = create_colorbar(im=im, ax=axs[0])
despine(axs[0])
axs[0].set(title=r"$\widehat{\beta}_{a0\theta}$" + "\n" + "(Scalar, frequency)")

vmax = np.abs(freq_riesz1_shifted).max()
opts["vmax"] = vmax
opts["vmin"] = -vmax

im = axs[1].imshow(np.real(freq_riesz1_shifted).T, **opts)
_, cb = create_colorbar(im=im, ax=axs[1])
despine(axs[1])
axs[1].set(
    title=r"$i\widehat{\mathcal{R}_1\beta}_{a0\theta}$" + "\n" + "(Riesz_1, frequency)"
)

im = axs[2].imshow(np.real(freq_riesz2_shifted).T, **opts)
_, cb = create_colorbar(im=im, ax=axs[2])
despine(axs[2])
axs[2].set(
    title=r"$i\widehat{\mathcal{R}_2\beta}_{a0\theta}$" + "\n" + "(Riesz_2, frequency)"
)

plt.tight_layout()

# %%
# Amplitude Computation
# ######################
#
# The amplitude of the monogenic signal is computed as:
#
# .. math::
#    |M| = \sqrt{|\beta|^2 + (\mathcal{R}_1\beta)^2 + (\mathcal{R}_2\beta)^2}
#
# For the scalar component, we use :math:`|\beta|` since it's complex (matching
# the standard curvelet transform behavior). The Riesz components are real-valued.

amplitude = np.sqrt(np.abs(scalar_low) ** 2 + riesz1_low**2 + riesz2_low**2)

fig, ax = plt.subplots(figsize=(5, 4))
vmax = amplitude.max()
opts = {"aspect": "equal", "cmap": "gray", "vmin": 0, "vmax": vmax}

im = ax.imshow(amplitude.T, **opts)
_, cb = create_colorbar(im=im, ax=ax)
despine(ax)
ax.set(title=r"$|M| = \sqrt{|\beta|^2 + \mathcal{R}_1\beta^2 + \mathcal{R}_2\beta^2}$")

plt.tight_layout()

# %%
# Round-Trip Consistency Check
# ############################
#
# According to Storath 2010, the monogenic curvelet transform should satisfy
# the reproducing formula:
#
# .. math::
#    M_f(x) = \int \langle M\beta_{ab\theta}, f \rangle \cdot M\beta_{ab\theta}(x) \, db \, d\theta \, \frac{da}{a^3}
#
# This means that ``backward_monogenic(forward_monogenic(f))`` should produce
# the same result as ``monogenic(f)``, which directly computes:
#
# .. math::
#    M_f = (f, -R_1 f, -R_2 f)
#
# Let's verify this component by component.

# Create a test image (zone plate for interesting structure)
x = np.linspace(-1, 1, shape[0])
y = np.linspace(-1, 1, shape[1])
X, Y = np.meshgrid(x, y, indexing="ij")
test_image = np.sin(20 * (X**2 + Y**2))

# Method 1: Direct monogenic signal computation
scalar_direct, riesz1_direct, riesz2_direct = transform.monogenic(test_image)

# Method 2: Round-trip through monogenic curvelet transform
coeffs = transform.forward_monogenic(test_image)
scalar_round, riesz1_round, riesz2_round = transform.backward_monogenic(coeffs)

# %%
# Scalar Component Comparison: f vs scalar
# ########################################
#
# The scalar component should match the original input ``f``.

fig, axs = plt.subplots(1, 4, figsize=(16, 4))

vmax = np.abs(test_image).max()
opts = {"aspect": "equal", "cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax}

# Original input f
im = axs[0].imshow(test_image.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[0])
despine(axs[0])
axs[0].set(title="Original input\n" + r"$f$")

# Direct: scalar_direct (should equal f)
im = axs[1].imshow(scalar_direct.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[1])
despine(axs[1])
axs[1].set(title="monogenic(f)[0]\n(should equal f)")

# Round-trip: scalar_round
im = axs[2].imshow(scalar_round.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[2])
despine(axs[2])
axs[2].set(title="backward(forward(f))[0]\n(scalar)")

# Difference
diff_scalar = scalar_round - test_image
vmax_diff = np.abs(diff_scalar).max()
opts_diff = {"aspect": "equal", "cmap": "RdBu_r", "vmin": -vmax_diff, "vmax": vmax_diff}
im = axs[3].imshow(diff_scalar.T, **opts_diff)
_, cb = create_colorbar(im=im, ax=axs[3])
despine(axs[3])
axs[3].set(title=f"Difference\nmax={vmax_diff:.4f}")

plt.suptitle(
    "Scalar Component: f vs backward_monogenic(forward_monogenic(f))[0]", y=1.02
)
plt.tight_layout()

# Print statistics
print("Scalar component comparison:")  # noqa: T201
print(f"  Max diff (f vs scalar_round): {np.abs(test_image - scalar_round).max():.6e}")  # noqa: T201
print(f"  Ratio (scalar_round / f) at center: {scalar_round[128, 128] / test_image[128, 128]:.4f}")  # noqa: T201

# %%
# Riesz_1 Component Comparison: -R₁f vs riesz1
# ############################################
#
# The riesz1 component should match ``-R₁f``.

fig, axs = plt.subplots(1, 4, figsize=(16, 4))

vmax = np.abs(riesz1_direct).max()
opts = {"aspect": "equal", "cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax}

# Direct: -R₁f
im = axs[0].imshow(riesz1_direct.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[0])
despine(axs[0])
axs[0].set(title="monogenic(f)[1]\n" + r"$-R_1 f$")

# Round-trip: riesz1_round
im = axs[1].imshow(riesz1_round.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[1])
despine(axs[1])
axs[1].set(title="backward(forward(f))[1]\n(riesz1)")

# Difference
diff_riesz1 = riesz1_round - riesz1_direct
vmax_diff = np.abs(diff_riesz1).max()
opts_diff = {"aspect": "equal", "cmap": "RdBu_r", "vmin": -vmax_diff, "vmax": vmax_diff}
im = axs[2].imshow(diff_riesz1.T, **opts_diff)
_, cb = create_colorbar(im=im, ax=axs[2])
despine(axs[2])
axs[2].set(title=f"Difference\nmax={vmax_diff:.4f}")

# Ratio (where riesz1_direct is non-negligible)
mask = np.abs(riesz1_direct) > 0.01 * np.abs(riesz1_direct).max()
ratio = np.zeros_like(riesz1_direct)
ratio[mask] = riesz1_round[mask] / riesz1_direct[mask]
im = axs[3].imshow(ratio.T, aspect="equal", cmap="viridis", vmin=0, vmax=3)
_, cb = create_colorbar(im=im, ax=axs[3])
despine(axs[3])
mean_ratio = np.mean(ratio[mask]) if mask.any() else 0
axs[3].set(title=f"Ratio (round/direct)\nmean={mean_ratio:.4f}")

plt.suptitle(
    r"Riesz_1 Component: $-R_1 f$ vs backward_monogenic(forward_monogenic(f))[1]",
    y=1.02,
)
plt.tight_layout()

# Print statistics
print("\nRiesz_1 component comparison:")  # noqa: T201
print(f"  Max diff: {np.abs(riesz1_direct - riesz1_round).max():.6e}")  # noqa: T201
print(f"  Mean ratio (where significant): {mean_ratio:.4f}")  # noqa: T201

# %%
# Riesz_2 Component Comparison: -R₂f vs riesz2
# ############################################
#
# The riesz2 component should match ``-R₂f``.

fig, axs = plt.subplots(1, 4, figsize=(16, 4))

vmax = np.abs(riesz2_direct).max()
opts = {"aspect": "equal", "cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax}

# Direct: -R₂f
im = axs[0].imshow(riesz2_direct.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[0])
despine(axs[0])
axs[0].set(title="monogenic(f)[2]\n" + r"$-R_2 f$")

# Round-trip: riesz2_round
im = axs[1].imshow(riesz2_round.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[1])
despine(axs[1])
axs[1].set(title="backward(forward(f))[2]\n(riesz2)")

# Difference
diff_riesz2 = riesz2_round - riesz2_direct
vmax_diff = np.abs(diff_riesz2).max()
opts_diff = {"aspect": "equal", "cmap": "RdBu_r", "vmin": -vmax_diff, "vmax": vmax_diff}
im = axs[2].imshow(diff_riesz2.T, **opts_diff)
_, cb = create_colorbar(im=im, ax=axs[2])
despine(axs[2])
axs[2].set(title=f"Difference\nmax={vmax_diff:.4f}")

# Ratio (where riesz2_direct is non-negligible)
mask = np.abs(riesz2_direct) > 0.01 * np.abs(riesz2_direct).max()
ratio = np.zeros_like(riesz2_direct)
ratio[mask] = riesz2_round[mask] / riesz2_direct[mask]
im = axs[3].imshow(ratio.T, aspect="equal", cmap="viridis", vmin=0, vmax=3)
_, cb = create_colorbar(im=im, ax=axs[3])
despine(axs[3])
mean_ratio = np.mean(ratio[mask]) if mask.any() else 0
axs[3].set(title=f"Ratio (round/direct)\nmean={mean_ratio:.4f}")

plt.suptitle(
    r"Riesz_2 Component: $-R_2 f$ vs backward_monogenic(forward_monogenic(f))[2]",
    y=1.02,
)
plt.tight_layout()

# Print statistics
print("\nRiesz_2 component comparison:")  # noqa: T201
print(f"  Max diff: {np.abs(riesz2_direct - riesz2_round).max():.6e}")  # noqa: T201
print(f"  Mean ratio (where significant): {mean_ratio:.4f}")  # noqa: T201

# %%
# Frequency Domain Analysis
# #########################
#
# To understand the mismatch, let's look at the frequency domain.
# We compare the FFT of the reconstructed components.

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Row 1: Direct monogenic (expected)
freq_f = np.fft.fftshift(np.fft.fft2(test_image))
freq_r1_direct = np.fft.fftshift(np.fft.fft2(riesz1_direct))
freq_r2_direct = np.fft.fftshift(np.fft.fft2(riesz2_direct))

# Row 2: Round-trip (actual)
freq_scalar_round = np.fft.fftshift(np.fft.fft2(scalar_round))
freq_r1_round = np.fft.fftshift(np.fft.fft2(riesz1_round))
freq_r2_round = np.fft.fftshift(np.fft.fft2(riesz2_round))


# Plot with log scale for better visualization
def plot_freq(ax, data, title):
    """Plot frequency magnitude on log scale."""
    mag = np.abs(data)
    mag[mag < 1e-10] = 1e-10  # Avoid log(0)
    im = ax.imshow(np.log10(mag).T, aspect="equal", cmap="viridis")
    create_colorbar(im=im, ax=ax)
    despine(ax)
    ax.set(title=title)


plot_freq(axs[0, 0], freq_f, "FFT(f)\n(expected scalar)")
plot_freq(axs[0, 1], freq_r1_direct, r"FFT($-R_1 f$)" + "\n(expected riesz1)")
plot_freq(axs[0, 2], freq_r2_direct, r"FFT($-R_2 f$)" + "\n(expected riesz2)")

plot_freq(axs[1, 0], freq_scalar_round, "FFT(scalar_round)\n(actual)")
plot_freq(axs[1, 1], freq_r1_round, "FFT(riesz1_round)\n(actual)")
plot_freq(axs[1, 2], freq_r2_round, "FFT(riesz2_round)\n(actual)")

plt.suptitle("Frequency Domain Comparison (log₁₀ magnitude)", y=1.02)
plt.tight_layout()

# %%
# Comparison with Standard UDCT Backward
# ######################################
#
# The standard UDCT backward transform should perfectly reconstruct f.
# Let's compare the scalar component with what standard backward gives.

# Standard UDCT round-trip
coeffs_standard = transform.forward(test_image)
recon_standard = transform.backward(coeffs_standard)

# Also try: what if we just use the scalar coefficients from monogenic
# and apply standard backward transform logic?
# Extract just the scalar coefficients (c0) from monogenic coefficients
scalar_coeffs_only = [
    [
        [coeffs[scale][dir][wedge][0] for wedge in range(len(coeffs[scale][dir]))]
        for dir in range(len(coeffs[scale]))
    ]
    for scale in range(len(coeffs))
]

# Apply standard backward to scalar-only coefficients
recon_from_scalar = transform.backward(scalar_coeffs_only)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))

vmax = np.abs(test_image).max()
opts = {"aspect": "equal", "cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax}

# Original
im = axs[0].imshow(test_image.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[0])
despine(axs[0])
axs[0].set(title="Original f")

# Standard UDCT backward
im = axs[1].imshow(recon_standard.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[1])
despine(axs[1])
diff_std = np.abs(test_image - recon_standard).max()
axs[1].set(title=f"Standard backward(forward(f))\nmax diff={diff_std:.2e}")

# Backward using only scalar coeffs from monogenic
im = axs[2].imshow(recon_from_scalar.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[2])
despine(axs[2])
diff_scalar_only = np.abs(test_image - recon_from_scalar).max()
axs[2].set(title=f"backward(c₀ only)\nmax diff={diff_scalar_only:.2e}")

# Monogenic backward scalar component
im = axs[3].imshow(scalar_round.T, **opts)
_, cb = create_colorbar(im=im, ax=axs[3])
despine(axs[3])
diff_mono = np.abs(test_image - scalar_round).max()
axs[3].set(title=f"backward_monogenic()[0]\nmax diff={diff_mono:.2e}")

plt.suptitle("Scalar Reconstruction Comparison", y=1.02)
plt.tight_layout()

print("\nScalar reconstruction comparison:")  # noqa: T201
print(f"  Standard backward:          max diff = {diff_std:.6e}")  # noqa: T201
print(f"  backward(c₀ only):          max diff = {diff_scalar_only:.6e}")  # noqa: T201
print(f"  backward_monogenic()[0]:    max diff = {diff_mono:.6e}")  # noqa: T201

# %%
# Alternative Reconstruction: Apply Riesz to Scalar
# ##################################################
#
# Since the scalar component correctly reconstructs f, we can try applying
# the Riesz transforms to it to get -R₁f and -R₂f.

from curvelets.numpy._riesz import riesz_filters

# Reconstruct scalar (which should equal f)
coeffs_scalar_only = [
    [
        [coeffs[scale][dir][wedge][0] for wedge in range(len(coeffs[scale][dir]))]
        for dir in range(len(coeffs[scale]))
    ]
    for scale in range(len(coeffs))
]
f_recon = transform.backward(coeffs_scalar_only)

# Apply Riesz transforms to reconstructed f
filters = riesz_filters(shape)
f_fft = np.fft.fftn(f_recon)
riesz1_from_scalar = -np.fft.ifftn(f_fft * filters[0]).real
riesz2_from_scalar = -np.fft.ifftn(f_fft * filters[1]).real

# Compare with direct monogenic
print("\nAlternative: Apply Riesz to reconstructed scalar:")  # noqa: T201
print(f"  max|f_recon - f|: {np.abs(f_recon - test_image).max():.6e}")  # noqa: T201
print(
    f"  max|-R₁f - riesz1_from_scalar|: {np.abs(riesz1_direct - riesz1_from_scalar).max():.6e}"
)  # noqa: T201
print(
    f"  max|-R₂f - riesz2_from_scalar|: {np.abs(riesz2_direct - riesz2_from_scalar).max():.6e}"
)  # noqa: T201

# Compare with backward_monogenic results
print("\nComparison with backward_monogenic:")  # noqa: T201
print(
    f"  Scalar: backward_monogenic error = {np.abs(test_image - scalar_round).max():.6e}"
)  # noqa: T201
print(
    f"  Riesz1: backward_monogenic error = {np.abs(riesz1_direct - riesz1_round).max():.6e}"
)  # noqa: T201
print(
    f"  Riesz2: backward_monogenic error = {np.abs(riesz2_direct - riesz2_round).max():.6e}"
)  # noqa: T201

# %%
# Cross-term Analysis
# ###################
#
# The quaternion multiplication adds cross-terms to the scalar reconstruction:
# scalar = c₀·W + c₁·(W·R₁) + c₂·(W·R₂)
#
# Let's analyze the contribution of each term.

# For one wedge, analyze the contributions
scale_idx = 1  # First high-frequency scale
dir_idx = 0
wedge_idx = 0

c0 = coeffs[scale_idx][dir_idx][wedge_idx][0]
c1 = coeffs[scale_idx][dir_idx][wedge_idx][1]
c2 = coeffs[scale_idx][dir_idx][wedge_idx][2]

print(
    f"\nCoefficient magnitudes for scale={scale_idx}, dir={dir_idx}, wedge={wedge_idx}:"
)  # noqa: T201
print(f"  |c₀| (scalar):  max={np.abs(c0).max():.6e}, mean={np.abs(c0).mean():.6e}")  # noqa: T201
print(f"  |c₁| (riesz1):  max={np.abs(c1).max():.6e}, mean={np.abs(c1).mean():.6e}")  # noqa: T201
print(f"  |c₂| (riesz2):  max={np.abs(c2).max():.6e}, mean={np.abs(c2).mean():.6e}")  # noqa: T201
print(f"  Ratio |c₁|/|c₀|: {np.abs(c1).mean() / np.abs(c0).mean():.6f}")  # noqa: T201
print(f"  Ratio |c₂|/|c₀|: {np.abs(c2).mean() / np.abs(c0).mean():.6f}")  # noqa: T201

# %%
# Summary Statistics
# ##################

print("\n" + "=" * 60)  # noqa: T201
print("SUMMARY: Component-by-Component Comparison")  # noqa: T201
print("=" * 60)  # noqa: T201
print(f"Scalar:  max|f - scalar_round| = {np.abs(test_image - scalar_round).max():.6e}")  # noqa: T201
print(f"Riesz1:  max|-R₁f - riesz1_round| = {np.abs(riesz1_direct - riesz1_round).max():.6e}")  # noqa: T201
print(f"Riesz2:  max|-R₂f - riesz2_round| = {np.abs(riesz2_direct - riesz2_round).max():.6e}")  # noqa: T201
print()  # noqa: T201
print("For consistency, all max differences should be < 1e-4")  # noqa: T201
print("=" * 60)  # noqa: T201

# %%
print("\nMonogenic curvelet visualization complete.")  # noqa: T201
