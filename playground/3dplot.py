# %%
from __future__ import annotations

from copy import deepcopy

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from curvelets.numpy.udct import SimpleUDCT

# %%
ax = plt.figure().add_subplot(projection="3d")
X, Y, Z = axes3d.get_test_data(0.05)
ax.contourf(X, Y, Z, cmap=cm.coolwarm)

# %%
nscales = 3
nbands = 3

shape = (32, 32, 32)
zeros_img = np.zeros(shape)
cfg = np.array([[3, 3], [6, 3]])
C = SimpleUDCT(shape=shape, nscales=nscales, nbands_per_direction=nbands)
coeffs_zero = C.forward(zeros_img)


# %%
ires = 1
idir = 0
iang = 0
icenters = tuple((np.asarray(coeffs_zero[ires][idir][iang].shape) + 1) // 2)

coeffs_curv = deepcopy(coeffs_zero)
coeffs_curv[1][0][0][icenters] = 1
curv = C.backward(coeffs_curv)

# %%
X, Y, Z = np.meshgrid(*[np.arange(s, dtype=float) for s in shape], indexing="ij")
abscurv = np.abs(curv)
curvmax = abscurv.max()
curvmin = abscurv.min()
keep = abscurv >= 0.1 * curvmax
# color = abscurv[keep]
# color /= color.max()

# %%
# fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
# surf = ax.scatter(X[keep], Y[keep], Z[keep], linewidth=0.2, antialiased=True)

# %%
# # from matplotlib import cm
# # from matplotlib.colors import LightSource
# # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

# keep = (0.85 * abscurv.max() > abscurv) & (abscurv > 0.5 * abscurv.max())

# surf = ax.plot_trisurf(X[keep], Y[keep], Z[keep], linewidth=0, antialiased=True)

# %%
norm = mpl.colors.Normalize(0, curvmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap="turbo")

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
surf = ax.scatter(
    X[keep],
    Y[keep],
    Z[keep],
    color=mapper.to_rgba(
        abscurv[keep], np.sqrt((abscurv[keep] - curvmin) / (curvmax - curvmin))
    ),
    linewidth=0.2,
    antialiased=True,
)
plt.show()

# %%
import pyvista as pv

pv.se

keep2 = (0.85 * abscurv.max() > abscurv) & (abscurv > 0.5 * abscurv.max())

points = np.c_[X[keep2], Y[keep2], Z[keep2]]
cloud = pv.PolyData(points)
cloud.plot()

# volume = cloud.delaunay_3d(alpha=2.0)
# shell = volume.extract_geometry()
# shell.plot()
