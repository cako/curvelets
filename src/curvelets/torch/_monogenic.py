"""Monogenic curvelet transform on top of the fused real transform.

The monogenic coefficients of a real image :math:`f` are built from real
transforms, so they run on the fused plan (one batched call per direction) and
inherit its gradients (through the adjoint transform) and CUDA graphs:

- the scalar channels are the real transform of :math:`f` (real and imaginary
  parts);
- the Riesz channel :math:`k` is the real part of the real-transform analysis
  of the spectrum :math:`\\hat f R_k`. With :math:`g_k = \\mathcal{F}^{-1}(\\hat f
  R_k)`, that is :math:`\\operatorname{Re} A(\\operatorname{Re} g_k) -
  \\operatorname{Im} A(\\operatorname{Im} g_k)` by linearity, where :math:`A` is
  the real transform. :math:`g_k` is real unless axis :math:`k` has even length
  (:math:`R_k` is not Hermitian on its Nyquist plane), so the second term is
  only computed for those axes.

The backward transform synthesizes each component with the real backward
transform (the tight frame property), giving :math:`(f, -R_1 f, \\ldots, -R_n f)`.
The coefficient layout (and the :meth:`UDCT.vect` order) matches
:class:`curvelets.numpy.UDCT`: each wedge is ``(*wedge_shape, ndim + 2)`` with
channels ``[scalar.real, scalar.imag, riesz_1, ..., riesz_ndim]``, and the flat
vector holds each wedge's channels one after the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ._plan import UDCTPlan
from ._riesz import riesz_filters
from .typing import UDCTCoefficients


@dataclass(frozen=True)
class _Wedge:
    start: int  # offset of the wedge's first channel in the monogenic vector
    shape: tuple[int, ...]


@dataclass
class MonogenicLayout:
    """Index maps between the real plan's coefficients and the monogenic vector.

    ``perm`` gathers the monogenic vector from the channel-major
    ``(num_channels * plan.ncoef)`` buffer ``[re(c), im(c), riesz_1, ...]``;
    ``inverse`` gathers that buffer back from the monogenic vector.
    """

    num_channels: int
    ncoef: int
    wedges: list[list[list[_Wedge]]]
    perm: torch.Tensor
    inverse: torch.Tensor

    @classmethod
    def build(cls, plan: UDCTPlan, num_channels: int) -> MonogenicLayout:
        """Layout for ``plan`` with ``num_channels`` channels per wedge."""
        # (start, size, shape) of every wedge in the plan's vector, nested in
        # the same order as UDCTPlan.unflatten.
        spans: list[list[list[tuple[int, int, tuple[int, ...]]]]] = [
            [[] for _ in range(ndir)] for ndir in plan.num_directions
        ]
        entry = 0
        for group in plan.groups:
            for i in range(group.count):
                scale_idx, dir_idx, _ = plan.layout[entry]
                spans[scale_idx][dir_idx].append((
                    group.offset + i * group.size,
                    group.size,
                    group.shape,
                ))
                entry += 1

        pieces: list[torch.Tensor] = []
        wedges: list[list[list[_Wedge]]] = []
        position = 0
        for scale_spans in spans:
            wedges.append([])
            for dir_spans in scale_spans:
                wedges[-1].append([])
                for start, size, shape in dir_spans:
                    wedges[-1][-1].append(_Wedge(position, shape))
                    channel_starts = (
                        torch.arange(num_channels, dtype=torch.int64) * plan.ncoef
                        + start
                    )
                    pieces.append(
                        (
                            channel_starts[:, None]
                            + torch.arange(size, dtype=torch.int64)
                        ).flatten()
                    )
                    position += num_channels * size
        perm = torch.cat(pieces)
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(perm.numel(), dtype=torch.int64)
        return cls(num_channels, perm.numel(), wedges, perm, inverse)

    def to(self, device: torch.device) -> MonogenicLayout:
        """Layout with its index maps on ``device``."""
        if self.perm.device == device:
            return self
        return MonogenicLayout(
            self.num_channels,
            self.ncoef,
            self.wedges,
            self.perm.to(device),
            self.inverse.to(device),
        )

    def unflatten(self, vector: torch.Tensor) -> UDCTCoefficients:
        """Views ``(*batch, *wedge_shape, num_channels)`` of a monogenic vector."""
        batch = vector.shape[:-1]
        coeffs: UDCTCoefficients = []
        for scale_wedges in self.wedges:
            coeffs.append([])
            for dir_wedges in scale_wedges:
                coeffs[-1].append([])
                for wedge in dir_wedges:
                    stop = wedge.start + self.num_channels * math.prod(wedge.shape)
                    block = vector[..., wedge.start : stop].reshape(
                        *batch, self.num_channels, *wedge.shape
                    )
                    coeffs[-1][-1].append(block.movedim(len(batch), -1))
        return coeffs

    def flatten(self, coefficients: UDCTCoefficients) -> torch.Tensor:
        """Monogenic vector ``(*batch, ncoef)`` of nested coefficients."""
        parts: list[torch.Tensor] = []
        for scale_wedges, scale_coeffs in zip(self.wedges, coefficients, strict=True):
            for dir_wedges, dir_coeffs in zip(scale_wedges, scale_coeffs, strict=True):
                for wedge, coeff in zip(dir_wedges, dir_coeffs, strict=True):
                    expected = (*wedge.shape, self.num_channels)
                    if tuple(coeff.shape[coeff.ndim - len(expected) :]) != expected:
                        msg = (
                            "Expected monogenic coefficients of shape "
                            f"(*batch, {', '.join(map(str, expected))}), "
                            f"got {tuple(coeff.shape)}."
                        )
                        raise ValueError(msg)
                    batch = coeff.shape[: coeff.ndim - len(expected)]
                    channels_first = coeff.movedim(-1, len(batch))
                    parts.append(channels_first.reshape(*batch, -1))
        return torch.cat(parts, dim=-1)


def riesz_spectra(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Riesz filters ``(ndim, *shape)`` in the complex dtype matching ``dtype``."""
    filters = torch.stack(riesz_filters(shape))
    return filters.to(device=device, dtype=dtype.to_complex())


def riesz_images(
    image: torch.Tensor, filters: torch.Tensor
) -> tuple[torch.Tensor, list[int]]:
    """Inputs to the real transform for the monogenic forward transform.

    Parameters
    ----------
    image : torch.Tensor
        Real input of shape ``(*batch, *shape)``.
    filters : torch.Tensor
        Riesz filters of shape ``(ndim, *shape)``.

    Returns
    -------
    tuple[torch.Tensor, list[int]]
        ``(*batch, 1 + ndim + len(even), *shape)`` real images
        ``[f, Re g_1, ..., Re g_ndim, Im g_k for k in even]`` and ``even``,
        the axes of even length.
    """
    ndim = filters.shape[0]
    dims = tuple(range(-ndim, 0))
    spectrum = torch.fft.fftn(image, dim=dims)  # pylint: disable=not-callable
    riesz = torch.fft.ifftn(spectrum.unsqueeze(-ndim - 1) * filters, dim=dims)  # pylint: disable=not-callable
    even = [k for k, n in enumerate(image.shape[-ndim:]) if n % 2 == 0]
    parts = [image.unsqueeze(-ndim - 1), riesz.real]
    if even:
        index = torch.tensor(even, device=image.device)
        parts.append(riesz.imag.index_select(-ndim - 1, index))
    return torch.cat(parts, dim=-ndim - 1), even


def channels_from_real(
    coefficients: torch.Tensor, ndim: int, even: list[int]
) -> torch.Tensor:
    """Monogenic channels from the real transform of :func:`riesz_images`.

    Parameters
    ----------
    coefficients : torch.Tensor
        Complex ``(*batch, 1 + ndim + len(even), ncoef)`` real-transform
        coefficients.
    ndim : int
        Number of dimensions.
    even : list[int]
        Axes of even length (see :func:`riesz_images`).

    Returns
    -------
    torch.Tensor
        Real ``(*batch, ndim + 2, ncoef)`` channels
        ``[re(c), im(c), riesz_1, ..., riesz_ndim]``.
    """
    scalar = coefficients[..., 0, :]
    channels = [scalar.real, scalar.imag]
    for k in range(ndim):
        riesz = coefficients[..., 1 + k, :].real
        if k in even:
            riesz = riesz - coefficients[..., 1 + ndim + even.index(k), :].imag
        channels.append(riesz)
    return torch.stack(channels, dim=-2)


def real_from_channels(channels: torch.Tensor) -> torch.Tensor:
    """Complex inputs ``(*batch, ndim + 1, ncoef)`` to the real backward transform.

    The inverse of the layout of :func:`channels_from_real` (for its
    ``(*batch, ndim + 2, ncoef)`` real channels): the complex scalar, then each
    Riesz channel as a complex tensor.
    """
    scalar = torch.complex(channels[..., 0, :], channels[..., 1, :])
    riesz = channels[..., 2:, :]
    return torch.cat(
        [scalar.unsqueeze(-2), torch.complex(riesz, torch.zeros_like(riesz))], dim=-2
    )
