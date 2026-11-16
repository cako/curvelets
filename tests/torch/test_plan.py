"""Tests for the fused execution plan (curvelets.torch._plan)."""

from __future__ import annotations

import copy
import io
import pickle
from typing import Literal

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from curvelets.torch import UDCT, UDCTModule
from curvelets.torch._backward_transform import _apply_backward_transform
from curvelets.torch._foldmap import MAX_LEVELS, GatherFold, ScatterFold
from curvelets.torch._forward_transform import (
    _apply_forward_transform_complex,
    _apply_forward_transform_real,
)
from curvelets.torch._plan import UDCTPlan

Kind = Literal["real", "complex"]
Mode = Literal["curvelet", "wavelet"]

CONFIGS = [
    ((32, 32), 3, 3),
    ((48, 40), 3, 6),
    ((64, 64), 4, 6),
    ((16, 16, 16), 3, 3),
]


def _use_strategy(udct: UDCT, strategy: str) -> UDCTPlan:
    """Install a plan with a forced fold strategy (the default follows the device)."""
    plan = UDCTPlan.build(
        udct._parameters,  # pylint: disable=protected-access
        udct.windows,
        udct._decimation_ratios_cpu,  # pylint: disable=protected-access
        complex_mode=udct._transform_kind == "complex",  # pylint: disable=protected-access
        fold_strategy=strategy,
    )
    udct._plan = plan  # pylint: disable=protected-access
    return plan


def _input(kind: Kind, shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=gen, dtype=torch.float64)
    if kind == "complex":
        x = torch.complex(x, torch.randn(shape, generator=gen, dtype=torch.float64))
    return x


@pytest.mark.parametrize(("shape", "num_scales", "wpd"), CONFIGS)  # type: ignore[misc]
@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[misc]
@pytest.mark.parametrize("mode", ["curvelet", "wavelet"])  # type: ignore[misc]
@pytest.mark.parametrize("strategy", ["scatter", "gather"])  # type: ignore[misc]
def test_plan_matches_per_wedge_reference(
    shape: tuple[int, ...],
    num_scales: int,
    wpd: int,
    kind: Kind,
    mode: Mode,
    strategy: str,
) -> None:
    """Both fold strategies equal the per-wedge reference implementation."""
    udct = UDCT(
        shape=shape,
        num_scales=num_scales,
        wedges_per_direction=wpd,
        transform_kind=kind,
        high_frequency_mode=mode,
    )
    _use_strategy(udct, strategy)
    params = udct._parameters  # pylint: disable=protected-access
    decim = udct._decimation_ratios_cpu  # pylint: disable=protected-access
    x = _input(kind, shape)

    reference_fwd = (
        _apply_forward_transform_complex
        if kind == "complex"
        else _apply_forward_transform_real
    )
    expected = reference_fwd(x, params, udct.windows, decim)
    got = udct.forward(x)
    for s_exp, s_got in zip(expected, got, strict=True):
        for d_exp, d_got in zip(s_exp, s_got, strict=True):
            for w_exp, w_got in zip(d_exp, d_got, strict=True):
                torch.testing.assert_close(w_got, w_exp, rtol=1e-12, atol=1e-12)

    expected_bwd = _apply_backward_transform(
        expected, params, udct.windows, decim, use_complex_transform=kind == "complex"
    )
    torch.testing.assert_close(udct.backward(got), expected_bwd, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[misc]
def test_plan_batched_matches_single(kind: Kind) -> None:
    """Leading batch dimensions give the same result as a Python loop."""
    udct = UDCT(
        shape=(32, 32), num_scales=3, wedges_per_direction=3, transform_kind=kind
    )
    plan = udct._get_plan()  # pylint: disable=protected-access
    batch = torch.stack([_input(kind, (32, 32), seed) for seed in range(3)]).reshape(
        3, 1, 32, 32
    )
    flat = plan.forward_flat(batch)
    assert flat.shape == (3, 1, plan.ncoef)
    for i in range(3):
        torch.testing.assert_close(flat[i, 0], udct.vect(udct.forward(batch[i, 0])))
    recon = plan.backward_flat(flat)
    torch.testing.assert_close(recon, batch)


def test_vect_fast_path_matches_vect_forward() -> None:
    """``UDCT._forward_vect`` equals ``vect(forward(x))``."""
    udct = UDCT(shape=(48, 40), num_scales=3, wedges_per_direction=6)
    x = _input("real", (48, 40))
    torch.testing.assert_close(
        udct._forward_vect(x),  # pylint: disable=protected-access
        udct.vect(udct.forward(x)),
    )


def test_udct_module_vmap_matches_loop() -> None:
    """``torch.vmap`` over ``UDCTModule`` still works with the fused plan."""
    module = UDCTModule(shape=(28, 28), num_scales=2, wedges_per_direction=3)
    batch = torch.randn(4, 28, 28, dtype=torch.float64)
    batched = torch.vmap(module)(batch)
    for i in range(4):
        torch.testing.assert_close(batched[i], module(batch[i]))


@pytest.mark.parametrize(
    ("shape", "wpd"), [((64, 64), 3), ((64, 64), 6), ((16, 16, 16), 3)]
)  # type: ignore[misc]
def test_launch_count_does_not_grow_with_wedges(
    shape: tuple[int, ...], wpd: int
) -> None:
    """GPU strategy: gathers per transform are bounded by fold levels, not wedges."""
    udct = UDCT(shape=shape, num_scales=3, wedges_per_direction=wpd)
    plan = _use_strategy(udct, "gather")
    assert isinstance(plan.fwd_map, GatherFold)
    assert isinstance(plan.bwd_map, GatherFold)
    assert isinstance(plan.bwd_conj_map, GatherFold)
    assert len(plan.fwd_map.levels) <= MAX_LEVELS + 1
    assert len(plan.bwd_map.levels) <= MAX_LEVELS + 1
    conj_levels = 1 + len(plan.bwd_conj_map.levels)
    x = _input("real", shape)
    coeffs = udct.forward(x)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        udct.forward(x)
    names = [event.name for event in prof.events()]
    assert names.count("aten::index_select") == 1 + len(plan.fwd_map.levels)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        udct.backward(coeffs)
    names = [event.name for event in prof.events()]
    assert names.count("aten::index_select") == (
        1 + len(plan.bwd_map.levels) + conj_levels
    )


def test_cpu_defaults_to_scatter_folds() -> None:
    """CPU plans use scatter folds (blocked backward); other devices get gather folds."""
    plan = UDCT(shape=(32, 32), num_scales=3, wedges_per_direction=3)._get_plan()  # pylint: disable=protected-access
    assert isinstance(plan.fwd_map, ScatterFold)
    assert plan.bwd_map is None
    assert plan.cpu_chunks
    assert all(
        isinstance(fold, ScatterFold) for chunk in plan.cpu_chunks for fold in chunk.bwd
    )


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[misc]
def test_small_cpu_chunks_match_gather(
    kind: Kind, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Splitting groups into many CPU chunks gives the same result as one gather fold."""
    import curvelets.torch._plan as plan_module

    monkeypatch.setattr(plan_module, "CPU_CHUNK_ELEMENTS", 64)
    udct = UDCT(
        shape=(64, 64), num_scales=4, wedges_per_direction=6, transform_kind=kind
    )
    x = _input(kind, (64, 64)).unsqueeze(0).repeat(2, 1, 1)
    plan = _use_strategy(udct, "scatter")
    assert plan.cpu_chunks is not None
    assert len(plan.cpu_chunks) > len(plan.groups)
    blocked = udct._forward_vect(x)  # pylint: disable=protected-access
    blocked_back = udct._backward_vect(blocked)  # pylint: disable=protected-access
    _use_strategy(udct, "gather")
    torch.testing.assert_close(blocked, udct._forward_vect(x))  # pylint: disable=protected-access
    torch.testing.assert_close(blocked_back, udct._backward_vect(blocked))  # pylint: disable=protected-access


@pytest.mark.parametrize("fold_cls", [GatherFold, ScatterFold])  # type: ignore[misc]
@pytest.mark.parametrize("batch", [1, 3])  # type: ignore[misc]
def test_fold_matches_dense_scatter(
    fold_cls: type[GatherFold | ScatterFold], batch: int
) -> None:
    """Both folds equal a plain scatter-add (GatherFold: incl. the padded tail)."""
    gen = torch.Generator().manual_seed(3)
    num_targets, num_sources, n = 50, 40, 600  # ~12 contributions per target
    targets = torch.randint(0, num_targets, (n,), generator=gen)
    sources = torch.randint(0, num_sources, (n,), generator=gen)
    weights = torch.rand(n, generator=gen, dtype=torch.float64)
    fold = fold_cls.build(targets, sources, weights, num_targets, num_sources)
    if isinstance(fold, GatherFold):
        assert any(level.width > 1 for level in fold.levels)

    src = torch.randn(batch, num_sources, dtype=torch.complex128, generator=gen)
    expected = torch.zeros(batch, num_targets, dtype=torch.complex128).index_add(
        1, targets, src[:, sources] * weights
    )
    torch.testing.assert_close(fold.apply(src), expected)


def test_odd_shape_half_spectrum_matches_reference() -> None:
    """Odd last dimension (no Nyquist bin): the half-spectrum plan matches the reference.

    (Shapes not divisible by the decimation ratios do not reconstruct perfectly
    in any backend, so this compares against the reference, not the input.)
    """
    shape = (48, 51)
    udct = UDCT(shape=shape, num_scales=3, wedges_per_direction=3)
    params = udct._parameters  # pylint: disable=protected-access
    decim = udct._decimation_ratios_cpu  # pylint: disable=protected-access
    x = _input("real", shape)
    expected = _apply_forward_transform_real(x, params, udct.windows, decim)
    got = udct.forward(x)
    torch.testing.assert_close(
        udct.vect(got), udct.vect(expected), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        udct.backward(got),
        _apply_backward_transform(expected, params, udct.windows, decim),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[misc]
def test_transform_is_bitwise_deterministic(kind: Kind) -> None:
    """Repeated calls give bitwise-identical results (no atomics in the folds)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    udct = UDCT(
        shape=(64, 64), num_scales=4, wedges_per_direction=6, transform_kind=kind
    )
    udct.apply_to_tensors(lambda t: t.to(device))
    x = _input(kind, (64, 64)).to(device)
    first = udct.vect(udct.forward(x))
    recon = udct.backward(udct.forward(x))
    for _ in range(3):
        assert torch.equal(udct.vect(udct.forward(x)), first)
        assert torch.equal(udct.backward(udct.forward(x)), recon)


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[misc]
def test_direct_and_autograd_paths_agree(kind: Kind) -> None:
    """The out= fast path and the differentiable fallback give the same result."""
    udct = UDCT(
        shape=(48, 40), num_scales=3, wedges_per_direction=6, transform_kind=kind
    )
    x = _input(kind, (48, 40))
    fast = udct._forward_vect(x)  # pylint: disable=protected-access
    slow = udct._forward_vect(x.clone().requires_grad_(True))  # pylint: disable=protected-access
    torch.testing.assert_close(slow.detach(), fast, rtol=0, atol=0)
    back_fast = udct._backward_vect(fast)  # pylint: disable=protected-access
    back_slow = udct._backward_vect(fast.clone().requires_grad_(True))  # pylint: disable=protected-access
    torch.testing.assert_close(back_slow.detach(), back_fast, rtol=0, atol=0)


def test_backward_accepts_real_coefficients() -> None:
    """Real-valued coefficients (e.g. after taking .real) are promoted to complex."""
    udct = UDCT(shape=(32, 32), num_scales=3, wedges_per_direction=3)
    vec = udct._forward_vect(_input("real", (32, 32))).real  # pylint: disable=protected-access
    torch.testing.assert_close(
        udct._backward_vect(vec),  # pylint: disable=protected-access
        udct._backward_vect(vec.to(torch.complex128)),  # pylint: disable=protected-access
    )


def test_cpu_scratch_reuse_does_not_leak_between_calls() -> None:
    """Reused CPU scratch buffers never leak a previous call into the result."""
    udct = UDCT(shape=(32, 32), num_scales=3, wedges_per_direction=3)
    x1 = _input("real", (32, 32), seed=1)
    x2 = _input("real", (32, 32), seed=2)
    first = udct._forward_vect(x1).clone()  # pylint: disable=protected-access
    second = udct._forward_vect(x2)  # pylint: disable=protected-access
    torch.testing.assert_close(udct._forward_vect(x1), first)  # pylint: disable=protected-access
    fresh = UDCT(shape=(32, 32), num_scales=3, wedges_per_direction=3)
    torch.testing.assert_close(second, fresh._forward_vect(x2))  # pylint: disable=protected-access
    batch = torch.stack([x1, x2])
    torch.testing.assert_close(udct._forward_vect(batch)[0], first)  # pylint: disable=protected-access


@pytest.mark.parametrize("fold_cls", [GatherFold, ScatterFold])  # type: ignore[misc]
def test_fold_casts_weights_once_for_single_precision(
    fold_cls: type[GatherFold | ScatterFold],
) -> None:
    """float64 weights are cast (and cached) for complex64 inputs, on both folds."""
    gen = torch.Generator().manual_seed(5)
    num_targets, num_sources, n = 30, 20, 300
    targets = torch.randint(0, num_targets, (n,), generator=gen)
    sources = torch.randint(0, num_sources, (n,), generator=gen)
    weights = torch.rand(n, generator=gen, dtype=torch.float64)
    fold = fold_cls.build(targets, sources, weights, num_targets, num_sources)
    src = torch.randn(2, num_sources, dtype=torch.complex64, generator=gen)
    expected = torch.zeros(2, num_targets, dtype=torch.complex64).index_add(
        1, targets, src[:, sources] * weights.to(torch.float32)
    )
    first = fold.apply(src)
    cached = dict(fold._cast)  # pylint: disable=protected-access
    assert cached
    second = fold.apply(src)
    assert fold._cast.keys() == cached.keys()  # pylint: disable=protected-access
    assert all(
        fold._cast[k] is v  # pylint: disable=protected-access
        for k, v in cached.items()
    )
    assert first.dtype == torch.complex64
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, first)


@pytest.mark.parametrize("strategy", ["scatter", "gather"])  # type: ignore[misc]
def test_single_precision_transform_matches_double(strategy: str) -> None:
    """float32 inputs run end to end through both strategies (weights cast once)."""
    udct = UDCT(shape=(32, 32), num_scales=3, wedges_per_direction=3)
    _use_strategy(udct, strategy)
    x = _input("real", (32, 32))
    coeffs = udct._forward_vect(x.float())  # pylint: disable=protected-access
    assert coeffs.dtype == torch.complex64
    torch.testing.assert_close(
        coeffs,
        udct._forward_vect(x).to(torch.complex64),  # pylint: disable=protected-access
        atol=1e-5,
        rtol=1e-4,
    )
    recon = udct._backward_vect(coeffs)  # pylint: disable=protected-access
    assert recon.dtype == torch.float32
    torch.testing.assert_close(recon, x.float(), atol=1e-4, rtol=1e-4)


def test_plan_rejects_unknown_strategy() -> None:
    """An unknown fold strategy is an error, not a silent fallback."""
    udct = UDCT(shape=(32, 32), num_scales=3, wedges_per_direction=3)
    with pytest.raises(ValueError, match="unknown fold strategy"):
        _use_strategy(udct, "dense")


def test_plan_requires_periodized_maps() -> None:
    """Windows without attached periodized maps cannot be planned."""
    udct = UDCT(shape=(32, 32), num_scales=3, wedges_per_direction=3)
    windows = copy.deepcopy(udct.windows)
    windows[1][0][0].out_shape = None
    with pytest.raises(ValueError, match="periodized maps"):
        UDCTPlan.build(
            udct._parameters,  # pylint: disable=protected-access
            windows,
            udct._decimation_ratios_cpu,  # pylint: disable=protected-access
            complex_mode=False,
        )


def test_used_transform_can_be_copied_and_pickled() -> None:
    """A transform that has run (plan and CPU scratch built) can be deep-copied,
    pickled and saved with ``torch.save``."""
    module = UDCTModule(shape=(32, 32), num_scales=3, wedges_per_direction=3)
    x = _input("real", (32, 32)).to(torch.get_default_dtype())
    expected = module(x)
    for clone in (
        copy.deepcopy(module),
        pickle.loads(pickle.dumps(module)),
    ):
        torch.testing.assert_close(clone(x), expected)
    buffer = io.BytesIO()
    torch.save(module, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    torch.testing.assert_close(loaded(x), expected)
