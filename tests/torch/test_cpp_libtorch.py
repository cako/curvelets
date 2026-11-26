"""Tests for C++ (libtorch) plan export and standalone C++ execution."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
import torch

from curvelets.torch import UDCT

_HAS_TOOLCHAIN = shutil.which("cmake") is not None and any(
    shutil.which(cc) is not None for cc in ("c++", "g++", "clang++", "cl")
)

pytestmark = pytest.mark.skipif(
    not _HAS_TOOLCHAIN,
    reason="CMake and a C++ compiler are required to build libtorch tests",
)

Kind = Literal["real", "complex", "monogenic"]
HFMode = Literal["curvelet", "wavelet"]


def _find_exe(build_dir: Path, name: str) -> Path:
    for candidate in (
        build_dir / name,
        build_dir / f"{name}.exe",
        build_dir / "Release" / f"{name}.exe",
    ):
        if candidate.is_file():
            return candidate
    msg = f"Executable {name} not found in {build_dir}"
    raise FileNotFoundError(msg)


@pytest.fixture(scope="session")  # type: ignore[untyped-decorator]
def cpp_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the C++ libtorch library and test/example executables once."""
    cmake_exe = shutil.which("cmake")
    assert cmake_exe is not None
    cpp_dir = Path(__file__).resolve().parents[2] / "cpp"
    build_dir = tmp_path_factory.mktemp("cpp_build")
    cxx11_abi = int(bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True)))  # pylint: disable=protected-access
    cfg_res = subprocess.run(
        [
            cmake_exe,
            "-S",
            str(cpp_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DCURVELETS_TORCH_CXX11_ABI={cxx11_abi}",
            "-DCURVELETS_BUILD_TESTS=ON",
            "-DCURVELETS_BUILD_EXAMPLES=ON",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cfg_res.returncode == 0, (
        f"CMake configure failed:\n{cfg_res.stdout}\n{cfg_res.stderr}"
    )
    build_res = subprocess.run(
        [cmake_exe, "--build", str(build_dir), "--config", "Release", "--parallel"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build_res.returncode == 0, (
        f"CMake build failed:\n{build_res.stdout}\n{build_res.stderr}"
    )
    return Path(build_dir)


def _make_input(
    shape: tuple[int, ...],
    batch: tuple[int, ...],
    kind: Kind,
    dtype: torch.dtype,
    seed: int = 0,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if kind == "complex":
        cdtype = torch.promote_types(dtype, torch.complex64)
        return torch.randn(*batch, *shape, dtype=cdtype, generator=gen)
    return torch.randn(*batch, *shape, dtype=dtype, generator=gen)


def test_save_plan_keeps_udct_stateless(tmp_path: Path) -> None:
    """save_plan writes a valid plan without mutating UDCT.state_dict() or _plan."""
    udct = UDCT(shape=(32, 32), num_scales=3, dtype=torch.float32)
    assert udct.state_dict() == {}
    # pylint: disable=protected-access
    assert udct._plan is None
    plan_path = udct.save_plan(tmp_path / "plan.pt")
    assert plan_path.is_file()
    assert udct.state_dict() == {}
    assert udct._plan is None
    # pylint: enable=protected-access


def test_cpp_standalone_suite(cpp_build: Path, tmp_path: Path) -> None:
    """Export plans/references from Python and run zero-Python C++ tests & example."""
    configs: list[
        tuple[str, tuple[int, ...], Kind, HFMode, torch.dtype, tuple[int, ...]]
    ] = [
        ("real_2d_f32", (32, 32), "real", "curvelet", torch.float32, (2,)),
        ("real_2d_f64", (32, 32), "real", "curvelet", torch.float64, ()),
        ("real_wavelet_2d_f32", (24, 36), "real", "wavelet", torch.float32, (2, 2)),
        ("complex_2d_f64", (24, 36), "complex", "curvelet", torch.float64, (2,)),
        ("monogenic_2d_f64", (32, 32), "monogenic", "curvelet", torch.float64, (2,)),
        ("real_3d_f32", (16, 16, 16), "real", "curvelet", torch.float32, (2,)),
        ("real_odd_2d_f64", (25, 27), "real", "curvelet", torch.float64, (2,)),
    ]

    for name, shape, kind, hf_mode, dtype, batch in configs:
        udct = UDCT(
            shape=shape,
            num_scales=3,
            wedges_per_direction=3,
            high_frequency_mode=hf_mode,
            transform_kind=kind,
            dtype=dtype,
        )
        udct.save_plan(tmp_path / f"{name}_plan.pt")
        x = _make_input(shape, batch, kind, dtype, seed=31).requires_grad_(True)
        y = udct.forward(x, structured=False)
        recon = udct.backward(y)
        (y.abs().pow(2).sum() + recon.abs().sum()).backward()
        assert x.grad is not None

        xs = x.detach().clone().requires_grad_(True)
        coeffs_ref = udct.forward(xs, structured=True)
        loss_struct = (
            sum(c.abs().pow(2).sum() for s in coeffs_ref for d in s for c in d)
            + udct.backward(coeffs_ref).abs().sum()
        )
        loss_struct.backward()
        assert xs.grad is not None

        flat_wedges = [c.detach() for s in coeffs_ref for d in s for c in d]
        ref_dict: dict[str, torch.Tensor] = {
            "x": x.detach(),
            "y": y.detach(),
            "recon": recon.detach(),
            "grad": x.grad.detach(),
            "struct_num_wedges": torch.tensor(len(flat_wedges), dtype=torch.int64),
            "struct_grad": xs.grad.detach(),
        }
        for idx, wedge in enumerate(flat_wedges):
            ref_dict[f"struct_w_{idx}"] = wedge
        torch.save(ref_dict, tmp_path / f"{name}_ref.pt")

    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"}
    }
    test_exe = _find_exe(cpp_build, "test_udct_cpp")
    if sys.platform == "win32":
        torch_lib = str(Path(torch.__file__).resolve().parent / "lib")
        exe_dir = str(test_exe.parent)
        existing_path = clean_env.get("PATH", "")
        clean_env["PATH"] = os.pathsep.join(
            p for p in (exe_dir, torch_lib, existing_path) if p
        )
    res = subprocess.run(
        [str(test_exe), str(tmp_path)],
        env=clean_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"C++ test failed:\n{res.stdout}\n{res.stderr}"

    example_exe = _find_exe(cpp_build, "train_model")
    ex_res = subprocess.run(
        [str(example_exe), str(tmp_path / "real_2d_f32_plan.pt")],
        env=clean_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ex_res.returncode == 0, (
        f"C++ example failed:\n{ex_res.stdout}\n{ex_res.stderr}"
    )
