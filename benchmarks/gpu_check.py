"""GPU check for a branch: run the torch test suite, then a benchmark subset.

Run it on a machine with a CUDA GPU::

    python benchmarks/gpu_check.py --export benchmarks/data/check.json

Extra arguments go to ``gpu_bench.py``, except ``--profile``, which also prints
per-kernel times (``profile_plan.py``) for a large 2D and a 3D configuration.
Exits non-zero if the tests fail (the benchmark still runs, for diagnosis).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    args = sys.argv[1:]
    want_profile = "--profile" in args
    args = [a for a in args if a != "--profile"]
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/torch",
            "tests/numpy_torch_comparison",
            "-q",
            "-rs",
        ],
        cwd=ROOT,
        check=False,
    )
    bench = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "gpu_bench.py"),
            "--dtypes",
            "float32",
            "--no-cpu-ref",
            *args,
        ],
        cwd=ROOT,
        check=False,
    )
    if want_profile:
        for extra in (
            ["--shape", "2048,2048", "--scales", "5", "--wpd", "6"],
            ["--shape", "128,128,128", "--scales", "3", "--wpd", "3"],
        ):
            subprocess.run(
                [sys.executable, str(ROOT / "benchmarks" / "profile_plan.py"), *extra],
                cwd=ROOT,
                check=False,
            )
    return tests.returncode or bench.returncode


if __name__ == "__main__":
    sys.exit(main())
