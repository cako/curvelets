"""GPU check for a branch: run the torch test suite, then a benchmark subset.

Run it on a machine with a CUDA GPU::

    python benchmarks/gpu_check.py --export benchmarks/data/check.json

Extra arguments go to ``gpu_bench.py``. Exits non-zero if the tests fail (the
benchmark still runs, for diagnosis).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
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
            *sys.argv[1:],
        ],
        cwd=ROOT,
        check=False,
    )
    return tests.returncode or bench.returncode


if __name__ == "__main__":
    sys.exit(main())
