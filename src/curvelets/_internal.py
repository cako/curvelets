from __future__ import annotations

__all__ = ["MATPLOTLIB_ENABLED", "TORCH_ENABLED"]

from importlib.util import find_spec

MATPLOTLIB_ENABLED = find_spec("matplotlib") is not None
TORCH_ENABLED = find_spec("torch") is not None
