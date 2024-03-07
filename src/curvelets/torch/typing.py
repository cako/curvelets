from __future__ import annotations

import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


from torch import Tensor

if sys.version_info <= (3, 9):
    from typing import List  # noqa: UP035

    UDCTCoefficients: TypeAlias = List[List[List[Tensor]]]  # noqa: UP006
    UDCTWindows: TypeAlias = List[List[List[List[Tensor]]]]  # noqa: UP006
else:
    # torch does not have a generic Complex Tensor
    UDCTCoefficients: TypeAlias = list[list[list[Tensor]]]
    UDCTWindows: TypeAlias = list[list[list[list[Tensor]]]]
