# Curvelets C++ (`libtorch`) Runtime

Standalone C++ `curvelets::UDCT` (`torch::nn::Module`) for running and training models containing the Uniform Discrete Curvelet Transform in C++ without a Python runtime.

## 1. Export a UDCT plan from Python

```python
import torch
from curvelets.torch import UDCT

udct = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3, dtype=torch.float32)
udct.save_plan("udct_plan.pt")
```

## 2. Build the C++ library with CMake

Requires CMake >= 3.18, a C++20 compiler, and `libtorch` (standalone or from a PyTorch installation).

```bash
cmake -S cpp -B cpp/build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build cpp/build --config Release -j
```

### Using with CMake `FetchContent`

Downstream CMake projects can pull `curvelets` directly without installing Python or distributing prebuilt binaries:

```cmake
include(FetchContent)
FetchContent_Declare(
    curvelets
    GIT_REPOSITORY https://github.com/cako/curvelets.git
    GIT_TAG v1.4  # or another release tag / commit
    SOURCE_SUBDIR cpp
)
FetchContent_MakeAvailable(curvelets)

target_link_libraries(my_app PRIVATE curvelets::torch)
```

When consumed via `FetchContent`, tests, examples, and install rules are disabled by default.

## 3. Use and train in pure C++ (`libtorch`)

`udct->forward(x)` returns the flat `(*batch, ncoef)` coefficient layout (matching `forward(x, structured=False)` in Python), and `udct->backward(coeffs)` returns `(*batch, *shape)` (or `(ndim + 1, *batch, *shape)` for monogenic transforms).

```cpp
#include "curvelets/udct.hpp"

int main() {
    curvelets::UDCT udct("udct_plan.pt");

    torch::Tensor x = torch::randn({4, 64, 64}, torch::requires_grad());
    torch::Tensor coeffs = udct->forward(x);
    torch::Tensor recon = udct->backward(coeffs);
    recon.sum().backward();
}
```

See `examples/train_model.cpp` for a complete C++ training loop with `torch::optim::Adam`.

## Capabilities and Limitations

* **Supported:** Inference and training (1st- and 2nd-order adjoint autograd, `.to(device)` / `.to(dtype)`, real/complex/monogenic transforms, N-D) in a Python-free C++ application on Linux (GCC/Clang), macOS (AppleClang), and Windows (MSVC).
* **Limitations:**
  * Plans are generated once in Python per transform configuration (`shape`, `num_scales` / `angular_wedges_config`, `high_frequency_mode`, `transform_kind`, etc.) and loaded from `.pt` files in C++.
  * Built from source with CMake (no prebuilt C++ binaries yet).
  * On Windows, `curvelets_torch.dll` and `libtorch`'s DLLs (`torch/lib`) must be placed alongside the executable or added to `PATH`.
