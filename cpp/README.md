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

```bash
cmake -S cpp -B cpp/build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build cpp/build --config Release -j
```

## 3. Use and train in pure C++ (`libtorch`)

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
