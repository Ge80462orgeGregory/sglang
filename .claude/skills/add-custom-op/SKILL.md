# Skill: Add Custom Op to SGLang

This skill guides you through adding a custom PyTorch operator to SGLang, covering both the Python-side registration and the C++/CUDA backend integration.

## Overview

Custom ops in SGLang follow a consistent pattern:
1. Define the op schema and register it with `torch.library`
2. Implement CPU fallback (for testing/correctness)
3. Implement CUDA kernel (for performance)
4. Write unit tests covering correctness, edge cases, and error handling

## File Layout

```
sgl-kernel/
  src/
    sgl-kernel/
      csrc/
        my_op.cu          # CUDA kernel implementation
        my_op.h           # Header with declarations
      ops/
        my_op.py          # Python op registration
  tests/
    test_my_op.py         # Unit tests
```

## Step-by-Step Guide

### 1. Define the Python Op

```python
# sgl-kernel/src/sgl-kernel/ops/my_op.py
import torch
from torch import Tensor
from typing import Optional

# Register op schema
@torch.library.custom_op("sglang::my_op", mutates_args=())
def my_op(x: Tensor, scale: float) -> Tensor:
    """My custom op: scales input tensor by a scalar."""
    ...

@my_op.register_fake
def _(x: Tensor, scale: float) -> Tensor:
    return torch.empty_like(x)
```

### 2. Implement CPU Fallback

Always provide a CPU implementation for correctness testing:

```python
@torch.library.register_kernel("sglang::my_op", "cpu")
def my_op_cpu(x: Tensor, scale: float) -> Tensor:
    return x * scale
```

### 3. Implement CUDA Kernel

```cuda
// sgl-kernel/src/sgl-kernel/csrc/my_op.cu
#include "my_op.h"
#include <torch/extension.h>

__global__ void my_op_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] * scale;
    }
}

torch::Tensor my_op_cuda(torch::Tensor x, float scale) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    my_op_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        scale, n
    );
    return out;
}
```

### 4. Register CUDA Kernel

```python
@torch.library.register_kernel("sglang::my_op", "cuda")
def my_op_cuda_py(x: Tensor, scale: float) -> Tensor:
    from sgl_kernel._C import my_op_cuda
    return my_op_cuda(x, scale)
```

### 5. Write Tests

```python
# tests/test_my_op.py
import pytest
import torch
from sgl_kernel.ops.my_op import my_op

def torch_my_op(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Reference implementation using pure PyTorch."""
    return x * scale


def test_my_op_correctness():
    """Verify CUDA output matches reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    scale = 2.5
    out = my_op(x, scale)
    ref = torch_my_op(x.cpu(), scale).cuda()
    torch.testing.assert_close(out, ref)


def test_my_op_cpu_error():
    """Ensure CPU tensors raise an error when CUDA is expected."""
    x = torch.randn(128, device="cpu")
    with pytest.raises(RuntimeError):
        my_op(x, 1.0)  # Should fail if CUDA-only path is enforced


def test_my_op_shape_preserved():
    """Output shape must match input shape."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.randn(4, 8, 16, device="cuda")
    out = my_op(x, 3.0)
    assert out.shape == x.shape


def test_my_op_zero_scale():
    """Scaling by zero should produce a zero tensor."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.randn(256, device="cuda")
    out = my_op(x, 0.0)
    torch.testing.assert_close(out, torch.zeros_like(x))
```

## Common Pitfalls

- **Dtype handling**: Always check and handle `float16`, `bfloat16`, and `float32` separately in CUDA kernels.
- **In-place vs out-of-place**: Use `mutates_args=("out",)` if your op writes to an existing buffer.
- **Shape validation**: Add explicit shape checks before launching kernels to avoid cryptic CUDA errors.
- **Memory alignment**: Ensure tensors are contiguous with `.contiguous()` before passing raw pointers.

## Testing Checklist

- [ ] Correctness against PyTorch reference
- [ ] CPU error handling (or CPU fallback if supported)
- [ ] Shape preservation
- [ ] Edge cases (zero, empty tensor, large tensor)
- [ ] Multiple dtypes if applicable
- [ ] Benchmark vs baseline if performance-critical

## References

- [PyTorch Custom Ops Tutorial](https://pytorch.org/tutorials/advanced/custom_ops_landing.html)
- Existing skills: `add-triton-kernel`, `add-jit-kernel`, `add-sgl-kernel`
- SGLang kernel source: `sgl-kernel/src/sgl-kernel/csrc/`
