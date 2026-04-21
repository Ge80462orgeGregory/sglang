# Skill: Add Triton Kernel to SGLang

This skill guides you through adding a new Triton kernel to SGLang, including writing the kernel, integrating it with the Python API, and writing tests.

## Overview

Triton kernels in SGLang follow a consistent pattern:
1. Write the Triton kernel function with `@triton.jit`
2. Wrap it in a Python launcher function
3. Add autotuning configs if needed
4. Write correctness and performance tests
5. Register the kernel in the appropriate module

## File Structure

```
sglang/
  sgl-kernel/
    src/
      sgl_kernel/
        triton/
          your_kernel.py       # Kernel implementation
    tests/
      test_your_kernel.py      # Tests
```

## Step 1: Write the Triton Kernel

Example: element-wise ReLU activation kernel.

```python
# sgl-kernel/src/sgl_kernel/triton/relu.py

import torch
import triton
import triton.language as tl


# Autotuning configurations — tune block size for different input shapes
autotune_configs = [
    triton.Config({"BLOCK_SIZE": 1024}),
    triton.Config({"BLOCK_SIZE": 2048}),
    triton.Config({"BLOCK_SIZE": 4096}),
]


@triton.autotune(configs=autotune_configs, key=["n_elements"])
@triton.jit
def _relu_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Compile-time block size
):
    """Triton kernel for element-wise ReLU: out = max(x, 0)."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def relu(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Element-wise ReLU using a Triton kernel.

    Args:
        x: Input tensor. Must be on CUDA and float16/float32/bfloat16.
        out: Optional pre-allocated output tensor with the same shape as x.
             If None, a new tensor is allocated.

    Returns:
        Tensor with ReLU applied element-wise.

    Raises:
        ValueError: If x is not on a CUDA device.
        ValueError: If out shape does not match x shape.
    """
    if not x.is_cuda:
        raise ValueError(f"Input tensor must be on CUDA, got device: {x.device}")

    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape:
        raise ValueError(
            f"Output shape {out.shape} does not match input shape {x.shape}"
        )

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731

    _relu_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
    )
    return out
```

## Step 2: Write Tests

```python
# sgl-kernel/tests/test_relu.py

import pytest
import torch

from sgl_kernel.triton.relu import relu


def torch_relu(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.relu(x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1024,), (4096,), (128, 256), (32, 64, 128)])
def test_relu_correctness(dtype, shape):
    """Test that Triton ReLU matches PyTorch reference."""
    x = torch.randn(shape, dtype=dtype, device="cuda")
    expected = torch_relu(x)
    actual = relu(x)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_relu_out_param():
    """Test that pre-allocated output tensor is used correctly."""
    x = torch.randn(1024, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    result = relu(x, out=out)
    assert result.data_ptr() == out.data_ptr(), "Should reuse pre-allocated buffer"
    torch.testing.assert_close(result, torch_relu(x))


def test_relu_cpu_error():
    """Test that CPU input raises ValueError."""
    x = torch.randn(1024, dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="must be on CUDA"):
        relu(x)


def test_relu_shape_mismatch_error():
    """Test that mismatched output shape raises ValueError."""
    x = torch.randn(1024, dtype=torch.float32, device="cuda")
    out = torch.empty(512, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="shape"):
        relu(x, out=out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_relu_negative_values(dtype):
    """Test that negative values are zeroed out."""
    x = torch.full((1024,), -1.0, dtype=dtype, device="cuda")
    out = relu(x)
    assert (out == 0).all(), "All negative values should become 0"


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_relu_positive_values(dtype):
    """Test that positive values are unchanged."""
    x = torch.full((1024,), 2.0, dtype=dtype, device="cuda")
    out = relu(x)
    torch.testing.assert_close(out, x)
```

## Step 3: Register the Kernel

Add the new kernel to `sgl_kernel/triton/__init__.py`:

```python
from sgl_kernel.triton.relu import relu

__all__ = ["relu"]
```

## Step 4: Run Tests

```bash
# Run correctness tests
pytest sgl-kernel/tests/test_relu.py -v

# Run with specific dtype
pytest sgl-kernel/tests/test_relu.py -v -k "float32"

# Run benchmarks (if using pytest-benchmark)
pytest sgl-kernel/tests/test_relu.py -v --benchmark-only
```

## Common Pitfalls

### 1. Incorrect Grid Calculation

```python
# WRONG: off-by-one for non-power-of-two sizes
grid = (n_elements // BLOCK_SIZE,)

# CORRECT: use triton.cdiv for ceiling division
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
```

### 2. Missing Mask in Load/Store

Always use a mask when the tensor size is not a multiple of BLOCK_SIZE:

```python
# WRONG: no mask — undefined behavior for out-of-bounds
x = tl.load(x_ptr + offsets)

# CORRECT: use mask to guard boundary accesses
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, x, mask=mask)
```

### 3. dtype Mismatch in Triton

Triton infers dtype from pointer type. Ensure Python-side tensors have the correct dtype before passing to the kernel:

```python
# Cast to expected dtype before kernel call
x = x.to(torch.float32)
```

### 4. Autotuning Key Selection

Choose autotuning keys that capture the dominant performance-affecting dimensions:

```python
# Good: key on the dimension that drives BLOCK_SIZE choice
@triton.autotune(configs=configs, key=["n_elements"])

# For matrix ops, key on both dimensions
@triton.autotune(configs=configs, key=["M", "N", "K"])
```

## Benchmarking Template

```python
import torch
import triton
from sgl_kernel.triton.relu import relu


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="relu-performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.randn(size, dtype=torch.float32, device="cuda")
    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: relu(x), quantiles=quantiles
        )
    elif provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.relu(x), quantiles=quantiles
        )
    # Bandwidth: read input + write output
    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6  # noqa: E731
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)
```

## Integration with SGLang Runtime

For kernels used during model inference, register them in the appropriate attention or model layer:

```python
# In sglang/srt/layers/activation.py
from sgl_kernel.triton.relu import relu as triton_relu


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation fused with element-wise multiply."""
    # Use Triton kernel when available and on CUDA
    if x.is_cuda:
        return triton_silu_and_mul(x)
    return torch.nn.functional.silu(x[..., : x.shape[-1] // 2]) * x[..., x.shape[-1] // 2 :]
```

## Checklist

- [ ] Kernel has `@triton.jit` decorator
- [ ] Launcher validates input device (CUDA only)
- [ ] Launcher validates output shape if `out` param provided
- [ ] Boundary mask used in `tl.load` / `tl.store`
- [ ] Autotuning configs cover expected input size range
- [ ] Tests cover: correctness, out-param, CPU error, shape mismatch
- [ ] Kernel registered in `__init__.py`
- [ ] Benchmark script included for performance tracking
