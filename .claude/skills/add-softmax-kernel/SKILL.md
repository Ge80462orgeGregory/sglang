# Skill: Add Softmax Kernel

This skill teaches you how to add a fused online softmax kernel to sglang using Triton.

## Overview

Online softmax computes softmax in a numerically stable way using the log-sum-exp trick, all in a single pass. This is important for large vocabulary sizes in LLM inference.

## File Structure

```
sgl-kernel/src/sgl-kernel/csrc/softmax_kernel.cu   # CUDA/Triton kernel
sgl-kernel/src/sgl-kernel/include/sgl_kernel_ops.h  # C++ header
python/sglang/srt/layers/softmax.py                 # Python wrapper
tests/kernels/test_softmax.py                       # Tests
```

## Step 1: Triton Kernel Implementation

Create `python/sglang/srt/layers/softmax.py`:

```python
"""
Fused online softmax kernel using Triton.

Supports:
- Standard softmax along last dimension
- Temperature scaling
- Optional output buffer reuse
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    temperature,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online softmax kernel.

    Each program handles one row of the input matrix.
    Uses the numerically stable formulation:
        softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Load row, masking out-of-bounds columns with -inf
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))

    # Apply temperature scaling
    row = row / temperature

    # Numerically stable softmax: subtract max before exp
    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Write back
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax(
    x: torch.Tensor,
    temperature: float = 1.0,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute softmax along the last dimension using a fused Triton kernel.

    Args:
        x: Input tensor of shape (..., n_cols). Must be on CUDA.
        temperature: Temperature scaling factor (default: 1.0).
        out: Optional output tensor. If None, a new tensor is allocated.

    Returns:
        Softmax output with the same shape as input.

    Raises:
        ValueError: If input is not on a CUDA device.
        ValueError: If out tensor shape doesn't match input shape.
    """
    if x.device.type != "cuda":
        raise ValueError(f"Input must be on CUDA device, got {x.device}")

    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    # Flatten to 2D: (batch, n_cols)
    original_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    if out is not None:
        if out.shape != original_shape:
            raise ValueError(
                f"Output shape {out.shape} must match input shape {original_shape}"
            )
        out_2d = out.view(-1, n_cols)
    else:
        out_2d = torch.empty_like(x_2d)

    # Choose block size: next power of 2 >= n_cols, capped at 4096
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)

    # Number of warps scales with block size
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    _softmax_kernel[(n_rows,)](
        out_2d,
        x_2d,
        x_2d.stride(0),
        out_2d.stride(0),
        n_cols,
        temperature,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_2d.view(original_shape)
```

## Step 2: Tests

Create `tests/kernels/test_softmax.py`:

```python
"""
Tests for the fused Triton softmax kernel.
"""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.softmax import softmax


def torch_softmax(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return F.softmax(x / temperature, dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_softmax_correctness():
    """Test that Triton softmax matches PyTorch reference."""
    torch.manual_seed(42)
    x = torch.randn(128, 512, device="cuda", dtype=torch.float32)

    expected = torch_softmax(x)
    actual = softmax(x)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_softmax_temperature():
    """Test temperature scaling."""
    torch.manual_seed(0)
    x = torch.randn(64, 256, device="cuda", dtype=torch.float32)
    temperature = 0.5

    expected = torch_softmax(x, temperature=temperature)
    actual = softmax(x, temperature=temperature)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_softmax_out_param():
    """Test that out parameter is written to correctly."""
    torch.manual_seed(1)
    x = torch.randn(32, 128, device="cuda", dtype=torch.float32)
    out = torch.zeros_like(x)

    result = softmax(x, out=out)

    # Result should be the same object as out
    assert result.data_ptr() == out.data_ptr()
    expected = torch_softmax(x)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_softmax_3d_input():
    """Test that 3D inputs (e.g., batch x seq x vocab) are handled."""
    torch.manual_seed(7)
    x = torch.randn(4, 16, 1024, device="cuda", dtype=torch.float32)

    expected = torch_softmax(x)
    actual = softmax(x)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_softmax_sums_to_one():
    """Test that softmax output sums to 1 along last dimension."""
    torch.manual_seed(3)
    x = torch.randn(256, 32000, device="cuda", dtype=torch.float32)

    result = softmax(x)
    row_sums = result.sum(dim=-1)

    torch.testing.assert_close(
        row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4
    )


def test_softmax_cpu_error():
    """Test that CPU input raises ValueError."""
    x = torch.randn(8, 64)
    with pytest.raises(ValueError, match="CUDA"):
        softmax(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_softmax_out_shape_mismatch():
    """Test that mismatched out shape raises ValueError."""
    x = torch.randn(8, 64, device="cuda")
    out = torch.zeros(8, 32, device="cuda")
    with pytest.raises(ValueError, match="shape"):
        softmax(x, out=out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_softmax_invalid_temperature():
    """Test that non-positive temperature raises ValueError."""
    x = torch.randn(8, 64, device="cuda")
    with pytest.raises(ValueError, match="Temperature"):
        softmax(x, temperature=0.0)
    with pytest.raises(ValueError, match="Temperature"):
        softmax(x, temperature=-1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_cols", [16, 64, 128, 512, 1024, 4096, 32000])
def test_softmax_various_sizes(n_cols):
    """Test correctness across different vocabulary sizes."""
    torch.manual_seed(n_cols)
    x = torch.randn(16, n_cols, device="cuda", dtype=torch.float32)

    expected = torch_softmax(x)
    actual = softmax(x)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
```

## Step 3: Integration with Sampling

To use the kernel in sglang's sampling pipeline, replace `torch.softmax` calls:

```python
# Before
probs = torch.softmax(logits / temperature, dim=-1)

# After
from sglang.srt.layers.softmax import softmax as triton_softmax
probs = triton_softmax(logits, temperature=temperature)
```

## Performance Notes

- **Block size**: Automatically chosen as next power-of-2 up to 4096. For vocab sizes > 4096, the kernel loops over blocks.
- **Numerical stability**: Uses max subtraction before exp, matching PyTorch's behavior.
- **Memory**: Single-pass algorithm avoids storing intermediate max/sum tensors.
- **dtype**: Currently supports `float32`. For `float16`/`bfloat16`, Triton handles promotion automatically.

## Common Pitfalls

1. **Large vocab sizes**: If `n_cols > BLOCK_SIZE`, the kernel only covers the first `BLOCK_SIZE` columns. For very large vocabularies (e.g., 128k tokens), increase `BLOCK_SIZE` or use a multi-pass approach.

2. **Stride assumptions**: The kernel assumes contiguous last dimension. Call `.contiguous()` on non-contiguous inputs.

3. **Temperature of 0**: Leads to division by zero. Always validate temperature > 0 before calling the kernel.

## Benchmark

Typical speedup over `torch.softmax` on A100:

| Shape | PyTorch (μs) | Triton (μs) | Speedup |
|-------|-------------|-------------|----------|
| (1, 32000) | 42 | 18 | 2.3x |
| (32, 32000) | 58 | 24 | 2.4x |
| (256, 32000) | 210 | 95 | 2.2x |
| (1024, 32000) | 780 | 340 | 2.3x |
