# Skill: Add Layer Norm Kernel

This skill demonstrates how to add a fused layer normalization kernel to sglang using Triton.

## Overview

Layer normalization is a critical operation in transformer models. This skill shows how to implement a fused forward pass kernel that computes mean, variance, and normalization in a single pass for better performance.

## File Structure

```
sgl-kernel/src/sgl-kernel/csrc/
    layer_norm_kernel.cu          # CUDA/Triton kernel implementation
sgl-kernel/src/sgl-kernel/
    ops.py                        # Python bindings
tests/
    test_layer_norm_kernel.py     # Unit tests
```

## Step 1: Implement the Triton Kernel

Create `sgl-kernel/src/sgl-kernel/csrc/layer_norm_kernel.py`:

```python
import triton
import triton.language as tl
import torch


@triton.jit
def _layer_norm_fwd_kernel(
    X,          # pointer to input tensor [N, D]
    W,          # pointer to weight (gamma) [D]
    B,          # pointer to bias (beta) [D]
    Y,          # pointer to output tensor [N, D]
    Mean,       # pointer to mean output [N]
    Rstd,       # pointer to reciprocal std output [N]
    stride_x,   # stride for X rows
    N,          # number of rows
    D,          # number of columns (hidden dim)
    eps,        # epsilon for numerical stability
    BLOCK_D: tl.constexpr,  # block size along D dimension
):
    # Each program handles one row
    row = tl.program_id(0)
    X_ptr = X + row * stride_x
    Y_ptr = Y + row * stride_x

    # Load row into SRAM
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(x, axis=0) / D
    tl.store(Mean + row, mean)

    # Compute variance
    xmean = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xmean * xmean, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # Normalize and apply affine transform
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xmean * rstd * w + b
    tl.store(Y_ptr + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_kernel(
    DX,         # pointer to input grad [N, D]
    DY,         # pointer to output grad [N, D]
    X,          # pointer to input [N, D]
    W,          # pointer to weight [D]
    Mean,       # pointer to saved mean [N]
    Rstd,       # pointer to saved rstd [N]
    stride,     # stride for rows
    N,
    D,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    xhat = (x - mean) * rstd
    wdy = w * dy
    c1 = tl.sum(xhat * wdy, axis=0) / D
    c2 = tl.sum(wdy, axis=0) / D
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(DX + row * stride + cols, dx, mask=mask)
```

## Step 2: Python Wrapper

Create `sgl-kernel/src/sgl-kernel/layer_norm.py`:

```python
import torch
from torch import Tensor
from typing import Optional

# Import the compiled kernel or fall back to triton
try:
    from sgl_kernel import layer_norm_forward as _layer_norm_forward
except ImportError:
    _layer_norm_forward = None


class LayerNormFunction(torch.autograd.Function):
    """Autograd-compatible fused layer norm."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
        eps: float = 1e-5,
    ) -> Tensor:
        assert x.is_cuda, "Input must be on CUDA device"
        assert x.ndim >= 2, "Input must be at least 2D"

        # Flatten to 2D: [N, D]
        shape = x.shape
        x_2d = x.view(-1, shape[-1])
        N, D = x_2d.shape

        y = torch.empty_like(x_2d)
        mean = torch.empty(N, dtype=torch.float32, device=x.device)
        rstd = torch.empty(N, dtype=torch.float32, device=x.device)

        # Choose block size (next power of 2 >= D, max 4096)
        BLOCK_D = max(triton.next_power_of_2(D), 64)
        BLOCK_D = min(BLOCK_D, 4096)

        from .csrc.layer_norm_kernel import _layer_norm_fwd_kernel
        _layer_norm_fwd_kernel[(N,)](
            x_2d, weight, bias, y, mean, rstd,
            x_2d.stride(0), N, D, eps,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(x_2d, weight, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.D = D
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy: Tensor):
        x, weight, mean, rstd = ctx.saved_tensors
        N, D = x.shape
        BLOCK_D = ctx.BLOCK_D

        dy_2d = dy.view(N, D).contiguous()
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        db = torch.empty_like(weight) if weight is not None else None

        from .csrc.layer_norm_kernel import _layer_norm_bwd_dx_kernel
        _layer_norm_bwd_dx_kernel[(N,)](
            dx, dy_2d, x, weight, mean, rstd,
            x.stride(0), N, D,
            BLOCK_D=BLOCK_D,
        )

        # Compute weight/bias grads across batch dimension
        dw = (dy_2d * (x - mean[:, None]) * rstd[:, None]).sum(0)
        db = dy_2d.sum(0)

        return dx.view_as(x), dw, db, None


def layer_norm(
    x: Tensor,
    normalized_shape: tuple,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Fused layer normalization using Triton kernel.

    Args:
        x: Input tensor of shape [..., D]
        normalized_shape: Shape to normalize over (typically (D,))
        weight: Optional learnable scale parameter (gamma)
        bias: Optional learnable bias parameter (beta)
        eps: Small value for numerical stability

    Returns:
        Normalized tensor of same shape as input
    """
    D = normalized_shape[-1]
    if weight is None:
        weight = torch.ones(D, device=x.device, dtype=x.dtype)
    if bias is None:
        bias = torch.zeros(D, device=x.device, dtype=x.dtype)

    return LayerNormFunction.apply(x, weight, bias, eps)
```

## Step 3: Tests

Create `tests/test_layer_norm_kernel.py`:

```python
import pytest
import torch
import torch.nn.functional as F

from sgl_kernel.layer_norm import layer_norm


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def torch_layer_norm(x, weight, bias, eps=1e-5):
    """Reference implementation using PyTorch built-in."""
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def test_layer_norm_correctness(device):
    """Test that fused kernel matches PyTorch reference."""
    torch.manual_seed(42)
    N, D = 128, 512
    x = torch.randn(N, D, device=device, dtype=torch.float16)
    weight = torch.randn(D, device=device, dtype=torch.float16)
    bias = torch.randn(D, device=device, dtype=torch.float16)

    y_ref = torch_layer_norm(x, weight, bias)
    y_tri = layer_norm(x, (D,), weight, bias)

    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=1e-2)


def test_layer_norm_3d_input(device):
    """Test with 3D input (batch, seq, hidden)."""
    torch.manual_seed(0)
    B, T, D = 4, 32, 256
    x = torch.randn(B, T, D, device=device, dtype=torch.float32)
    weight = torch.ones(D, device=device)
    bias = torch.zeros(D, device=device)

    y_ref = torch_layer_norm(x, weight, bias)
    y_tri = layer_norm(x, (D,), weight, bias)

    torch.testing.assert_close(y_tri, y_ref, atol=1e-4, rtol=1e-4)


def test_layer_norm_no_affine(device):
    """Test without weight and bias (pure normalization)."""
    torch.manual_seed(1)
    N, D = 64, 128
    x = torch.randn(N, D, device=device)

    y_tri = layer_norm(x, (D,))

    # Check that output has approximately zero mean and unit variance per row
    assert y_tri.mean(dim=-1).abs().max() < 1e-3
    assert (y_tri.std(dim=-1) - 1.0).abs().max() < 1e-2


def test_layer_norm_cpu_error():
    """Test that CPU input raises an error."""
    x = torch.randn(16, 64)
    with pytest.raises(AssertionError, match="CUDA"):
        layer_norm(x, (64,))


def test_layer_norm_large_hidden(device):
    """Test with large hidden dimension."""
    torch.manual_seed(7)
    N, D = 32, 4096
    x = torch.randn(N, D, device=device, dtype=torch.float16)
    weight = torch.randn(D, device=device, dtype=torch.float16)
    bias = torch.zeros(D, device=device, dtype=torch.float16)

    y_ref = torch_layer_norm(x, weight, bias)
    y_tri = layer_norm(x, (D,), weight, bias)

    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_layer_norm_dtypes(device, dtype):
    """Test correctness across different dtypes."""
    torch.manual_seed(3)
    N, D = 64, 256
    x = torch.randn(N, D, device=device, dtype=dtype)
    weight = torch.ones(D, device=device, dtype=dtype)
    bias = torch.zeros(D, device=device, dtype=dtype)

    y_ref = torch_layer_norm(x, weight, bias)
    y_tri = layer_norm(x, (D,), weight, bias)

    atol = 1e-2 if dtype != torch.float32 else 1e-4
    torch.testing.assert_close(y_tri, y_ref, atol=atol, rtol=atol)
```

## Step 4: Register Op (Optional)

If you need to register this as a custom PyTorch op for use in `torch.compile`:

```python
# In sgl-kernel/src/sgl-kernel/ops.py
import torch
from torch import Tensor

@torch.library.custom_op("sgl::layer_norm", mutates_args=())
def layer_norm_op(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    from .layer_norm import layer_norm
    return layer_norm(x, (x.shape[-1],), weight, bias, eps)

@layer_norm_op.register_fake
def _(x, weight, bias, eps):
    return torch.empty_like(x)
```

## Performance Tips

1. **Block size tuning**: Use `triton.autotune` to find optimal `BLOCK_D` for your hardware.
2. **Persistent kernels**: For small `D`, consider persistent kernel patterns to amortize launch overhead.
3. **Mixed precision**: The kernel internally uses fp32 for accumulation to avoid precision loss.
4. **Fused residual add**: Consider fusing with the residual connection (`x = layer_norm(x + residual)`) for extra performance.

## Common Pitfalls

- **Non-contiguous inputs**: Always call `.contiguous()` before passing to the kernel if strides are non-standard.
- **D not power of 2**: The `triton.next_power_of_2` call handles this, but verify your block size covers all elements.
- **Epsilon too small**: For fp16, use `eps >= 1e-5` to avoid NaN in reciprocal std computation.

## Benchmarking

```python
import torch
import triton
from sgl_kernel.layer_norm import layer_norm

def benchmark_layer_norm(N=1024, D=4096, dtype=torch.float16, num_warmup=25, num_rep=100):
    x = torch.randn(N, D, device='cuda', dtype=dtype)
    w = torch.ones(D, device='cuda', dtype=dtype)
    b = torch.zeros(D, device='cuda', dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        layer_norm(x, (D,), w, b)

    # Benchmark
    ms = triton.testing.do_bench(lambda: layer_norm(x, (D,), w, b), rep=num_rep)
    gbps = (3 * x.numel() * x.element_size()) / ms * 1e-6  # GB/s
    print(f"N={N}, D={D}, dtype={dtype}: {ms:.3f} ms, {gbps:.1f} GB/s")

benchmark_layer_norm()
```
