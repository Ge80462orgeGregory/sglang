# Skill: Add RoPE (Rotary Position Embedding) Kernel

This skill covers adding a custom RoPE kernel to sglang, including both a Triton implementation and integration with the existing attention infrastructure.

## Overview

Rotary Position Embedding (RoPE) is a position encoding technique used in modern LLMs (LLaMA, Mistral, etc.). It rotates query and key vectors in the attention mechanism based on their position in the sequence.

## File Structure

```
sgl-kernel/src/sgl-kernel/csrc/rope_kernel.cu     # CUDA kernel (optional)
python/sglang/srt/layers/rotary_embedding.py      # Main implementation
tests/kernels/test_rope.py                         # Tests
```

## Implementation

### 1. Triton Kernel (`python/sglang/srt/layers/rotary_embedding.py`)

```python
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def _rope_fwd_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    seq_len,
    num_heads_q,
    num_heads_k,
    head_dim,
    half_dim,
    stride_q_seq,
    stride_q_head,
    stride_k_seq,
    stride_k_head,
    stride_cos_seq,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for applying RoPE to query and key tensors.

    Args:
        q_ptr: Query tensor pointer [seq_len, num_heads_q, head_dim]
        k_ptr: Key tensor pointer [seq_len, num_heads_k, head_dim]
        cos_ptr: Cosine cache pointer [max_seq_len, half_dim]
        sin_ptr: Sine cache pointer [max_seq_len, half_dim]
        q_out_ptr: Output query pointer
        k_out_ptr: Output key pointer
        seq_len: Sequence length
        num_heads_q: Number of query heads
        num_heads_k: Number of key heads
        head_dim: Head dimension
        half_dim: head_dim // 2
        stride_*: Tensor strides
        BLOCK_SIZE: Triton block size
    """
    pid = tl.program_id(0)
    seq_idx = pid // (num_heads_q + num_heads_k)
    head_idx = pid % (num_heads_q + num_heads_k)
    is_query = head_idx < num_heads_q

    if seq_idx >= seq_len:
        return

    # Load cos/sin for this position
    cos_offset = seq_idx * stride_cos_seq
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_dim

    cos = tl.load(cos_ptr + cos_offset + offsets, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + cos_offset + offsets, mask=mask, other=0.0)

    if is_query:
        base_ptr = q_ptr + seq_idx * stride_q_seq + head_idx * stride_q_head
        out_ptr = q_out_ptr + seq_idx * stride_q_seq + head_idx * stride_q_head
    else:
        k_head_idx = head_idx - num_heads_q
        base_ptr = k_ptr + seq_idx * stride_k_seq + k_head_idx * stride_k_head
        out_ptr = k_out_ptr + seq_idx * stride_k_seq + k_head_idx * stride_k_head

    # Load first and second half
    x1 = tl.load(base_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(base_ptr + half_dim + offsets, mask=mask, other=0.0)

    # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    tl.store(out_ptr + offsets, out1, mask=mask)
    tl.store(out_ptr + half_dim + offsets, out2, mask=mask)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding (RoPE) to query and key tensors.

    Args:
        q: Query tensor of shape [seq_len, num_heads_q, head_dim]
        k: Key tensor of shape [seq_len, num_heads_k, head_dim]
        cos: Cosine cache of shape [max_seq_len, head_dim // 2]
        sin: Sine cache of shape [max_seq_len, head_dim // 2]
        q_out: Optional pre-allocated output for queries
        k_out: Optional pre-allocated output for keys

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    assert q.device.type == "cuda", "RoPE kernel requires CUDA tensors"
    assert q.dim() == 3 and k.dim() == 3, "Expected [seq_len, num_heads, head_dim]"
    assert q.shape[-1] == k.shape[-1], "Query and key must have same head_dim"
    assert q.shape[0] == k.shape[0], "Query and key must have same seq_len"

    seq_len, num_heads_q, head_dim = q.shape
    _, num_heads_k, _ = k.shape
    half_dim = head_dim // 2

    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    assert cos.shape[-1] == half_dim, f"cos last dim {cos.shape[-1]} != {half_dim}"

    if q_out is None:
        q_out = torch.empty_like(q)
    if k_out is None:
        k_out = torch.empty_like(k)

    total_programs = seq_len * (num_heads_q + num_heads_k)
    BLOCK_SIZE = triton.next_power_of_2(half_dim)

    _rope_fwd_kernel[(total_programs,)](
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        seq_len,
        num_heads_q,
        num_heads_k,
        head_dim,
        half_dim,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        cos.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return q_out, k_out


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build cosine and sine caches for RoPE.

    Args:
        seq_len: Maximum sequence length
        head_dim: Dimension of each attention head
        base: Base for computing frequencies (default: 10000)
        device: Target device
        dtype: Output dtype

    Returns:
        Tuple of (cos_cache, sin_cache), each of shape [seq_len, head_dim // 2]
    """
    half_dim = head_dim // 2
    # Compute inverse frequencies: 1 / (base^(2i/d)) for i in [0, half_dim)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    # Outer product: [seq_len] x [half_dim] -> [seq_len, half_dim]
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)

    cos_cache = freqs.cos().to(dtype)
    sin_cache = freqs.sin().to(dtype)

    return cos_cache, sin_cache
```

### 2. Reference PyTorch Implementation (for testing)

```python
def torch_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference PyTorch implementation of RoPE for correctness testing.
    """
    def rotate_half(x):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    # cos/sin: [seq_len, half_dim] -> [seq_len, 1, head_dim]
    seq_len = q.shape[0]
    cos_full = torch.cat([cos[:seq_len], cos[:seq_len]], dim=-1).unsqueeze(1)
    sin_full = torch.cat([sin[:seq_len], sin[:seq_len]], dim=-1).unsqueeze(1)

    q_rot = q * cos_full + rotate_half(q) * sin_full
    k_rot = k * cos_full + rotate_half(k) * sin_full

    return q_rot, k_rot
```

### 3. Tests (`tests/kernels/test_rope.py`)

```python
import pytest
import torch
from python.sglang.srt.layers.rotary_embedding import (
    apply_rope,
    build_rope_cache,
    torch_apply_rope,
)


@pytest.fixture
def rope_inputs():
    seq_len = 64
    num_heads_q = 32
    num_heads_k = 8  # GQA
    head_dim = 128
    device = torch.device("cuda")
    dtype = torch.float16

    q = torch.randn(seq_len, num_heads_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(seq_len, num_heads_k, head_dim, device=device, dtype=dtype)
    cos, sin = build_rope_cache(seq_len, head_dim, device=device, dtype=dtype)

    return q, k, cos, sin


def test_rope_correctness(rope_inputs):
    """Test that Triton RoPE matches PyTorch reference."""
    q, k, cos, sin = rope_inputs

    q_triton, k_triton = apply_rope(q, k, cos, sin)
    q_ref, k_ref = torch_apply_rope(q.float(), k.float(), cos.float(), sin.float())

    torch.testing.assert_close(q_triton.float(), q_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_triton.float(), k_ref, atol=1e-2, rtol=1e-2)


def test_rope_out_param(rope_inputs):
    """Test that pre-allocated output buffers work correctly."""
    q, k, cos, sin = rope_inputs
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    q_result, k_result = apply_rope(q, k, cos, sin, q_out=q_out, k_out=k_out)

    assert q_result.data_ptr() == q_out.data_ptr()
    assert k_result.data_ptr() == k_out.data_ptr()


def test_rope_cpu_error(rope_inputs):
    """Test that CPU tensors raise an appropriate error."""
    q, k, cos, sin = rope_inputs
    q_cpu = q.cpu()
    k_cpu = k.cpu()
    cos_cpu = cos.cpu()
    sin_cpu = sin.cpu()

    with pytest.raises(AssertionError, match="CUDA"):
        apply_rope(q_cpu, k_cpu, cos_cpu, sin_cpu)


def test_rope_shapes(rope_inputs):
    """Test output shapes match input shapes."""
    q, k, cos, sin = rope_inputs
    q_out, k_out = apply_rope(q, k, cos, sin)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


@pytest.mark.parametrize("seq_len", [1, 16, 128, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_rope_various_shapes(seq_len, head_dim):
    """Test RoPE across various sequence lengths and head dimensions."""
    device = torch.device("cuda")
    num_heads = 8

    q = torch.randn(seq_len, num_heads, head_dim, device=device)
    k = torch.randn(seq_len, num_heads, head_dim, device=device)
    cos, sin = build_rope_cache(seq_len, head_dim, device=device)

    q_out, k_out = apply_rope(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_rope_cache_shape():
    """Test that build_rope_cache returns correct shapes."""
    seq_len, head_dim = 2048, 128
    cos, sin = build_rope_cache(seq_len, head_dim)

    assert cos.shape == (seq_len, head_dim // 2)
    assert sin.shape == (seq_len, head_dim // 2)
```

## Integration with Attention

To integrate RoPE with the existing attention module:

```python
# In python/sglang/srt/layers/attention.py
from sglang.srt.layers.rotary_embedding import apply_rope, build_rope_cache

class Attention(nn.Module):
    def __init__(self, ...):
        ...
        self.cos_cache, self.sin_cache = build_rope_cache(
            max_seq_len, head_dim, base=rope_base, device=device
        )

    def forward(self, q, k, v, positions):
        # Apply RoPE to queries and keys
        cos = self.cos_cache[positions]
        sin = self.sin_cache[positions]
        q, k = apply_rope(q, k, cos, sin)
        # ... rest of attention
```

## Performance Notes

- The Triton kernel fuses the rotation operation, avoiding extra memory reads/writes
- For decode (seq_len=1), consider a specialized single-token kernel
- The kernel supports GQA (num_heads_q != num_heads_k)
- Use `torch.compile` as a fallback for CPU or unsupported dtypes

## Common Issues

1. **Head dim must be even**: RoPE splits the head dimension in half; ensure `head_dim % 2 == 0`
2. **Dtype mismatch**: cos/sin cache dtype should match q/k dtype for best performance
3. **Position offset**: For KV cache reuse, pass absolute positions, not relative ones
4. **Extended context (YaRN/LongRoPE)**: Modify `build_rope_cache` to apply frequency scaling
