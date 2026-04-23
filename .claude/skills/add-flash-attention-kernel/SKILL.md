# Skill: Add Flash Attention Kernel

This skill guides you through adding a custom Flash Attention kernel to sglang, either via Triton JIT, sgl-kernel (CUDA), or leveraging existing optimized backends.

## Overview

Flash Attention is a memory-efficient attention mechanism that tiles the attention computation to avoid materializing the full N×N attention matrix. When adding a new attention kernel to sglang, you typically need to:

1. Implement the kernel (Triton or CUDA)
2. Register it with the attention backend dispatch
3. Write correctness + performance tests
4. Optionally expose it via a benchmark script

---

## File Layout

```
sglang/
  sgl_kernel/
    csrc/
      flash_attn/          # CUDA kernels (if using sgl-kernel path)
    ops/
      attention.py         # Python wrappers
  sglang/
    srt/
      layers/
        attention/
          triton_ops/
            flash_attn.py  # Triton kernel (if using Triton path)
          backend.py       # Dispatch logic
tests/
  kernels/
    test_flash_attn.py
```

---

## Step 1 — Choose Your Implementation Path

### Path A: Triton JIT Kernel

Use when you want pure-Python portability and fast iteration. See `.claude/skills/add-triton-kernel/SKILL.md` for the full Triton workflow.

```python
# sglang/srt/layers/attention/triton_ops/flash_attn.py
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    sm_scale,
):
    """Flash Attention forward kernel (simplified single-pass)."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Pointers to Q/K/V blocks
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    # Running statistics for online softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(Q_block_ptr)

    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        # QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # Causal mask
        if IS_CAUSAL:
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            qk += tl.where(offs_m[:, None] >= offs_n[None, :], 0, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Normalize
    acc = acc / l_i[:, None]

    # Write output
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(tl.float16))


def flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    is_causal: bool = True,
    BLOCK_M: int = 128,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    """Flash Attention forward pass.

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim). float16.
        k: Key tensor of shape (batch, heads, seq_len, head_dim). float16.
        v: Value tensor of shape (batch, heads, seq_len, head_dim). float16.
        sm_scale: Softmax scale factor (typically 1/sqrt(head_dim)).
        is_causal: Whether to apply causal masking.
        BLOCK_M: Tile size along query sequence dimension.
        BLOCK_N: Tile size along key/value sequence dimension.

    Returns:
        Output tensor of shape (batch, heads, seq_len, head_dim). float16.
    """
    assert q.dtype == torch.float16, "Flash Attention Triton kernel requires float16"
    assert q.is_cuda, "Inputs must be on CUDA"
    Z, H, N_CTX, D_HEAD = q.shape
    out = torch.empty_like(q)

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D_HEAD,
        IS_CAUSAL=is_causal,
        sm_scale=sm_scale,
        num_warps=4,
        num_stages=2,
    )
    return out
```

### Path B: sgl-kernel (CUDA Extension)

Use when you need maximum performance and can write CUDA C++. See `.claude/skills/add-sgl-kernel/SKILL.md`.

Key files to create:
- `sgl_kernel/csrc/flash_attn/flash_attn_fwd.cu` — CUDA kernel
- `sgl_kernel/csrc/flash_attn/flash_attn_fwd.cuh` — Header
- `sgl_kernel/ops/attention.py` — Python binding
- Entry in `sgl_kernel/csrc/ops.h` and `setup.py`

---

## Step 2 — Reference (PyTorch) Implementation

Always write a reference implementation for correctness testing:

```python
def torch_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    is_causal: bool = True,
) -> torch.Tensor:
    """Reference Flash Attention using PyTorch scaled_dot_product_attention."""
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        scale=sm_scale,
        is_causal=is_causal,
    )
```

---

## Step 3 — Tests

```python
# tests/kernels/test_flash_attn.py
import math
import pytest
import torch

from sglang.srt.layers.attention.triton_ops.flash_attn import flash_attn_fwd


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("seq_len", [128, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_attn_correctness(batch, heads, seq_len, head_dim, is_causal):
    """Compare Triton Flash Attention output to PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    sm_scale = 1.0 / math.sqrt(head_dim)
    shape = (batch, heads, seq_len, head_dim)

    q = torch.randn(shape, dtype=torch.float16, device="cuda")
    k = torch.randn(shape, dtype=torch.float16, device="cuda")
    v = torch.randn(shape, dtype=torch.float16, device="cuda")

    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=sm_scale, is_causal=is_causal
    )
    out = flash_attn_fwd(q, k, v, sm_scale=sm_scale, is_causal=is_causal)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_flash_attn_cpu_error():
    """Kernel must raise on CPU input."""
    q = torch.randn(1, 8, 64, 64, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with pytest.raises(AssertionError, match="CUDA"):
        flash_attn_fwd(q, k, v, sm_scale=0.125)


def test_flash_attn_dtype_error():
    """Kernel must raise on non-float16 input."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    q = torch.randn(1, 8, 64, 64, dtype=torch.float32, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with pytest.raises(AssertionError, match="float16"):
        flash_attn_fwd(q, k, v, sm_scale=0.125)


@pytest.mark.parametrize("seq_len", [128, 256, 512, 1024, 2048])
def test_flash_attn_benchmark(seq_len, benchmark):
    """Benchmark Triton Flash Attention vs PyTorch SDPA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch, heads, head_dim = 2, 16, 64
    sm_scale = 1.0 / math.sqrt(head_dim)
    shape = (batch, heads, seq_len, head_dim)

    q = torch.randn(shape, dtype=torch.float16, device="cuda")
    k = torch.randn(shape, dtype=torch.float16, device="cuda")
    v = torch.randn(shape, dtype=torch.float16, device="cuda")

    benchmark(flash_attn_fwd, q, k, v, sm_scale, True)
```

---

## Step 4 — Backend Dispatch Integration

If you want sglang's attention layer to automatically select your kernel:

```python
# sglang/srt/layers/attention/backend.py  (excerpt)
from sglang.srt.layers.attention.triton_ops.flash_attn import flash_attn_fwd


class AttentionBackend:
    """Dispatch attention computation to the best available kernel."""

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sm_scale: float,
        is_causal: bool = True,
        backend: str = "auto",
    ) -> torch.Tensor:
        if backend == "triton" or (backend == "auto" and q.is_cuda and q.dtype == torch.float16):
            return flash_attn_fwd(q, k, v, sm_scale, is_causal)
        # Fallback to PyTorch SDPA
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=sm_scale, is_causal=is_causal
        )
```

---

## Common Pitfalls

| Symptom | Likely Cause | Fix |
|---|---|---|  
| NaN outputs | Missing softmax normalization | Ensure `acc / l_i` at the end of the kernel |
| Wrong causal mask | Off-by-one in `hi` bound | Use `(start_m + 1) * BLOCK_M` not `start_m * BLOCK_M` |
| Slow perf | `num_stages=1` | Try `num_stages=2` or `3` for A100/H100 |
| OOM on long seqs | Large `BLOCK_M` | Reduce `BLOCK_M` to 64 or 32 |
| Mismatched strides | Contiguous assumption | Call `.contiguous()` before passing to kernel |
| Incorrect results on non-power-of-2 seq | Missing padding | Add `tl.where` mask on boundary tiles |

---

## Performance Tuning

```python
# Autotune config example
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=2, num_warps=8),
    ],
    key=["N_CTX", "BLOCK_DMODEL"],
)
@triton.jit
def _flash_attn_fwd_kernel_autotuned(...):
    ...
```

Typical achieved performance on A100 (fp16, causal, head_dim=128):

| Seq Len | TFLOPS |
|---|---|
| 512  | ~180 |
| 1024 | ~210 |
| 2048 | ~230 |
| 4096 | ~240 |

---

## Checklist

- [ ] Kernel compiles without errors (`triton.compile` or `nvcc`)
- [ ] Correctness test passes vs PyTorch SDPA (`atol=1e-2`)
- [ ] CPU / dtype error tests pass
- [ ] Causal and non-causal variants tested
- [ ] Benchmark shows improvement over baseline
- [ ] Dispatch logic updated in `backend.py`
- [ ] CI workflow passes (see `.claude/skills/ci-workflow-guide/SKILL.md`)
