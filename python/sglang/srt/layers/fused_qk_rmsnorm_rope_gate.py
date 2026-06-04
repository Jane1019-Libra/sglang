"""Fused Q/K GemmaRMSNorm + NeoX RoPE + optional gate deinterleave.

Single Triton kernel replacing the 4-op sequence in
Qwen3_5AttentionDecoderLayer.forward_prepare_native:
  1. Q/Gate deinterleave (view + chunk)
  2. GemmaRMSNorm on Q (per-head)
  3. GemmaRMSNorm on K (per-head)
  4. NeoX RoPE on Q and K

Grid: (num_tokens, num_q_heads + num_kv_heads).
  pid_h < num_q_heads: Q programs (deinterleave + norm + RoPE + store gate)
  pid_h >= num_q_heads: K programs (norm + RoPE)
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_qk_gemma_rmsnorm_rope_gate_kernel(
    # Input pointers
    q_gate_ptr,
    k_ptr,
    # Output pointers
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    # Norm weights (raw, kernel applies +1.0)
    q_weight_ptr,
    k_weight_ptr,
    # RoPE
    cos_sin_cache_ptr,
    positions_ptr,
    # Strides
    stride_qg_t,
    stride_k_t,
    stride_qo_t,
    stride_ko_t,
    stride_gate_t,
    stride_gate_h,
    stride_cos_t,
    # Scalars
    eps,
    num_tokens,
    # Constexpr
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= num_tokens:
        return

    half_offs = tl.arange(0, HALF_DIM)

    # Load cos/sin for this token's position
    pos = tl.load(positions_ptr + pid_t)
    cos_base = pos * stride_cos_t
    cos = tl.load(cos_sin_cache_ptr + cos_base + half_offs).to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + cos_base + HALF_DIM + half_offs).to(tl.float32)

    if pid_h < NUM_Q_HEADS:
        # === Q path: deinterleave + GemmaRMSNorm + NeoX RoPE ===
        head_id = pid_h

        if HAS_GATE:
            # q_gate layout: [T, num_q_heads * 2 * head_dim]
            # For head h: q at offset h*2*HEAD_DIM, gate at h*2*HEAD_DIM + HEAD_DIM
            q_base = pid_t * stride_qg_t + head_id * 2 * HEAD_DIM
        else:
            # q layout: [T, num_q_heads * head_dim]
            q_base = pid_t * stride_qg_t + head_id * HEAD_DIM

        # Load Q as two halves
        q1 = tl.load(q_gate_ptr + q_base + half_offs).to(tl.float32)
        q2 = tl.load(q_gate_ptr + q_base + HALF_DIM + half_offs).to(tl.float32)

        if HAS_GATE:
            # Load and store gate (passthrough, no processing)
            gate_base = q_base + HEAD_DIM
            g1 = tl.load(q_gate_ptr + gate_base + half_offs)
            g2 = tl.load(q_gate_ptr + gate_base + HALF_DIM + half_offs)
            gate_out_base = pid_t * stride_gate_t + head_id * stride_gate_h
            tl.store(gate_out_ptr + gate_out_base + half_offs, g1)
            tl.store(gate_out_ptr + gate_out_base + HALF_DIM + half_offs, g2)

        # GemmaRMSNorm: out = x * rsqrt(mean(x^2) + eps) * (weight + 1.0)
        w1 = tl.load(q_weight_ptr + half_offs).to(tl.float32) + 1.0
        w2 = tl.load(q_weight_ptr + HALF_DIM + half_offs).to(tl.float32) + 1.0
        q_var = (tl.sum(q1 * q1) + tl.sum(q2 * q2)) / HEAD_DIM
        q_rstd = tl.math.rsqrt(q_var + eps)
        q1_n = q1 * q_rstd * w1
        q2_n = q2 * q_rstd * w2

        # NeoX RoPE: o1 = x1*cos - x2*sin, o2 = x2*cos + x1*sin
        o1 = q1_n * cos - q2_n * sin
        o2 = q2_n * cos + q1_n * sin

        # Store Q output
        q_out_base = pid_t * stride_qo_t + head_id * HEAD_DIM
        tl.store(
            q_out_ptr + q_out_base + half_offs,
            o1.to(q_out_ptr.dtype.element_ty),
        )
        tl.store(
            q_out_ptr + q_out_base + HALF_DIM + half_offs,
            o2.to(q_out_ptr.dtype.element_ty),
        )
    else:
        # === K path: GemmaRMSNorm + NeoX RoPE ===
        kv_head_id = pid_h - NUM_Q_HEADS
        k_base = pid_t * stride_k_t + kv_head_id * HEAD_DIM

        # Load K as two halves
        k1 = tl.load(k_ptr + k_base + half_offs).to(tl.float32)
        k2 = tl.load(k_ptr + k_base + HALF_DIM + half_offs).to(tl.float32)

        # GemmaRMSNorm
        w1 = tl.load(k_weight_ptr + half_offs).to(tl.float32) + 1.0
        w2 = tl.load(k_weight_ptr + HALF_DIM + half_offs).to(tl.float32) + 1.0
        k_var = (tl.sum(k1 * k1) + tl.sum(k2 * k2)) / HEAD_DIM
        k_rstd = tl.math.rsqrt(k_var + eps)
        k1_n = k1 * k_rstd * w1
        k2_n = k2 * k_rstd * w2

        # NeoX RoPE
        o1 = k1_n * cos - k2_n * sin
        o2 = k2_n * cos + k1_n * sin

        # Store K output
        k_out_base = pid_t * stride_ko_t + kv_head_id * HEAD_DIM
        tl.store(
            k_out_ptr + k_out_base + half_offs,
            o1.to(k_out_ptr.dtype.element_ty),
        )
        tl.store(
            k_out_ptr + k_out_base + HALF_DIM + half_offs,
            o2.to(k_out_ptr.dtype.element_ty),
        )


def fused_qk_gemma_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    has_gate: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fused Q/K GemmaRMSNorm + NeoX RoPE + optional gate deinterleave.

    Args:
        q_gate: [T, num_q_heads * 2 * head_dim] when gated,
                [T, num_q_heads * head_dim] when not. May be non-contiguous.
        k: [T, num_kv_heads * head_dim]. May be non-contiguous.
        q_weight: [head_dim] raw GemmaRMSNorm weight (kernel applies +1.0).
        k_weight: [head_dim] raw GemmaRMSNorm weight.
        cos_sin_cache: [max_pos, head_dim] = [cos(half) || sin(half)].
        positions: [T] int token positions.
        eps: RMSNorm epsilon.
        num_q_heads: Number of Q heads (after TP split).
        num_kv_heads: Number of KV heads (after TP split).
        head_dim: Head dimension.
        has_gate: Whether q_gate contains interleaved Q+gate.

    Returns:
        (q, k, gate):
          q: [T, num_q_heads * head_dim]
          k: [T, num_kv_heads * head_dim]
          gate: [T, num_q_heads, head_dim] if has_gate else None
    """
    T = q_gate.shape[0]
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    HALF_DIM = head_dim // 2

    q_out = torch.empty(T, q_size, dtype=q_gate.dtype, device=q_gate.device)
    k_out = torch.empty(T, kv_size, dtype=k.dtype, device=k.device)

    if has_gate:
        gate_out = torch.empty(
            T, num_q_heads, head_dim, dtype=q_gate.dtype, device=q_gate.device
        )
    else:
        gate_out = q_out  # dummy, won't be written

    num_warps = 4 if head_dim <= 128 else 8

    grid = (T, num_q_heads + num_kv_heads)
    _fused_qk_gemma_rmsnorm_rope_gate_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        gate_out.stride(1) if has_gate else 0,
        cos_sin_cache.stride(0),
        eps,
        T,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        HALF_DIM=HALF_DIM,
        HAS_GATE=has_gate,
        num_warps=num_warps,
    )

    return q_out, k_out, gate_out if has_gate else None
