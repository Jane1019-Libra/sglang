"""Fused Q/K GemmaRMSNorm + NeoX RoPE + optional gate deinterleave.

Single Triton kernel replacing the 4-op sequence in
Qwen3_5AttentionDecoderLayer.forward_prepare_native:
  1. Q/Gate deinterleave (view + chunk)
  2. GemmaRMSNorm on Q (per-head)
  3. GemmaRMSNorm on K (per-head)
  4. NeoX RoPE on Q and K (partial or full rotation)

Grid: (num_tokens, num_q_heads + num_kv_heads).
  pid_h < num_q_heads: Q programs (deinterleave + norm + RoPE + store gate)
  pid_h >= num_q_heads: K programs (norm + RoPE)

Supports partial_rotary_factor < 1.0: only the first ROTARY_DIM elements of
each head get RoPE; the remaining elements are normalized but not rotated
(e.g. Qwen3.5 MoE: head_dim=256, partial_rotary_factor=0.25, rotary_dim=64).
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
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    HAS_NOPE: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= num_tokens:
        return

    rot_offs = tl.arange(0, HALF_ROTARY)

    # Load cos/sin for this token's position
    pos = tl.load(positions_ptr + pid_t)
    cos_base = pos * stride_cos_t
    cos = tl.load(cos_sin_cache_ptr + cos_base + rot_offs).to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + cos_base + HALF_ROTARY + rot_offs).to(tl.float32)

    if pid_h < NUM_Q_HEADS:
        # === Q path ===
        head_id = pid_h
        if HAS_GATE:
            q_base = pid_t * stride_qg_t + head_id * 2 * HEAD_DIM
        else:
            q_base = pid_t * stride_qg_t + head_id * HEAD_DIM

        # Load rotary portion as two halves
        q_r1 = tl.load(q_gate_ptr + q_base + rot_offs).to(tl.float32)
        q_r2 = tl.load(q_gate_ptr + q_base + HALF_ROTARY + rot_offs).to(tl.float32)
        var_acc = tl.sum(q_r1 * q_r1) + tl.sum(q_r2 * q_r2)

        # Load nope portion
        if HAS_NOPE:
            nope_offs = tl.arange(0, NOPE_BLOCK)
            nope_mask = nope_offs < NOPE_DIM
            q_nope = tl.load(
                q_gate_ptr + q_base + ROTARY_DIM + nope_offs,
                mask=nope_mask,
                other=0.0,
            ).to(tl.float32)
            var_acc += tl.sum(q_nope * q_nope)

        # Gate passthrough
        if HAS_GATE:
            gate_base = q_base + HEAD_DIM
            g_r1 = tl.load(q_gate_ptr + gate_base + rot_offs)
            g_r2 = tl.load(q_gate_ptr + gate_base + HALF_ROTARY + rot_offs)
            gate_out_base = pid_t * stride_gate_t + head_id * stride_gate_h
            tl.store(gate_out_ptr + gate_out_base + rot_offs, g_r1)
            tl.store(gate_out_ptr + gate_out_base + HALF_ROTARY + rot_offs, g_r2)
            if HAS_NOPE:
                g_nope = tl.load(
                    q_gate_ptr + gate_base + ROTARY_DIM + nope_offs,
                    mask=nope_mask,
                )
                tl.store(
                    gate_out_ptr + gate_out_base + ROTARY_DIM + nope_offs,
                    g_nope,
                    mask=nope_mask,
                )

        # GemmaRMSNorm
        q_var = var_acc / HEAD_DIM
        q_rstd = tl.math.rsqrt(q_var + eps)
        w_r1 = tl.load(q_weight_ptr + rot_offs).to(tl.float32) + 1.0
        w_r2 = tl.load(q_weight_ptr + HALF_ROTARY + rot_offs).to(tl.float32) + 1.0
        q_r1_n = q_r1 * q_rstd * w_r1
        q_r2_n = q_r2 * q_rstd * w_r2

        # NeoX RoPE on rotary portion
        o1 = q_r1_n * cos - q_r2_n * sin
        o2 = q_r2_n * cos + q_r1_n * sin

        # Store Q
        q_out_base = pid_t * stride_qo_t + head_id * HEAD_DIM
        tl.store(q_out_ptr + q_out_base + rot_offs, o1.to(q_out_ptr.dtype.element_ty))
        tl.store(
            q_out_ptr + q_out_base + HALF_ROTARY + rot_offs,
            o2.to(q_out_ptr.dtype.element_ty),
        )

        if HAS_NOPE:
            w_nope = (
                tl.load(
                    q_weight_ptr + ROTARY_DIM + nope_offs,
                    mask=nope_mask,
                    other=0.0,
                ).to(tl.float32)
                + 1.0
            )
            q_nope_n = q_nope * q_rstd * w_nope
            tl.store(
                q_out_ptr + q_out_base + ROTARY_DIM + nope_offs,
                q_nope_n.to(q_out_ptr.dtype.element_ty),
                mask=nope_mask,
            )
    else:
        # === K path ===
        kv_head_id = pid_h - NUM_Q_HEADS
        k_base = pid_t * stride_k_t + kv_head_id * HEAD_DIM

        k_r1 = tl.load(k_ptr + k_base + rot_offs).to(tl.float32)
        k_r2 = tl.load(k_ptr + k_base + HALF_ROTARY + rot_offs).to(tl.float32)
        var_acc = tl.sum(k_r1 * k_r1) + tl.sum(k_r2 * k_r2)

        if HAS_NOPE:
            nope_offs = tl.arange(0, NOPE_BLOCK)
            nope_mask = nope_offs < NOPE_DIM
            k_nope = tl.load(
                k_ptr + k_base + ROTARY_DIM + nope_offs,
                mask=nope_mask,
                other=0.0,
            ).to(tl.float32)
            var_acc += tl.sum(k_nope * k_nope)

        # GemmaRMSNorm
        k_var = var_acc / HEAD_DIM
        k_rstd = tl.math.rsqrt(k_var + eps)
        w_r1 = tl.load(k_weight_ptr + rot_offs).to(tl.float32) + 1.0
        w_r2 = tl.load(k_weight_ptr + HALF_ROTARY + rot_offs).to(tl.float32) + 1.0
        k_r1_n = k_r1 * k_rstd * w_r1
        k_r2_n = k_r2 * k_rstd * w_r2

        # NeoX RoPE
        o1 = k_r1_n * cos - k_r2_n * sin
        o2 = k_r2_n * cos + k_r1_n * sin

        # Store K
        k_out_base = pid_t * stride_ko_t + kv_head_id * HEAD_DIM
        tl.store(k_out_ptr + k_out_base + rot_offs, o1.to(k_out_ptr.dtype.element_ty))
        tl.store(
            k_out_ptr + k_out_base + HALF_ROTARY + rot_offs,
            o2.to(k_out_ptr.dtype.element_ty),
        )

        if HAS_NOPE:
            w_nope = (
                tl.load(
                    k_weight_ptr + ROTARY_DIM + nope_offs,
                    mask=nope_mask,
                    other=0.0,
                ).to(tl.float32)
                + 1.0
            )
            k_nope_n = k_nope * k_rstd * w_nope
            tl.store(
                k_out_ptr + k_out_base + ROTARY_DIM + nope_offs,
                k_nope_n.to(k_out_ptr.dtype.element_ty),
                mask=nope_mask,
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
    rotary_dim: Optional[int] = None,
    has_gate: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fused Q/K GemmaRMSNorm + NeoX RoPE + optional gate deinterleave.

    Args:
        q_gate: [T, num_q_heads * 2 * head_dim] when gated,
                [T, num_q_heads * head_dim] when not. May be non-contiguous.
        k: [T, num_kv_heads * head_dim]. May be non-contiguous.
        q_weight: [head_dim] raw GemmaRMSNorm weight (kernel applies +1.0).
        k_weight: [head_dim] raw GemmaRMSNorm weight.
        cos_sin_cache: [max_pos, rotary_dim] = [cos(half) || sin(half)].
        positions: [T] int token positions.
        eps: RMSNorm epsilon.
        num_q_heads: Number of Q heads (after TP split).
        num_kv_heads: Number of KV heads (after TP split).
        head_dim: Head dimension.
        rotary_dim: Rotary embedding dimension. Defaults to head_dim (full).
        has_gate: Whether q_gate contains interleaved Q+gate.

    Returns:
        (q, k, gate):
          q: [T, num_q_heads * head_dim]
          k: [T, num_kv_heads * head_dim]
          gate: [T, num_q_heads, head_dim] if has_gate else None
    """
    if rotary_dim is None:
        rotary_dim = head_dim

    T = q_gate.shape[0]
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    HALF_ROTARY = rotary_dim // 2
    NOPE_DIM = head_dim - rotary_dim
    HAS_NOPE = NOPE_DIM > 0
    NOPE_BLOCK = triton.next_power_of_2(NOPE_DIM) if NOPE_DIM > 0 else 1

    q_out = torch.empty(T, q_size, dtype=q_gate.dtype, device=q_gate.device)
    k_out = torch.empty(T, kv_size, dtype=k.dtype, device=k.device)

    if has_gate:
        gate_out = torch.empty(
            T, num_q_heads, head_dim, dtype=q_gate.dtype, device=q_gate.device
        )
    else:
        gate_out = q_out  # dummy

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
        ROTARY_DIM=rotary_dim,
        HALF_ROTARY=HALF_ROTARY,
        HAS_NOPE=HAS_NOPE,
        NOPE_BLOCK=NOPE_BLOCK,
        NOPE_DIM=NOPE_DIM,
        HAS_GATE=has_gate,
        num_warps=num_warps,
    )

    return q_out, k_out, gate_out if has_gate else None
