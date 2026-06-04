from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

import itertools

import pytest
import torch

from sglang.srt.layers.fused_qk_rmsnorm_rope_gate import (
    fused_qk_gemma_rmsnorm_rope_gate,
    fused_qk_gemma_rmsnorm_rope_gate_v2,
)

DTYPES = [torch.float16, torch.bfloat16]


def _ref_gemma_rmsnorm(x, weight, eps):
    """Reference GemmaRMSNorm: x * rsqrt(mean(x^2) + eps) * (weight + 1.0)."""
    x_fp32 = x.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    return (x_fp32 * torch.rsqrt(var + eps) * (weight.float() + 1.0)).to(x.dtype)


def _ref_gemma_rmsnorm_with_gemma_weight(x, gemma_weight, eps):
    """Same but takes pre-computed gemma_weight (already +1), for v2 testing."""
    x_fp32 = x.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    return (x_fp32 * torch.rsqrt(var + eps) * gemma_weight.float()).to(x.dtype)


def _ref_neox_rope_partial(x, cos, sin, rotary_dim):
    """Reference NeoX RoPE with partial rotation on x: [T, num_heads, head_dim]."""
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half = rotary_dim // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1, o2, x_pass], dim=-1)


def _reference(
    q_gate,
    k,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    eps,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    has_gate,
):
    """Full reference: deinterleave + GemmaRMSNorm + partial NeoX RoPE."""
    T = q_gate.shape[0]
    if has_gate:
        q_gate_3d = q_gate.view(T, num_q_heads, 2 * head_dim)
        q = q_gate_3d[..., :head_dim].contiguous().reshape(T, -1)
        gate = q_gate_3d[..., head_dim:].contiguous()
    else:
        q = q_gate.contiguous()
        gate = None

    # Per-head GemmaRMSNorm
    q_by_head = q.reshape(-1, head_dim)
    q_normed = _ref_gemma_rmsnorm(q_by_head, q_weight, eps).view(T, -1)

    k_by_head = k.contiguous().reshape(-1, head_dim)
    k_normed = _ref_gemma_rmsnorm(k_by_head, k_weight, eps).view(T, -1)

    # RoPE cos/sin (cache shape: [max_pos, rotary_dim])
    half = rotary_dim // 2
    cs = cos_sin_cache[positions]  # [T, rotary_dim]
    cos = cs[:, :half].unsqueeze(1).float()  # [T, 1, half]
    sin = cs[:, half:].unsqueeze(1).float()

    q_roped = _ref_neox_rope_partial(
        q_normed.view(T, num_q_heads, head_dim).float(), cos, sin, rotary_dim
    )
    q_out = q_roped.to(q_gate.dtype).view(T, -1)

    k_roped = _ref_neox_rope_partial(
        k_normed.view(T, num_kv_heads, head_dim).float(), cos, sin, rotary_dim
    )
    k_out = k_roped.to(k.dtype).view(T, -1)

    return q_out, k_out, gate


def _reference_v2(
    q_gate,
    k,
    q_gemma_weight,
    k_gemma_weight,
    cos_sin_cache,
    positions,
    eps,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    has_gate,
):
    """Reference for v2: uses gemma_weight (already +1), bf16 round-trip after norm."""
    T = q_gate.shape[0]
    if has_gate:
        q_gate_3d = q_gate.view(T, num_q_heads, 2 * head_dim)
        q = q_gate_3d[..., :head_dim].contiguous().reshape(T, -1)
        gate = q_gate_3d[..., head_dim:].contiguous()
    else:
        q = q_gate.contiguous()
        gate = None

    q_by_head = q.reshape(-1, head_dim)
    q_normed = _ref_gemma_rmsnorm_with_gemma_weight(
        q_by_head, q_gemma_weight, eps
    ).view(T, -1)

    k_by_head = k.contiguous().reshape(-1, head_dim)
    k_normed = _ref_gemma_rmsnorm_with_gemma_weight(
        k_by_head, k_gemma_weight, eps
    ).view(T, -1)

    half = rotary_dim // 2
    cs = cos_sin_cache[positions]
    cos = cs[:, :half].unsqueeze(1).float()
    sin = cs[:, half:].unsqueeze(1).float()

    q_roped = _ref_neox_rope_partial(
        q_normed.view(T, num_q_heads, head_dim).float(), cos, sin, rotary_dim
    )
    q_out = q_roped.to(q_gate.dtype).view(T, -1)

    k_roped = _ref_neox_rope_partial(
        k_normed.view(T, num_kv_heads, head_dim).float(), cos, sin, rotary_dim
    )
    k_out = k_roped.to(k.dtype).view(T, -1)

    return q_out, k_out, gate


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


def _make_inputs(T, num_q_heads, num_kv_heads, head_dim, rotary_dim, dtype, has_gate):
    """Create inputs mimicking qkv.split() non-contiguous views."""
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q_gate_size = q_size * 2 if has_gate else q_size

    qkv = torch.randn(T, q_gate_size + kv_size + kv_size, dtype=dtype, device="cuda")
    q_gate, k, v = qkv.split([q_gate_size, kv_size, kv_size], dim=-1)

    q_weight = torch.randn(head_dim, dtype=dtype, device="cuda") * 0.1
    k_weight = torch.randn(head_dim, dtype=dtype, device="cuda") * 0.1

    cos_sin_cache = torch.randn(8192, rotary_dim, dtype=torch.float32, device="cuda")
    positions = torch.randint(0, 8192, (T,), device="cuda")

    return q_gate, k, q_weight, k_weight, cos_sin_cache, positions


# --- Qwen3.5 MoE config: head_dim=256, rotary_dim=64 (partial_rotary_factor=0.25) ---


@pytest.mark.parametrize(
    "num_tokens, dtype, has_gate",
    list(itertools.product([1, 16, 256, 1024], DTYPES, [True, False])),
)
def test_qwen35_config(num_tokens, dtype, has_gate):
    """Test with real Qwen3.5 MoE config: hd=256, rd=64, qh=32, kvh=2."""
    head_dim, rotary_dim = 256, 64
    num_q_heads, num_kv_heads = 32, 2
    rtol, atol = (3e-2, 3e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    eps = 1e-6

    q_gate, k, q_w, k_w, cs_cache, pos = _make_inputs(
        num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim, dtype, has_gate
    )
    q_ref, k_ref, gate_ref = _reference(
        q_gate,
        k,
        q_w,
        k_w,
        cs_cache,
        pos,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        has_gate,
    )
    q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        q_w,
        k_w,
        cs_cache,
        pos,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim=rotary_dim,
        has_gate=has_gate,
    )

    torch.testing.assert_close(q_out, q_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_out, k_ref, rtol=rtol, atol=atol)
    if has_gate:
        assert gate_out is not None
        torch.testing.assert_close(gate_out, gate_ref, rtol=0, atol=0)
    else:
        assert gate_out is None


# --- Full rotation config (rotary_dim == head_dim) ---


@pytest.mark.parametrize(
    "num_tokens, num_q_heads, num_kv_heads, dtype, has_gate",
    list(
        itertools.product(
            [1, 16, 256],
            [8, 32],
            [2, 8],
            DTYPES,
            [True, False],
        )
    ),
)
def test_full_rotation(num_tokens, num_q_heads, num_kv_heads, dtype, has_gate):
    """Test with full rotation (rotary_dim == head_dim)."""
    head_dim = 128
    rotary_dim = head_dim
    rtol, atol = (3e-2, 3e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    eps = 1e-6

    q_gate, k, q_w, k_w, cs_cache, pos = _make_inputs(
        num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim, dtype, has_gate
    )
    q_ref, k_ref, gate_ref = _reference(
        q_gate,
        k,
        q_w,
        k_w,
        cs_cache,
        pos,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        has_gate,
    )
    q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        q_w,
        k_w,
        cs_cache,
        pos,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim=rotary_dim,
        has_gate=has_gate,
    )

    torch.testing.assert_close(q_out, q_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_out, k_ref, rtol=rtol, atol=atol)
    if has_gate:
        torch.testing.assert_close(gate_out, gate_ref, rtol=0, atol=0)
    else:
        assert gate_out is None


@pytest.mark.parametrize("dtype", DTYPES)
def test_output_shapes_partial(dtype):
    """Verify shapes with partial rotation."""
    T, num_q_heads, num_kv_heads, head_dim, rotary_dim = 64, 32, 2, 256, 64
    eps = 1e-6

    q_gate, k, q_w, k_w, cs_cache, pos = _make_inputs(
        T, num_q_heads, num_kv_heads, head_dim, rotary_dim, dtype, has_gate=True
    )
    q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        q_w,
        k_w,
        cs_cache,
        pos,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim=rotary_dim,
        has_gate=True,
    )

    assert q_out.shape == (T, num_q_heads * head_dim)
    assert k_out.shape == (T, num_kv_heads * head_dim)
    assert gate_out.shape == (T, num_q_heads, head_dim)
    assert q_out.is_contiguous()
    assert k_out.is_contiguous()
    assert gate_out.is_contiguous()


# --- v2 kernel (bf16 round-trip, matches unfused path) ---


@pytest.mark.parametrize(
    "num_tokens, dtype, has_gate",
    list(itertools.product([1, 16, 256, 1024], DTYPES, [True, False])),
)
def test_v2_qwen35_config(num_tokens, dtype, has_gate):
    """v2 kernel with real Qwen3.5 MoE config."""
    head_dim, rotary_dim = 256, 64
    num_q_heads, num_kv_heads = 32, 2
    rtol, atol = (3e-2, 3e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    eps = 1e-6

    q_gate, k, q_w, k_w, cs_cache, pos = _make_inputs(
        num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim, dtype, has_gate
    )
    # v2 uses gemma_weight (already +1, in same dtype as model weights)
    q_w_gemma = q_w + 1.0
    k_w_gemma = k_w + 1.0

    q_ref, k_ref, gate_ref = _reference_v2(
        q_gate,
        k,
        q_w_gemma,
        k_w_gemma,
        cs_cache,
        pos,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        has_gate,
    )
    q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_rope_gate_v2(
        q_gate,
        k,
        q_w_gemma,
        k_w_gemma,
        cs_cache,
        pos,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim=rotary_dim,
        has_gate=has_gate,
    )

    torch.testing.assert_close(q_out, q_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_out, k_ref, rtol=rtol, atol=atol)
    if has_gate:
        assert gate_out is not None
        torch.testing.assert_close(gate_out, gate_ref, rtol=0, atol=0)
    else:
        assert gate_out is None


@pytest.mark.parametrize(
    "num_tokens, dtype, has_gate",
    list(itertools.product([1, 16, 256], DTYPES, [True, False])),
)
def test_v2_full_rotation(num_tokens, dtype, has_gate):
    """v2 kernel with full rotation (rotary_dim == head_dim)."""
    head_dim, rotary_dim = 128, 128
    num_q_heads, num_kv_heads = 32, 4
    rtol, atol = (3e-2, 3e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    eps = 1e-6

    q_gate, k, q_w, k_w, cs_cache, pos = _make_inputs(
        num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim, dtype, has_gate
    )
    q_w_gemma = q_w + 1.0
    k_w_gemma = k_w + 1.0

    q_ref, k_ref, gate_ref = _reference_v2(
        q_gate,
        k,
        q_w_gemma,
        k_w_gemma,
        cs_cache,
        pos,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        has_gate,
    )
    q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_rope_gate_v2(
        q_gate,
        k,
        q_w_gemma,
        k_w_gemma,
        cs_cache,
        pos,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim=rotary_dim,
        has_gate=has_gate,
    )

    torch.testing.assert_close(q_out, q_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_out, k_ref, rtol=rtol, atol=atol)
    if has_gate:
        torch.testing.assert_close(gate_out, gate_ref, rtol=0, atol=0)
    else:
        assert gate_out is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
