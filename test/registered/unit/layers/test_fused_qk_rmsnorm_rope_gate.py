from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

import itertools

import pytest
import torch

from sglang.srt.layers.fused_qk_rmsnorm_rope_gate import (
    fused_qk_gemma_rmsnorm_rope_gate,
)

DTYPES = [torch.float16, torch.bfloat16]
TOKEN_COUNTS = [1, 4, 16, 64, 256, 1024]
HEAD_DIMS = [128]
NUM_Q_HEADS_LIST = [8, 32]
NUM_KV_HEADS_LIST = [2, 8]


def _ref_gemma_rmsnorm(x, weight, eps):
    """Reference GemmaRMSNorm: x * rsqrt(mean(x^2) + eps) * (weight + 1.0)."""
    x_fp32 = x.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    return (x_fp32 * torch.rsqrt(var + eps) * (weight.float() + 1.0)).to(x.dtype)


def _ref_neox_rope(x, cos, sin):
    """Reference NeoX-style RoPE on x: [T, num_heads, head_dim]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1, o2], dim=-1)


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
    has_gate,
):
    """Full reference: deinterleave + GemmaRMSNorm + NeoX RoPE."""
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

    # RoPE cos/sin
    half = head_dim // 2
    cs = cos_sin_cache[positions]  # [T, head_dim]
    cos = cs[:, :half].unsqueeze(1).float()  # [T, 1, half]
    sin = cs[:, half:].unsqueeze(1).float()

    q_roped = _ref_neox_rope(q_normed.view(T, num_q_heads, head_dim).float(), cos, sin)
    q_out = q_roped.to(q_gate.dtype).view(T, -1)

    k_roped = _ref_neox_rope(k_normed.view(T, num_kv_heads, head_dim).float(), cos, sin)
    k_out = k_roped.to(k.dtype).view(T, -1)

    return q_out, k_out, gate


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


def _make_inputs(T, num_q_heads, num_kv_heads, head_dim, dtype, has_gate):
    """Create inputs mimicking qkv.split() non-contiguous views."""
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q_gate_size = q_size * 2 if has_gate else q_size

    # Simulate qkv.split() producing non-contiguous views
    qkv = torch.randn(T, q_gate_size + kv_size + kv_size, dtype=dtype, device="cuda")
    q_gate, k, v = qkv.split([q_gate_size, kv_size, kv_size], dim=-1)

    q_weight = torch.randn(head_dim, dtype=dtype, device="cuda") * 0.1
    k_weight = torch.randn(head_dim, dtype=dtype, device="cuda") * 0.1

    cos_sin_cache = torch.randn(8192, head_dim, dtype=torch.float32, device="cuda")
    positions = torch.randint(0, 8192, (T,), device="cuda")

    return q_gate, k, q_weight, k_weight, cos_sin_cache, positions


@pytest.mark.parametrize(
    "num_tokens, num_q_heads, num_kv_heads, dtype, has_gate",
    list(
        itertools.product(
            [1, 16, 256, 1024],
            NUM_Q_HEADS_LIST,
            NUM_KV_HEADS_LIST,
            DTYPES,
            [True, False],
        )
    ),
)
def test_correctness(num_tokens, num_q_heads, num_kv_heads, dtype, has_gate):
    head_dim = 128
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    eps = 1e-6

    q_gate, k, q_w, k_w, cs_cache, pos = _make_inputs(
        num_tokens, num_q_heads, num_kv_heads, head_dim, dtype, has_gate
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
        has_gate=has_gate,
    )

    torch.testing.assert_close(q_out, q_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_out, k_ref, rtol=rtol, atol=atol)

    if has_gate:
        assert gate_out is not None
        torch.testing.assert_close(gate_out, gate_ref, rtol=0, atol=0)
    else:
        assert gate_out is None


@pytest.mark.parametrize("dtype", DTYPES)
def test_single_token(dtype):
    head_dim = 128
    num_q_heads, num_kv_heads = 32, 4
    eps = 1e-6
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)

    q_gate, k, q_w, k_w, cs_cache, pos = _make_inputs(
        1, num_q_heads, num_kv_heads, head_dim, dtype, has_gate=True
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
        True,
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
        has_gate=True,
    )

    torch.testing.assert_close(q_out, q_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_out, k_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(gate_out, gate_ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_output_shapes(dtype):
    T, num_q_heads, num_kv_heads, head_dim = 64, 32, 8, 128
    eps = 1e-6

    q_gate, k, q_w, k_w, cs_cache, pos = _make_inputs(
        T, num_q_heads, num_kv_heads, head_dim, dtype, has_gate=True
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
        has_gate=True,
    )

    assert q_out.shape == (T, num_q_heads * head_dim)
    assert k_out.shape == (T, num_kv_heads * head_dim)
    assert gate_out.shape == (T, num_q_heads, head_dim)
    assert q_out.dtype == dtype
    assert k_out.dtype == dtype
    assert gate_out.dtype == dtype
    assert q_out.is_contiguous()
    assert k_out.is_contiguous()
    assert gate_out.is_contiguous()


@pytest.mark.parametrize("dtype", DTYPES)
def test_sequential_positions(dtype):
    """Test with sequential positions (typical prefill pattern)."""
    T, num_q_heads, num_kv_heads, head_dim = 256, 16, 4, 128
    eps = 1e-6
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)

    q_gate, k, q_w, k_w, cs_cache, _ = _make_inputs(
        T, num_q_heads, num_kv_heads, head_dim, dtype, has_gate=True
    )
    positions = torch.arange(T, device="cuda")

    q_ref, k_ref, gate_ref = _reference(
        q_gate,
        k,
        q_w,
        k_w,
        cs_cache,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        True,
    )

    q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        q_w,
        k_w,
        cs_cache,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        has_gate=True,
    )

    torch.testing.assert_close(q_out, q_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_out, k_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_contiguous_input(dtype):
    """Test with fully contiguous inputs (not from qkv.split())."""
    T, num_q_heads, num_kv_heads, head_dim = 64, 16, 4, 128
    eps = 1e-6
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)

    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim

    q_gate = torch.randn(T, q_size * 2, dtype=dtype, device="cuda")
    k = torch.randn(T, kv_size, dtype=dtype, device="cuda")
    q_w = torch.randn(head_dim, dtype=dtype, device="cuda") * 0.1
    k_w = torch.randn(head_dim, dtype=dtype, device="cuda") * 0.1
    cs_cache = torch.randn(8192, head_dim, dtype=torch.float32, device="cuda")
    pos = torch.randint(0, 8192, (T,), device="cuda")

    assert q_gate.is_contiguous()
    assert k.is_contiguous()

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
        True,
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
        has_gate=True,
    )

    torch.testing.assert_close(q_out, q_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_out, k_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(gate_out, gate_ref, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
