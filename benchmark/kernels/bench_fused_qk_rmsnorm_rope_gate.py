"""Benchmark fused_qk_gemma_rmsnorm_rope_gate: v1 vs v2 vs unfused.

Real Qwen3.5 MoE config: head_dim=256, rotary_dim=64 (partial_rotary_factor=0.25),
num_attention_heads=32, num_key_value_heads=2.
"""

import torch
import triton

from sglang.srt.layers.fused_qk_rmsnorm_rope_gate import (
    fused_qk_gemma_rmsnorm_rope_gate,
    fused_qk_gemma_rmsnorm_rope_gate_v2,
)


def _unfused_reference(
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
):
    T = q_gate.shape[0]
    q_gate_3d = q_gate.view(T, num_q_heads, 2 * head_dim)
    q, gate = torch.chunk(q_gate_3d, 2, dim=-1)
    q = q.reshape(T, -1)

    q_by_head = q.reshape(-1, head_dim).float()
    q_var = q_by_head.pow(2).mean(-1, keepdim=True)
    q_normed = (q_by_head * torch.rsqrt(q_var + eps) * (q_weight.float() + 1.0)).to(
        q.dtype
    )
    q = q_normed.view(T, -1)

    k_by_head = k.contiguous().reshape(-1, head_dim).float()
    k_var = k_by_head.pow(2).mean(-1, keepdim=True)
    k_normed = (k_by_head * torch.rsqrt(k_var + eps) * (k_weight.float() + 1.0)).to(
        k.dtype
    )
    k_out = k_normed.view(T, -1)

    half = rotary_dim // 2
    cs = cos_sin_cache[positions]
    cos = cs[:, :half].unsqueeze(1)
    sin = cs[:, half:].unsqueeze(1)

    q_3d = q.view(T, num_q_heads, head_dim).float()
    q_rot, q_pass = q_3d[..., :rotary_dim], q_3d[..., rotary_dim:]
    q1, q2 = q_rot[..., :half], q_rot[..., half:]
    q_out = (
        torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin, q_pass], -1)
        .to(q.dtype)
        .view(T, -1)
    )

    k_3d = k_out.view(T, num_kv_heads, head_dim).float()
    k_rot, k_pass = k_3d[..., :rotary_dim], k_3d[..., rotary_dim:]
    k1, k2 = k_rot[..., :half], k_rot[..., half:]
    k_out = (
        torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin, k_pass], -1)
        .to(k.dtype)
        .view(T, -1)
    )

    return q_out, k_out, gate


CONFIGS = [
    (32, 2, 256, 64),  # Qwen3.5 MoE TP1
    (16, 1, 256, 64),  # Qwen3.5 MoE TP2
]


def make_bench(nqh, nkvh, hd, rd):
    eps = 1e-6

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            line_arg="impl",
            line_vals=["v2", "v1", "unfused"],
            line_names=["v2 (bf16 roundtrip)", "v1 (all-fp32)", "Unfused PyTorch"],
            styles=[("green", "-"), ("blue", "--"), ("orange", ":")],
            ylabel="us",
            plot_name=f"qk_rmsnorm_rope_gate-qh{nqh}_kvh{nkvh}_hd{hd}_rd{rd}",
            args={"nqh": nqh, "nkvh": nkvh, "hd": hd, "rd": rd},
        )
    )
    def bench(num_tokens, impl, nqh, nkvh, hd, rd, dtype=torch.bfloat16):
        qg_size = nqh * 2 * hd
        kv_size = nkvh * hd
        qkv = torch.randn(
            num_tokens, qg_size + kv_size + kv_size, dtype=dtype, device="cuda"
        )
        q_gate, k, _ = qkv.split([qg_size, kv_size, kv_size], dim=-1)
        q_w_raw = torch.randn(hd, dtype=dtype, device="cuda") * 0.1
        k_w_raw = torch.randn(hd, dtype=dtype, device="cuda") * 0.1
        q_w_gemma = q_w_raw + 1.0
        k_w_gemma = k_w_raw + 1.0
        cache = torch.randn(8192, rd, dtype=torch.float32, device="cuda")
        pos = torch.randint(0, 8192, (num_tokens,), device="cuda")

        if impl == "v2":
            fn = lambda: fused_qk_gemma_rmsnorm_rope_gate_v2(
                q_gate,
                k,
                q_w_gemma,
                k_w_gemma,
                cache,
                pos,
                eps,
                nqh,
                nkvh,
                hd,
                rotary_dim=rd,
                has_gate=True,
            )
        elif impl == "v1":
            fn = lambda: fused_qk_gemma_rmsnorm_rope_gate(
                q_gate,
                k,
                q_w_raw,
                k_w_raw,
                cache,
                pos,
                eps,
                nqh,
                nkvh,
                hd,
                rotary_dim=rd,
                has_gate=True,
            )
        else:
            fn = lambda: _unfused_reference(
                q_gate,
                k,
                q_w_raw,
                k_w_raw,
                cache,
                pos,
                eps,
                nqh,
                nkvh,
                hd,
                rd,
            )

        ms = triton.testing.do_bench(fn, warmup=100, rep=200)
        return ms * 1000

    return bench


if __name__ == "__main__":
    for nqh, nkvh, hd, rd in CONFIGS:
        print(f"\n===== Qwen3.5: qh={nqh}, kvh={nkvh}, hd={hd}, rd={rd} =====")
        make_bench(nqh, nkvh, hd, rd).run(print_data=True)
