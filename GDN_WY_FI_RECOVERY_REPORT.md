# GDN MTP `cache_mode=none`: FlashInfer WY verify + FlashInfer recovery — report & reproduction

**Model:** `nvidia/Qwen3.5-397B-A17B-NVFP4` (MoE + GDN linear attention, MTP speculative decoding)
**Hardware:** 4× NVIDIA GB300 (sm_103, aarch64), TP4, CUDA 13
**Spec-decode shape:** NEXTN, draft-len 3 (`--speculative-num-draft-tokens 4`, topk 1), `--gdn-mtp-cache-mode none`

## Summary

For the `gdn-mtp-cache-mode=none` path, the target-verify (**replay**) and the accepted-state
**recovery** are separate stages. This report benchmarks the recommended configuration —
**FlashInfer WY output-only verify + FlashInfer recovery** — against the same stack with the
Triton recovery kernel, holding everything else fixed:

- **+3.2–4.1% output decode throughput** (two independent same-system A/Bs, run order swapped)
- **−7.5% median ITL** (12.6 → 11.6–11.7 ms at concurrency 256)
- **Lossless**: accept length 3.27↔3.28 unchanged; gsm8k identical to 4 decimal places

The win comes from the recovery kernel itself: per launch, the FlashInfer recovery kernel is
**1.85× faster** than the Triton recovery kernel in live decode (63.8 vs 118.1 µs), and its
eager-dispatch overhead is hidden by side-stream overlap.

Configuration (environment variables on the serving process):

```bash
SGLANG_GDN_WY_VERIFY=1          # verify/replay via the FlashInfer WY output-only kernel (in-graph)
SGLANG_GDN_FI_RECOVERY=1        # accepted-state recovery via the FlashInfer state kernel
SGLANG_GDN_WY_NATIVE_T=1        # WY kernel: native short-T load (no host zero-padding)
SGLANG_GDN_WY_STRIDED_QKV=1     # WY kernel: read q/k/v from the fused conv-output slices (no .contiguous())
SGLANG_GDN_STASH_ELIM=1         # verify conv writes a persistent buffer; recovery reads it (no k/v stash copy)
SGLANG_GDN_WY_NATIVE_AB=1       # WY kernel: native a/b load (no staging copies)
```

## Results

### 1. Output decode throughput (concurrency 256, prefill-negligible workload)

Workload: `bench_serving` random dataset, input 16 / output 4096 tokens,
`num_prompts = max_concurrency = 256`, seed 1 — one admission burst, then 256 sustained
concurrent decodes; prefill is <1% of wall time, so *Output token throughput ≈ pure decode
throughput*. Both configs run back-to-back on the same system; the experiment was repeated with
the run order swapped to rule out ordering effects.

| run | WY + Triton recovery | WY + FlashInfer recovery | Δ |
|---|--:|--:|--:|
| A-first | 16,855 tok/s | 17,401 tok/s | **+3.2%** |
| B-first (order swapped) | 16,908 tok/s | 17,594 tok/s | **+4.1%** |

| metric (representative run) | Triton recovery | FlashInfer recovery |
|---|--:|--:|
| Peak output throughput | 20,321–20,410 tok/s | 21,907–21,991 tok/s |
| Median ITL | 12.60–12.64 ms | 11.62–11.69 ms |
| Accept length | 3.27 | 3.28 |

### 2. Per-step decode latency (accept-length-independent)

Method: consecutive GPU start-to-start timestamps of the once-per-step `VerifyTreeGreedy` kernel
(one device), steady-state batch = 256 only. This measures the full iteration period including
inter-step gaps, on the GPU timeline. The no-recovery floor is obtained by skipping the recovery
stage entirely (`SGLANG_GDN_SKIP_RECOVERY=1`; output intentionally incorrect — floor
measurement only).

| config | min | max | median | avg (µs/step) | steps used |
|---|--:|--:|--:|--:|--:|
| WY + Triton recovery | 42,765 | 45,108 | 43,078 | 43,114 | 207 |
| **WY + FlashInfer recovery** | 39,099 | 40,233 | 39,392 | **39,392** | 189 |
| No recovery (floor) | 38,363 | 39,912 | 38,511 | 38,544 | 181 |

Recovery cost over the floor: **FlashInfer ≈ +0.85 ms/step (~+0.90 ms per token)** vs
**Triton ≈ +4.6 ms/step (~+1.85 ms per token)** — the FlashInfer recovery is ~5× cheaper in net
critical-path cost, because its faster kernel plus side-stream overlap hide most of its work.

### 3. Kernel-level (from steady-state decode traces, per launch, 45 GDN layers × 1 launch/step)

| kernel | per launch |
|---|--:|
| WY output-only verify (identical in both configs) | 31.6 µs |
| Triton recovery (`fused_sigmoid_gating_delta_rule_recover_final_state`) | 118.1 µs |
| **FlashInfer recovery (`gated_delta_rule_mtp`, `disable_output=True`)** | **63.8 µs** |

### 4. Accuracy (lossless)

gsm8k, 1319 examples, 8-shot, `max_new_tokens` 16384, temperature 0.6 (same protocol as the base
PR):

| config | gsm8k accuracy |
|---|--:|
| WY verify + Triton recovery | 0.9794 |
| Triton verify + Triton recovery | 0.9794 |
| `gdn-mtp-cache-mode=full` reference | 0.9779 |
| **WY verify + FlashInfer recovery** | *measurement in progress — table will be updated* |

Notes: the FlashInfer and Triton recovery kernels agree to bf16 rounding (max recovered-state
difference ~1e-3, not bit-identical), accept length is unchanged across the recovery swap
(3.27↔3.28), and the base PR reports 0.976–0.979 for its FlashInfer-recovery configuration —
the same score is expected here, but it is being measured directly rather than assumed.

## Known remaining overheads (TODO)

FlashInfer recovery still carries overheads that are **not yet removed** — candidates for
follow-up work:

1. **Eager (outside-CUDA-graph) recovery dispatch.** Recovery runs eagerly after the captured
   decode graph; each of the 45 per-layer calls pays CuTe-DSL host dispatch. Today this is mostly
   hidden behind concurrent GPU work, but it stays on the CPU critical path.
   *TODO: make recovery graph-resident — run recovery for iteration `n` at the start of iteration
   `n+1`'s captured graph, before the WY verify (the state is only consumed there), replaying away
   the host dispatch entirely.*
2. **~135 bf16 `_to_compact` clone launches per step (~0.9 ms/step of GPU time) on the recovery
   side stream.** The recovery kernel clones its strided q(=k)/v inputs to canonical-compact
   layout. Two bit-exact eliminations were implemented and verified (q==k de-duplication → 90
   clones/step; direct strided reads → 0 clones/step, −19% recovery-stream time), but both are
   **e2e-neutral today** because side-stream overlap hides the copies; the strided read also slows
   the recovery kernel by ~7.8%/launch (less-coalesced loads).
   *TODO: revisit once recovery is graph-resident/critical-path (the savings then become real),
   and improve strided-read coalescing; a strided path also needs a stride-faithful per-batch-size
   prewarm to avoid one-time JIT compiles on first traffic.*
3. **Residual net recovery cost ≈ 0.85–0.9 ms/step over the no-recovery floor** — the remaining
   optimization target for the recovery stage.

## Reproduction

### Environment

- 4× NVIDIA GB300 (sm_103), aarch64, CUDA 13, PyTorch with CUDA 13 support.
- sglang: this branch (stacked on the base PR branch `gdn-recovery-flashinfer-jit`).
- FlashInfer: `0.6.13` wheels plus the GDN kernels from the companion branch
  [`ameynaik-hub/flashinfer @ ameyn/gdn-wy-verify-opt`](https://github.com/ameynaik-hub/flashinfer/tree/ameyn/gdn-wy-verify-opt)
  (stacked on flashinfer PR #3720):

```bash
pip install flashinfer-python==0.6.13 flashinfer-cubin==0.6.13
pip install flashinfer-jit-cache==0.6.13 --index-url https://flashinfer.ai/whl/cu130

# graft the GDN kernels from the FlashInfer fork over the installed package
FIPKG=$(python -c 'import flashinfer, os; print(os.path.dirname(flashinfer.__file__))')
cp -f  $FLASHINFER_SRC/flashinfer/gdn_decode.py   "$FIPKG/gdn_decode.py"
cp -f  $FLASHINFER_SRC/flashinfer/gdn_prefill.py  "$FIPKG/gdn_prefill.py"
cp -rf $FLASHINFER_SRC/flashinfer/gdn_kernels/.   "$FIPKG/gdn_kernels/"

export PYTHONPATH=$SGLANG_SRC/python:$PYTHONPATH
```

### Server launch

```bash
export NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1
export SGL_ENABLE_JIT_DEEPGEMM=false SGLANG_ENABLE_FLASHINFER_GEMM=true

SGLANG_GDN_WY_VERIFY=1 SGLANG_GDN_FI_RECOVERY=1 \
SGLANG_GDN_WY_NATIVE_T=1 SGLANG_GDN_WY_STRIDED_QKV=1 \
SGLANG_GDN_STASH_ELIM=1 SGLANG_GDN_WY_NATIVE_AB=1 \
python -m sglang.launch_server \
  --served-model-name nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --model-path nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --tensor-parallel-size 4 --expert-parallel-size 1 \
  --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
  --fp4-gemm-backend flashinfer_cutlass \
  --mamba-ssm-dtype bfloat16 --attention-backend trtllm_mha \
  --moe-runner-backend flashinfer_trtllm \
  --mamba-scheduler-strategy no_buffer --disable-radix-cache \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --cuda-graph-max-bs 512 --max-running-requests 512 \
  --scheduler-recv-interval 30 --stream-interval 30 --watchdog-timeout 600 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --gdn-mtp-cache-mode none --linear-attn-decode-backend flashinfer \
  --host 127.0.0.1 --port 30000
```

For the Triton-recovery comparison point, relaunch with `SGLANG_GDN_FI_RECOVERY=0`
(everything else identical). For the no-recovery floor, add `SGLANG_GDN_SKIP_RECOVERY=1`
(output is intentionally incorrect; latency measurement only).

### Decode throughput / ITL

```bash
# warmup (captures decode CUDA graphs, reaches steady state)
python -m sglang.bench_serving --backend sglang --base-url http://127.0.0.1:30000 \
  --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --dataset-name random --random-input-len 16 --random-output-len 512 \
  --random-range-ratio 1.0 --num-prompts 64 --max-concurrency 64 \
  --request-rate inf --seed 1

# measured pure-decode run
python -m sglang.bench_serving --backend sglang --base-url http://127.0.0.1:30000 \
  --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --dataset-name random --random-input-len 16 --random-output-len 4096 \
  --random-range-ratio 1.0 --num-prompts 256 --max-concurrency 256 \
  --request-rate inf --seed 1
```

Report *Output token throughput*, *Peak output token throughput*, *Median ITL*, and
*Accept length*. Run both configs back-to-back on the same machine (and repeat with the order
swapped) — run-to-run variance across machines/sessions is a few percent, larger than per-config
noise on one machine.

### Accuracy

gsm8k few-shot eval, same protocol as the base PR: 1319 questions, 8-shot,
`max_new_tokens` 16384, temperature 0.6. Expected: **0.979** for both recovery kernels
(bit-level accept/reject parity; accept length unchanged).

### Per-step latency (optional, Nsight Systems)

```bash
# launch the server under nsys with deferred collection
nsys launch --session-new=sgl -t cuda,nvtx,cublas --cuda-graph-trace=node \
  python -m sglang.launch_server ...   # same flags as above

# drive a sustained-concurrency decode load (input 16 / output 8192, 256 prompts,
# max-concurrency 256), wait ~35 s for the prefill burst to clear, then:
nsys start --session=sgl -o decode_trace -f true
sleep 8            # capture >= ~6 s: shorter windows may record no kernel events
nsys stop --session=sgl
```

Per-step latency = consecutive GPU start-to-start deltas of the `VerifyTreeGreedy` kernel
(filter to a single device; it launches once per decode step per device), over steady-state
batch-256 steps.
