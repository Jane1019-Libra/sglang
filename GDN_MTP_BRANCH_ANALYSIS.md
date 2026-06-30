# GDN MTP Recovery Branch Analysis

**Branches covered:**
- `gdn-recovery-overlap` — baseline per-layer recovery + stream overlap
- `review/gdn-recovery-flashinfer` — FlashInfer kernel swap + JIT prewarm (builds on the above)

---

## Background: What problem is being solved?

### GDN 架构简介

**GDN (Gated Delta Network)** 是 Qwen 3.5 使用的 hybrid 架构中的 linear attention 组件。其 SSM 状态更新规则是「gated delta rule」：

```
h_{t+1} = h_t * exp(g_t)  +  k_t * (v_t - h_t^T @ k_t)^T
```

其中：
- `g_t = A_coeff * softplus(a_t + dt_bias)` — gating decay
- `beta_t = sigmoid(b_t)` — update gate
- `h_t ∈ R^{K × V}` — recurrent SSM state（每个 sequence 的核心状态）

这是一个 **recurrent** 模型：每个 token 依赖前一步的 `h`，无法并行化（不像 attention 可以用 flash-attn 一次处理整个序列）。

### MTP 推理下的 SSM 状态管理问题

**MTP（Multi-Token Prediction）= speculative decoding** — draft model 一次生成多个候选 token，target model 并行验证。

验证阶段流程：
```
[request] → draft(N tokens) → target verify → accept k ≤ N tokens → commit h_{k}
```

关键问题：**target verify 阶段是并行处理 draft tokens，但 SSM 是 recurrent 的** — 正确的 `h_{accepted}` 需要从 `h_0` 跑完 `k` 步迭代后才能得到。

**原始方案（`full` cache mode）**：在 target verify 的前向时，把每一步的中间状态 `h_1, h_2, ..., h_N` 全部缓存下来。Accept k 个 token 后直接从缓存中取出 `h_k`。

**问题**：这个缓存非常大：
```
shape: [num_layers, pool_size+1, draft_token_num, H_V, K, V]
```
对于 Qwen 3.5 的规模，这是 GB 级别的显存开销。

---

## Branch 1: `gdn-recovery-overlap`

共 3 个 GDN 专属 commit（在 `main` 之上）：

### Commit 1: `df6d7ba025` — [GDN] Add MTP cache mode plumbing

**作者**: Yangmin Li | **日期**: Jun 1, 2026

**目标**: 引入 `gdn_mtp_cache_mode` flag，让用户可以选择跳过中间状态缓存。

#### 修改文件

**`python/sglang/srt/server_args.py`**

新增 server arg：
```python
gdn_mtp_cache_mode: str = "full"  # choices: ["full", "none"]
```

- `"full"`: 原始行为，缓存每个 draft token 的 h-state
- `"none"`: 跳过中间缓存，事后重算 `h_{accepted}`

**`python/sglang/srt/mem_cache/memory_pool.py`**

修改 `MambaPool.SpeculativeState` dataclass：

```python
@dataclass(frozen=True, kw_only=True)
class SpeculativeState(State):
    # None in gdn_mtp_cache_mode=none; recovery reconstructs h_K after verify.
    intermediate_ssm: Optional[torch.Tensor] = None
    # Kept in both modes for accepted conv-state rollback.
    intermediate_conv_window: Optional[List[torch.Tensor]] = None
```

核心逻辑：
- `cache_mode == "none"` → `intermediate_ssm = None`（省去最大的那块显存）
- Conv window cache 在两种模式都保留（conv 状态的 rollback 仍依赖它）
- `get_tensor_size_bytes()` 适配 `None` 入参（不再 crash）

**设计理由**: `intermediate_ssm` 是 speculative 推理中最重的显存消耗（GB 级）；conv window 相对小，且 rollback 逻辑不能绕开它。

---

### Commit 2: `f2e54fa6b1` — [GDN] Recompute final MTP state without cache

**作者**: Yangmin Li | **日期**: Jun 1, 2026

**目标**: 实现 `cache_mode=none` 时的 SSM 状态恢复路径。

#### 新增 Triton kernel（`fla/fused_sigmoid_gating_recurrent.py`）

新增函数 `fused_sigmoid_gating_delta_rule_recover_final_state`，对应 kernel `fused_sigmoid_gating_delta_rule_recover_final_state_kernel`。

**与原始 kernel 的关键区别**：

| 原始 target_verify kernel | 新增 recovery kernel |
|---|---|
| 每步写中间 output | 只写最终 h_K，无 output |
| 每步写中间 h（缓存） | 从 h_0 跑 T 步，只写 h_{accepted} |
| 并行化高，但显存大 | 显存零开销，per-layer 串行 |

**Kernel 设计细节**：
- 入参 `accepted_steps[N]` (int32)：每个 sequence 真正接受了多少步
- 用 `active = step_idx <= accepted_step` mask 控制迭代边界（而非 early exit），triton-friendly
- 从 `h0_source[h0_indices[i]]` 读初始状态，计算完后写回同一 slot
- `USE_QK_L2NORM_IN_KERNEL=True`，`IS_KDA=False` 对应 Qwen 3.5 的 GDN 头配置

**`python/sglang/srt/layers/attention/linear/gdn_backend.py`**（`GDNAttnBackend`）

新增两个字段：
```python
self._no_cache_stash: Dict[int, Dict[str, torch.Tensor]] = {}
# key: layer_id, value: {k, v, a, b, A_log, dt_bias}
self._no_cache_draft_token_num: Optional[int] = None
```

在 `target_verify` 前向中，`intermediate_state_cache is None` 时：
- 分配 persistent buffer（`max_tokens = pool_size × draft_token_num`，覆盖所有 CUDA graph bs）
- 用 `stash_entry["k"][:, :actual_seq_len].copy_(key)` — **in-place 拷贝**，保持地址稳定（CUDA graph replay 需要固定地址）
- Buffer 分配在 `torch.inference_mode(False)` 上下文里，避免 inference mode 的不可变约束

**`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`**（`HybridLinearAttnBackend`）

原来 `_mamba_verify_update()` 里无条件调用 `fused_mamba_state_scatter_with_mask()`；现在分支：

```python
if intermediate_state_cache is not None:
    # mode=full: h_K 已从缓存里取
    fused_mamba_state_scatter_with_mask(...)
else:
    # mode=none: 从 h_0 重算
    self._no_cache_mtp_recompute(accepted_steps, state_indices_tensor)
```

新增 `_no_cache_mtp_recompute()`：遍历每个 GDN layer，调用 recovery kernel，把 `h_{accepted}` 写回 SSM pool。

---

### Commit 3: `36409229d0` — Overlap GDN cache_mode=none SSM recovery on a dedicated side stream

**作者**: Wenjing Lu | **日期**: Jun 15, 2026

**目标**: 把 recovery 的 GPU 工作搬到 side stream，与下一步的 draft phase 重叠执行，提升吞吐。

#### 设计思路

```
主 stream 时间线:
  step N: [draft] → [target_verify] → [mamba_verify_update → kick recovery to side_stream] → return
  step N+1: [init_forward_metadata → wait_event] → [draft] → [target_verify] → ...

side_stream 时间线:
  step N:                                                  [SSM recovery kernels (45 layers)]
  step N+1:                                                              ^ done before wait_event
```

recovery 的 GPU 工作（45 层 GDN 各跑一个 Triton kernel）隐藏在下一步的 draft phase 里。

#### 具体修改

**新增字段**（`HybridLinearAttnBackend.__init__`）：
```python
self._recovery_stream: Optional[torch.cuda.Stream] = None
self._recovery_event: Optional[torch.cuda.Event] = None
self._recovery_event_pending: bool = False
```

**新增 `_wait_recovery_if_pending()`**：
```python
if self._recovery_event_pending:
    torch.cuda.current_stream().wait_event(self._recovery_event)
    self._recovery_event_pending = False
```
在 `init_forward_metadata()` 和 `init_forward_metadata_replay_cuda_graph()` 开头调用，确保下一次 target forward 开始前 SSM pool 已经被正确更新。

**修改 `_no_cache_mtp_recompute()`**：

```python
def _run_recovery():
    for layer_id, stash in stash_per_layer.items():
        fused_sigmoid_gating_delta_rule_recover_final_state(...)

if torch.cuda.is_current_stream_capturing():
    # CUDA graph capture: 必须在 capture stream 上运行
    _run_recovery()
    return

# 否则走 side stream 路径：
self._recovery_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(self._recovery_stream):
    _run_recovery()
self._recovery_event.record(self._recovery_stream)
# 关键：pin 新分配的 int32 tensors，防止 allocator 在 side stream 还在用时回收
state_idx_i32.record_stream(self._recovery_stream)
accepted_steps_i32.record_stream(self._recovery_stream)
self._recovery_event_pending = True
```

**CUDA graph 下的正确性**：`init_forward_metadata_replay_cuda_graph()` 被重写，确保 replay 前也调用 `_wait_recovery_if_pending()`。注释说明：在 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` 模式下，wait 发生在 plan_stream，correctness 依赖 `eagle_worker_v2.verify` 的 `fwd_stream.wait_stream(plan_stream)` 传递性。

#### 性能数据

| batch size | 相比 per-layer base（无 overlap） |
|---|---|
| bs64 | -4%（overhead 大于收益） |
| bs128/256 | +5% |
| bs512 | +1.6%（GPU 已饱和） |
| vs batched recovery（所有层一次 kernel） | +22% (bs512) |

---

## Branch 2: `review/gdn-recovery-flashinfer`

在 `gdn-recovery-overlap` 的所有 3 个 commit 基础上，再追加 2 个 commit：

### Commit 4: `cc0a008155` — GDN recovery: B-padding + startup prewarm + zero-copy stash view for FlashInfer JIT stability

**作者**: Wenjing Lu | **日期**: Jun 20, 2026

**目标**: SM100（GB200）上替换 Triton recovery kernel → FlashInfer JIT kernel（PR #3502），并解决 JIT 冷启动 recompile 问题。

#### 问题背景

FlashInfer 的 `gated_delta_rule_mtp` kernel 使用 **CuTe DSL JIT 编译**，每个 `(B, T)` 组合第一次运行时需要约 **37 秒**重编译。如果每个不同的 bs 都触发重编，生产中将产生不可接受的首请求延迟。

#### 解法 1: B-padding（在 `_no_cache_mtp_recompute()` 里）

```python
_PAD_BS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
pi = bisect.bisect_left(_PAD_BS, B)
B_pad = _PAD_BS[pi] if pi < len(_PAD_BS) else B
```

把 B 向上取整到 `_PAD_BS` 中最近的 2^n，限制 JIT key 数量为最多 10 个。

**Padding 行的处理**：
- `accepted_steps[B..B_pad-1] = 0`，`state_idx[B..B_pad-1] = 0`（dummy slot）
- kernel 内部 loop_limit = 0，这些行不读 stash，把 h_0（dummy slot 原始值）写回 slot 0（无害）
- Stash buffer 已按 `max_tokens = pool_size × T_max` 分配，直接 view 成 `[B_pad, T, ...]` 零额外分配

```python
# Zero-copy view，无 GPU kernel 启动
k_bat = stash["k"][0, :actual_seq_len_pad].view(B_pad, T, H, K)
```

#### 解法 2: Startup prewarm（在 `model_runner.py` 里）

新增 `maybe_init_gdn_recovery_prewarm()`，在 `__init__` 阶段（CUDA graph 初始化之后）执行：

```python
for B_pad in [1, 2, 4, ..., 512]:
    for T in range(1, T_max + 1):
        gated_delta_rule_mtp(dummy_k, dummy_v, ..., disable_output=True)
```

总共 `10 × T_max` 个 JIT variant，全部在第一个请求到达前编译好。估计总时间 `~10 × T_max × 37s`（进度信息通过 logger.info 打印）。

**调用链**（`ModelRunner.__init__`）：
```python
self.init_cublas()
self.init_attention_backend()
self.kernel_warmup()
self._pre_initialize_flashinfer_allreduce_workspace()
self.init_device_graphs()
self.maybe_init_gdn_recovery_prewarm()  # ← 新增
```

#### FlashInfer kernel 调用（在 `_run_recovery()` 里）

```python
use_fi_recovery = (
    decode_kernel.__class__.__name__ == "FlashInferGDNKernel"
    and getattr(decode_kernel, "use_state_pool", False)
)
```

满足条件（SM100 + bf16 state pool）时，调用 FlashInfer API：

```python
from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

gated_delta_rule_mtp(
    A_log=stash["A_log"].detach().float(),
    a=stash["a"][...].view(B_pad, T, H),
    ...
    accepted_steps=acc_steps_pad,   # [B] int32，GPU 端 per-seq 控制步数
    disable_state_update=False,
    disable_output=True,            # 只更新 state，不生成 output
)
```

**与 Triton 路径的差异**：
- Triton：每层单独 launch（45 个 kernel）
- FlashInfer：同样每层一个 kernel，但 kernel 本身是 SM100 优化的 CuTe kernel

---

### Commit 5: `662840cbce` — Fix rebase: remove bogus kernel_warmup/init_device_graphs calls

**作者**: Wenjing Lu | **日期**: Jun 25, 2026

**纯清理**。rebase 时从旧的 pre-rebase main 误带入了 3 个调用（`kernel_warmup`、`_pre_initialize_flashinfer_allreduce_workspace`、`init_device_graphs`），这些在 origin/main 里已经存在于别处，本 branch 只应新增 `maybe_init_gdn_recovery_prewarm()`。删除重复项，避免双重初始化。

---

## 两个 Branch 对比

| 维度 | `gdn-recovery-overlap` | `review/gdn-recovery-flashinfer` |
|---|---|---|
| **基础功能** | `cache_mode=none` + Triton recovery kernel + side stream overlap | 同上，全部包含 |
| **Recovery kernel** | Triton `fused_sigmoid_gating_delta_rule_recover_final_state` | SM100 检测到 FlashInferGDNKernel 时切换 `gated_delta_rule_mtp`（PR #3502） |
| **JIT 问题** | 无处理（每个新 batch size 可能触发 37s 编译） | B-padding（限制 10 个 B 值）+ startup prewarm（预编译全部 variant） |
| **Stash 内存分配** | 同 overlap branch 的原始方式 | 零拷贝 view（无额外 cat/zero-fill GPU kernel） |
| **model_runner 改动** | 无 | 新增 `maybe_init_gdn_recovery_prewarm()` 调用链 |
| **目标硬件** | 通用（任何支持 GDN 的 GPU） | 主要针对 SM100（GB200）上 FlashInfer 路径；SM100 以下回退 Triton |
| **状态** | 已验证（bs512 GSM8K 0.978，accept length ~3.5） | 待验证（FlashInfer 路径的数值正确性尚未在 commit message 中确认） |
| **commit 数量** | 3 个 GDN commit | 5 个 GDN commit（3 + 2） |

---

## Qwen 3.5 关键 Spec

这些 branch 的实现依赖以下 Qwen 3.5 模型特性：

| 参数 | 说明 |
|---|---|
| `USE_QK_L2NORM_IN_KERNEL=True` | Qwen 3.5 的 GDN 层对 key 做 L2 normalization |
| `IS_KDA=False` | 非 KDA（kernel dimension aggregation）变体；a 和 dt_bias 是 per-head 标量，不是 per-K 向量 |
| `softplus_beta=1.0, softplus_threshold=20.0` | Qwen 3.5 hardcoded 的 softplus 参数 |
| 45 GDN layers | prewarm 和 recovery loop 都按 45 层推算 |
| `H` (num_heads), `K` (head_k_dim), `V` (head_v_dim) | 从 `state_source.shape[1:4]` 动态读取，具体值由模型权重决定 |
| `ssm_dtype = bf16`（SM100） | FlashInfer 路径（`gdn_decode_bf16_state`）要求 bf16 state pool |
| `draft_token_num = T_max` | speculative decode draft 步数，决定 prewarm 的 T 维度范围 |

---

## 总结

**共同出发点**：GDN + MTP（speculative decoding）场景下，原始 `full` cache mode 的 `intermediate_ssm` 缓存太大（`[L, pool+1, T, H, K, V]` GB 级显存）。两个 branch 都通过 `cache_mode=none` 把这块显存省掉，代价是接受后需要重算 `h_{accepted}`。

**`gdn-recovery-overlap`**：完整实现了 `none` 路径，并用 side stream overlap 把重算成本隐藏到下一步 draft phase 里。是可用的基础版本。

**`review/gdn-recovery-flashinfer`**：在上面基础上，针对 GB200（SM100）做了两件事：①把 recovery kernel 换成 FlashInfer CuTe kernel（更优的 SM100 利用率）；②用 B-padding + startup prewarm 彻底消除生产中的 JIT 冷启动问题。是面向 GB200 生产部署的完整版本，但 FlashInfer recovery 路径的数值正确性需要进一步验证。
