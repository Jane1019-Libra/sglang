# GDN MTP Profile 性能分析：FlashInfer vs Triton Recovery

> **结论一句话**：差距完全来自 `draft_extend` 阶段前的一个 ~1-2ms GPU idle gap。
> FlashInfer 的 recovery kernel 没有正确 overlap，导致主 stream 在等待 side stream 排空。

---

## 一、每个阶段的时长对比

每个 MTP iteration 分三段：`draft → TARGET_VERIFY → draft_extend`

| 阶段 | FlashInfer | Triton | 差值 |
|---|---|---|---|
| draft | 2,828 µs | 2,847 µs | ≈0，**无差异** |
| TARGET_VERIFY | 17,341 µs | 17,359 µs | ≈0，**无差异** |
| **draft_extend** | **3,188 µs** | **2,180 µs** | **+1,008 µs ← 主要来源** |
| 两步之间 gap | ~2,366 µs | ~1,824 µs | +542 µs |

**问题不在 TARGET_VERIFY，完全在 `draft_extend` 阶段（以及紧接其前的 gap）。**

---

## 二、Perfetto 截图对比

### 图一：FlashInfer（26ms/iteration）

```
时间轴（ns）: 942,000,000 → 949,000,000
```

```
主 stream：
  [step[TARGET_VERIFY bs=64]] [nccl][void...]
                                               ← ~1.5ms 空白 idle gap →
                                                                        [draft_extend][bmm...][nccl...]

CPU scheduler：
                             [scheduler.run_batch (黄绿色)]
                              ↑ 这里 CPU 在跑 _wait_recovery_if_pending，GPU 主 stream 空转
```

**关键发现**：TARGET_VERIFY 结束后，GPU 主 stream 上有约 **1.5ms 的完全空白**，然后 draft_extend 才开始。这段空白就是 `_wait_recovery_if_pending()` 在 CPU 端阻塞、等待 FlashInfer recovery 的 side stream 排空的时间。

---

### 图二：Triton（24ms/iteration）

```
时间轴（ns）: 948,000,000 → 951,000,000
```

```
主 stream（上方行）：
  [step[TARGET_VERIFY bs=64]][void...] → [draft_extend][bmm_Bfloat16...][bmm_Bf...][nccl...][void...]
                                          ↑ TV 结束后立即开始，无 gap

Side stream（下方行）：
  [f][f][f][f][f][f][fused...][f][f][f][f][f][f][f][fused_...][f][f][f][f]
  ↑ 这些是 fused_sigmoid_gating_delta_rule_recover_final_state_kernel（recovery），
    与 draft_extend 并行跑在 side stream 上，完全 overlap
```

**关键发现**：TARGET_VERIFY 结束后**立即**开始 draft_extend，没有任何 gap。同时在 side stream 上可以看到大量 `fused_sigmoid...` recovery kernel 在并行执行（下方行的 `f f f f fused...`），这证明 Triton 的 recovery 完全 overlap 进了 draft_extend。

---

## 三、Stream 活动对比

| | FlashInfer | Triton |
|---|---|---|
| draft_extend 期间活跃的 stream | `[23, 5951]` | `[23, 75, 79, 5951]` |
| recovery side stream | **不在 draft_extend 期间** | stream 75/79 **与 draft_extend 并行** |
| draft_extend 内 kernel 数量 | 67 个 | 123 个（含 recovery） |
| draft_extend GPU wall time | 3,188 µs | 2,180 µs |

Triton 的 recovery（810 次调用 × 40.6 µs ≈ 1.4ms GPU 工作量）在 `draft_extend` **期间**免费并行完成。

FlashInfer 的 `draft_extend` 里只有 67 个 kernel 却用了 3,188 µs——GPU 在 **stall**，不是在计算。

---

## 四、根本原因

**FlashInfer 的 recovery 没有正确 overlap。**

正确的执行顺序应该是：

```
Step N:
  draft → TARGET_VERIFY → [launch recovery on side_stream] → draft_extend
                                                               ↑ recovery 在这里并行跑

Step N+1:
  init_forward_metadata → _wait_recovery_if_pending()  ← 等待上一步 recovery 完成
  → draft → TARGET_VERIFY → ...
```

**Triton branch 是这样工作的**：recovery 在 draft_extend 期间在 side stream 上并行完成，`_wait_recovery_if_pending` 在下一步 draft 前 fire，等待时间极短（recovery 早已完成）。

**FlashInfer branch 的问题**：`kernel_cutlass_gdn_wide_vec_kernel` 调用了 **1,620 次**（Triton 只有 810 次，恰好 2×）。推测是 FlashInfer 的 CuTe DSL recovery kernel 在 side stream 上运行时间更长（尽管单次 kernel 更快，但 launch overhead 或 setup 开销更大），导致在 step N+1 的 draft_extend 开始前 recovery 还没完成，`_wait_recovery_if_pending` 在此时触发，造成约 **1.5ms 的 GPU idle gap**。

---

## 五、FlashInfer 的 recovery kernel 明明更快，为什么反而更慢？

| | FlashInfer recovery | Triton recovery |
|---|---|---|
| 单次 kernel 时长 | 更快（CuTe DSL 优化） | 40.6 µs/次 |
| 总调用次数 | 1,620 次（2×） | 810 次 |
| 是否 overlap 进 draft_extend | **否** | **是** |
| 对 critical path 的影响 | 造成 1.5ms stall | 无（完全并行） |

单次 kernel 快，但 overlap 失效 → 反而更慢。

---

## 六、精确的根本原因（数据分析后更新）

通过脚本精确量 TV→DE 之间 gap 内的 kernel 活动：

| | FlashInfer | Triton |
|---|---|---|
| TV→DE gap | **2,288 µs** | **553 µs** |
| stream 23（post-TV scatter 等） | 23 kernels, 775 µs | 23 kernels, 771 µs |
| stream 79（recovery side stream）| **45 kernels, 897 µs（全部在 gap 内）** | **2 kernels, 69 µs（其余 43 个在 DE 期间）** |
| 无 GPU 活动的 CPU 空档 | **~1,400 µs** | ~0 µs |

**Triton 的 recovery 和 draft_extend 真正 overlap 了**：CPU 快速提交完 recovery kernels，draft_extend 立刻开始，剩余 43 个 recovery kernels 在 DE 期间并行跑完。

**FlashInfer 的 recovery 没有 overlap**：45 个 recovery kernels 全部在 gap 内跑完，draft_extend 才开始。CPU 花了 ~1,400 µs 在 Python dispatch overhead 上，期间 GPU 主 stream 空转。

---

## 七、真正的根本原因：Python CPU dispatch overhead

**不是 GPU sync，不是 stream 配置错误，是 Python 层的 per-layer 开销。**

FlashInfer `_run_recovery()` 在 45 层的循环里每层都做：

```python
for layer_id, stash in stash_per_layer.items():  # 45 次
    layer_ssm_states = pool.mamba2_layer_cache(layer_id).temporal  # dict lookup
    k_bat = stash["k"][0, :actual_seq_len_pad].view(B_pad, T, H, K)  # slice + view
    gated_delta_rule_mtp(
        A_log=stash["A_log"].detach().float(),   # detach + dtype 转换（可能触发 GPU kernel）
        a=stash["a"][:actual_seq_len_pad].view(B_pad, T, ...),   # slice + view
        dt_bias=stash["dt_bias"].detach(),        # detach
        q=k_bat, k=k_bat,
        v=stash["v"][0, :actual_seq_len_pad].view(B_pad, T, ...),  # slice + view
        b=stash["b"][:actual_seq_len_pad].view(B_pad, T, ...),     # slice + view
        # ... 共 15 个参数
    )
```

这些 Python 操作（`.detach()`、`.float()`、`slice`、`.view()`）在 45 层 × 15 参数的循环下累积成 ~1.4ms 的 CPU overhead，期间 Python GIL 无法让位给 draft_extend 的 kernel 提交。

**PR #3502 (`gated_delta_rule_mtp`) 不支持 multi-layer batching**（`initial_state` 是 `[pool_size, HV, V, K]`，无 layers 维度），所以无法合并成单次调用。

---

## 八、实施的 Fix 及过程中发现的 Crash Bug

### 8.1 性能 Fix（已实施）

**目标**：消除 `_run_recovery()` 闭包内 45 层循环的 Python CPU overhead（~1.4ms），让 GPU 主 stream 更快进入 draft_extend。

**修改文件 1：`gdn_backend.py`**

在 stash 分配时预存 `A_log_f32`，避免每个 step 的每层都做 `.detach().float()`：

```python
stash_entry = {
    "k": torch.empty(...),
    ...
    "A_log": layer.A_log,
    "A_log_f32": layer.A_log.detach().float(),  # 只在分配时做一次
}
```

**修改文件 2：`hybrid_linear_attn_backend.py`**

把所有 Python 准备工作（`bisect`、`B_pad` 计算、45 层的 `view/slice`、dict lookup）全部移到 `_run_recovery()` 闭包外，提前构建 `_fi_layer_args` 列表：

```python
# 闭包外：一次性构建，所有 CPU 工作在此完成
_fi_layer_args = [
    (
        pool.mamba2_layer_cache(layer_id).temporal,
        stash["A_log_f32"],
        stash["a"][:actual_seq_len_pad].view(B_pad, T, ...),
        stash["dt_bias"],
        stash["k"][0, :actual_seq_len_pad].view(B_pad, T, ...),
        stash["v"][0, :actual_seq_len_pad].view(B_pad, T, ...),
        stash["b"][:actual_seq_len_pad].view(B_pad, T, ...),
    )
    for layer_id, stash in stash_per_layer.items()
]

def _run_recovery():
    # 闭包内只剩 45 次裸 kernel launch
    for (...) in _fi_layer_args:
        gated_delta_rule_mtp(...)
```

**预期效果**：TV→DE gap 从 ~2,288µs 降至接近 Triton 的 ~553µs。

---

### 8.2 Crash Bug（性能 Fix 引入，已修复）

**现象**：性能 Fix 上线后，server 在推理时 crash，报 `cudaErrorIllegalAddress`。

**根本原因：`acc_steps_pad` / `state_idx_pad` 缺少 `record_stream`**

当实际 batch size `B` 不是 `_PAD_BS = [1,2,4,8,...,128]` 中的精确值时（例如 B=3, 7, 63 等），代码会通过 `torch.cat` 创建两个**新的** GPU tensor：

```python
n_extra = B_pad - B  # > 0
_z = accepted_steps_i32.new_zeros(n_extra)
acc_steps_pad = torch.cat([accepted_steps_i32, _z])   # 全新分配
state_idx_pad = torch.cat([state_idx_i32, _z])         # 全新分配
```

这两个 tensor 被 side stream 上的 `gated_delta_rule_mtp` 内核读取。但由于 CUDA kernel launch 是**异步的**，Python 侧 `_run_recovery()` 返回后，函数继续执行并最终 return，此时 `acc_steps_pad` 和 `state_idx_pad` 的 Python 引用计数归零，CUDA caching allocator 认为这块 GPU 内存可以在 main stream 上复用。然而 side stream 的 kernel 还在读这块内存 → **非法内存访问**。

`record_stream(stream)` 的作用：告知 allocator "此 tensor 的内存在指定 stream 跑完之前不得回收"。原代码只对 `state_idx_i32` / `accepted_steps_i32`（原始 tensor）调了 `record_stream`，但对 `torch.cat` 产生的新 tensor 漏掉了。

**重要说明：此 bug 不在 CUDA graph capture 阶段触发。** Capture 阶段走的是：
```python
if torch.cuda.is_current_stream_capturing():
    _run_recovery()   # 同步在 capture stream 上跑，无 side stream
    return
```
不涉及 side stream，无竞争条件。Bug 只在**实际推理**中 B 不是精确 power-of-2 时触发。

**修复（`hybrid_linear_attn_backend.py`末尾加两行）**：

```python
state_idx_i32.record_stream(self._recovery_stream)
accepted_steps_i32.record_stream(self._recovery_stream)
# n_extra > 0 时 torch.cat 创建了新 tensor，必须单独 record_stream
if use_fi_recovery and n_extra > 0:
    acc_steps_pad.record_stream(self._recovery_stream)
    state_idx_pad.record_stream(self._recovery_stream)
```

当 `n_extra == 0` 时，`acc_steps_pad is accepted_steps_i32`（同一个 Python 对象），条件不成立，无重复操作。

---

## 九、当前代码状态

| 文件 | 修改内容 | 状态 |
|---|---|---|
| `gdn_backend.py` | 预存 `A_log_f32`，避免每步 dtype 转换 | ✅ 已实施 |
| `hybrid_linear_attn_backend.py` | 提前构建 `_fi_layer_args`（性能）+ `record_stream`（crash fix） | ✅ 已实施 |
| `model_runner.py` | `cache_mode=full` 时跳过 recovery prewarm（节省 94s 启动时间） | ✅ 已实施 |

待验证：重启 server（`--gdn-mtp-cache-mode none --linear-attn-decode-backend flashinfer`）后运行 benchmark，确认 TV→DE gap 是否缩短，且不再 crash。

---

## 十、Profile 文件位置

| Branch | 文件路径 |
|---|---|
| FlashInfer | `/scratch/fsw/portfolios/coreai/users/wenjingl/profiles/1782610886.7372391/mtp_bs64_recover_flashinfer*-TP-0.trace.json.gz` |
| Triton | `/scratch/fsw/portfolios/coreai/users/wenjingl/profiles/1782612714.5285528/mtp_bs64_recover_triton*-TP-0.trace.json.gz` |

用 [ui.perfetto.dev](https://ui.perfetto.dev) 打开，直接拖入 `.gz` 文件即可查看。
