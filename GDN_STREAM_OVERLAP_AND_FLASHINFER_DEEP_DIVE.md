# GDN MTP Recovery — 逐行精读：Side Stream Overlap & FlashInfer JIT Stability

本文针对两个 commit 做**一行一行**的深度解析，并逐一回答「这么做有没有先例 / 被允许吗」。

---

## 一、Commit 3：Side Stream Overlap（`36409229d0`）

**修改文件**: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

---

### 1.1 新增三个字段

```python
# hybrid_linear_attn_backend.py — HybridLinearAttnBackend.__init__()
self._recovery_stream: Optional[torch.cuda.Stream] = None
self._recovery_event: Optional[torch.cuda.Event] = None
self._recovery_event_pending: bool = False
```

**为什么这样设计**：

- `_recovery_stream`：专用 CUDA side stream，专门用于跑 recovery kernels。
  - 为什么 lazy init（`= None`）而不是立即 `torch.cuda.Stream()`？
    因为 `HybridLinearAttnBackend.__init__` 在 CUDA graph 初始化阶段很早被调用，
    此时可能还没有确定会用 `cache_mode=none`。Lazy init 避免无谓的 stream 对象开销，
    且 stream 创建本身有一次 CUDA context 的驱动层调用，推迟到真正需要时更干净。

- `_recovery_event`：用于 stream 同步。
  - `torch.cuda.Event()` 是一个 GPU timeline 上的时间戳，可以被另一个 stream wait。
  - 这里用 Event 而不是直接 `main_stream.wait_stream(recovery_stream)` 的原因：
    `wait_stream` 是全量等待（等对方当前 queue 全空），
    `wait_event` 只等到 event 被 record 的那个时刻之前的工作完成，更精确。

- `_recovery_event_pending`：一个 bool flag，标记「side stream 上有未完成的 recovery 工作」。
  - 为什么需要这个 flag？因为 event 对象一旦创建就可以被无限 wait，
    即使对应工作早已完成。没有 flag 就无法区分「event 已经完成了但我还没 wait」
    和「根本没有 recovery 工作」。

---

### 1.2 `_wait_recovery_if_pending()`

```python
def _wait_recovery_if_pending(self):
    if self._recovery_event_pending:
        torch.cuda.current_stream().wait_event(self._recovery_event)
        self._recovery_event_pending = False
```

**逐行解释**：

```python
if self._recovery_event_pending:
```
- 只在上一步真的把 recovery kick 到 side stream 时才 wait。
  如果 `cache_mode=full` 或者上一步是 prefill（没有 recovery），flag 是 False，直接跳过。

```python
torch.cuda.current_stream().wait_event(self._recovery_event)
```
- `current_stream` 是主 stream（forward 计算 stream）。
- `wait_event` 在 GPU timeline 上插入一个依赖：主 stream 上后续的 GPU ops
  必须等到 `_recovery_event` 被 record 时对应的 side stream work 完成后才能执行。
- 这是纯 GPU-side 的同步，**CPU 不会 block**。
- 具体到这里：下一次 target forward 读 SSM pool 之前，
  保证 side stream 已经把 `h_{accepted}` 写进了 temporal pool。

```python
self._recovery_event_pending = False
```
- 清 flag，防止下一次 init_forward_metadata 重复 wait 一个已经完成的 event。

**这个函数在哪里被调用**：

```python
# 1. init_forward_metadata()（eager path）
def init_forward_metadata(self, forward_batch: ForwardBatch):
    if forward_batch.forward_mode.is_draft_extend_v2():
        self.full_attn_backend.init_forward_metadata(forward_batch)
        return
    self._wait_recovery_if_pending()   # ← 每次新一步开始，先等上一步 recovery 完
    for attn_backend in self.attn_backend_list:
        attn_backend.init_forward_metadata(forward_batch)

# 2. init_forward_metadata_replay_cuda_graph()（CUDA graph replay path）
def init_forward_metadata_replay_cuda_graph(self, bs, ...):
    if not forward_mode.is_draft_extend_v2():
        self._wait_recovery_if_pending()   # ← 同上，replay 前也要 join
    for attn_backend in self.attn_backend_list:
        attn_backend.init_forward_metadata_replay_cuda_graph(...)
```

`is_draft_extend_v2()` 跳过的原因：draft 阶段不读 target SSM pool，
没有 RAW hazard，不需要 join。

---

### 1.3 `_no_cache_mtp_recompute()` — 核心调度逻辑

```python
def _run_recovery():
    for layer_id, stash in stash_per_layer.items():
        layer_cache = pool.mamba2_layer_cache(layer_id)
        layer_ssm_states = layer_cache.temporal
        fused_sigmoid_gating_delta_rule_recover_final_state(
            A_log=stash["A_log"],
            a=stash["a"][:actual_seq_len],
            ...
            initial_state_source=layer_ssm_states,
            initial_state_indices=state_idx_i32,
            accepted_steps=accepted_steps_i32,
            cache_steps=cache_steps,
        )
```

`_run_recovery` 被提成 closure（而不是直接写两遍）的原因：
capture path 和 eager path 都要跑同一段 kernel loop，closure 避免代码重复。

```python
if torch.cuda.is_current_stream_capturing():
    _run_recovery()
    return
```

**为什么 CUDA graph capture 必须 inline 跑**：

CUDA graph capture（`with graph.capture():`）录制的是当前 capture stream 上的 GPU ops。
如果此时 fork 到 side stream，那些 ops 就不会被录进 graph，replay 时 recovery 就丢了。
所以 capture 期间必须在当前（capture）stream 上同步执行。

`is_current_stream_capturing()` 是整个 codebase 里判断是否在 capture 的标准方法，
在 `lora/trtllm_lora_temp/moe_overlap.py`、`distributed/parallel_state.py` 等多处使用。

```python
if self._recovery_stream is None:
    self._recovery_stream = torch.cuda.Stream()
    self._recovery_event = torch.cuda.Event()
```

第一次 eager recovery 时才创建 stream 和 event（前面说的 lazy init）。

```python
self._recovery_stream.wait_stream(torch.cuda.current_stream())
```

**这行至关重要**：在 side stream 开始跑 recovery 之前，让它先等主 stream 追上来。

为什么？因为 stash（`k/v/a/b`）是在本步 target verify 的主 stream 上写的。
如果 side stream 不等主 stream，它可能在 stash 写完之前就开始读，产生 RAW hazard。

`wait_stream` 在 GPU timeline 插入：`side_stream` 等到主 stream 当前 queue 全空之后才开始。

```python
with torch.cuda.stream(self._recovery_stream):
    _run_recovery()
```

把 recovery kernels 提交到 side stream 的 queue 里。
`with torch.cuda.stream(s)` 只是切换 PyTorch 的「当前 stream」上下文，
CPU 立即返回，GPU 在 side stream 上异步执行。

```python
self._recovery_event.record(self._recovery_stream)
```

在 side stream 的 queue 末尾插一个 event stamp。
当 recovery kernels 全部执行完毕，event 就被 GPU 标记为「已完成」。
之后主 stream 调用 `wait_event` 时，如果 event 已完成，GPU 直接继续；
否则 GPU-side 阻塞直到完成（CPU 不 block）。

```python
state_idx_i32.record_stream(self._recovery_stream)
accepted_steps_i32.record_stream(self._recovery_stream)
self._recovery_event_pending = True
```

**`record_stream` 的作用**（关键易错点）：

`state_idx_i32` 和 `accepted_steps_i32` 是在 `_no_cache_mtp_recompute` 里新分配的：

```python
state_idx_i32 = state_indices_tensor.to(torch.int32).contiguous()
accepted_steps_i32 = accepted_steps.to(torch.int32).contiguous()
```

`.to()` 和 `.contiguous()` 可能产生新的 tensor 分配，这些分配是在主 stream 上的。
PyTorch 的 caching allocator 追踪每个 tensor 「最后在哪个 stream 上被使用」。
当 `_no_cache_mtp_recompute` 函数返回时，这两个局部变量引用计数降为 0，
allocator **可能立即把底层内存回收复用**——即使 side stream 还没读完。

`record_stream(recovery_stream)` 告诉 allocator：
「这块内存也在 recovery_stream 上被使用，在 recovery_stream 追过这个点之前，不要回收它。」

这是 PyTorch cross-stream memory safety 的标准做法。

**codebase 先例**：
- `managers/cache_controller.py:717` — `host_indices.record_stream(self.write_stream)`
- `managers/hisparse_coordinator.py:252` — 同样的 D2H/side-stream pin 模式
- `models/deepseek_v2.py:1220` — `shared_output.record_stream(self.alt_stream)`
- `managers/utils.py:34` — 有专门的注释解释 record_stream 的必要性

---

### 1.4 `init_forward_metadata_replay_cuda_graph` 为什么被重写

原来 `HybridLinearAttnBackend` 没有重写这个方法（用 base class 的）。
现在需要在 replay 路径也插入 `_wait_recovery_if_pending()`。

注释里特别说明了 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` 的情况：

```
NOTE: under SGLANG_ENABLE_OVERLAP_PLAN_STREAM this runs on plan_stream, not the
replay (fwd) stream; correctness relies on the worker doing
fwd_stream.wait_stream(plan_stream) before the replay (eagle_worker_v2.verify),
which transitively propagates this wait.
```

即：plan stream 上的 `wait_event` → `fwd_stream.wait_stream(plan_stream)` → fwd stream 等 plan stream → 传递地也等了 recovery event。

---

### 1.5 Side Stream Overlap 在 codebase 里的先例

以下是 origin/main 中使用相同模式的地方，说明这是被认可的设计：

| 文件 | 用途 |
|---|---|
| `models/deepseek_v2.py:1216-1222` | shared expert 在 alt_stream 上并行 |
| `models/qwen3_next.py:393-397, 750-756` | embedding 加载 overlap |
| `models/llama4.py:175,184` | cross-layer prefetch overlap |
| `models/grok.py:613-617` | MoE expert overlap |
| `models/bailing_moe_linear.py:350,358` | MoE overlap |
| `models/nemotron_h.py:286,303` | 同类 SSM model 的 overlap |
| `utils/offloader.py:309` | KV cache offload 到 host |
| `utils/multi_stream_utils.py` | 封装成 `maybe_execute_in_parallel` 工具函数 |
| `managers/cache_controller.py:717,719` | HiCache write stream + record_stream |

**结论**：Side stream overlap + record_stream + wait_event 是 sglang 的标准并行化手段，不存在任何被禁止的迹象。

---

## 二、Commit 4：B-padding + FlashInfer Prewarm（`cc0a008155`）

**修改文件**：
- `hybrid_linear_attn_backend.py` — `_run_recovery()` 里新增 FlashInfer 路径
- `model_runner.py` — 新增 `maybe_init_gdn_recovery_prewarm()`

---

### 2.1 FlashInfer kernel 检测

```python
# hybrid_linear_attn_backend.py — _no_cache_mtp_recompute()
dispatcher = getattr(self.linear_attn_backend, "kernel_dispatcher", None)
decode_kernel = getattr(dispatcher, "decode_kernel", None)
use_fi_recovery = (
    decode_kernel is not None
    and decode_kernel.__class__.__name__ == "FlashInferGDNKernel"
    and getattr(decode_kernel, "use_state_pool", False)
)
```

**为什么用 `getattr` + 字符串类名检测，而不是 `isinstance`**：

`FlashInferGDNKernel` 只有在安装了支持 SM100 的 FlashInfer 时才存在。
直接 `import` 会在非 SM100 环境抛 ImportError。
用 `getattr(..., None)` + `__class__.__name__` 检测是 duck-typing 风格，
不需要 import，非 SM100 环境自然 fall back 到 Triton 路径。

`use_state_pool=True` 的检查：FlashInfer 的 `gated_delta_rule_mtp` 需要
state pool 格式（`[pool_size+1, HV, V, K]` 连续内存），
如果 GDN 配置不满足这个条件（比如非 bf16），不能用这个 kernel。

---

### 2.2 B-padding 逻辑

```python
_PAD_BS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
pi = bisect.bisect_left(_PAD_BS, B)
B_pad = _PAD_BS[pi] if pi < len(_PAD_BS) else B
```

**为什么要 padding**：

FlashInfer 的 `gated_delta_rule_mtp` 使用 **CuTe DSL JIT 编译**。
kernel 的模板参数包含 `(B, T)`，每个不同的 `(B, T)` 组合第一次运行时触发约 **37 秒** 的重编译。

如果不 padding，sglang 的 scheduler 可能产生从 1 到 512 的任意 batch size，
每个新 bs 都会触发一次 37s 的编译 stall，生产不可接受。

`bisect.bisect_left(_PAD_BS, B)` 找到「第一个 ≥ B 的位置」，等同于 ceil-to-next-power-of-2：
- B=100 → B_pad=128
- B=128 → B_pad=128（精确命中）
- B=129 → B_pad=256

这把 JIT key 从无界集合缩减到最多 10 个，配合 prewarm 就能全部预编译完。

**codebase 先例**：

`model_executor/runner/base_cuda_graph_runner.py:140-149` 有完全相同的逻辑：

```python
@staticmethod
def _pad_to_bucket(raw_size: int, buckets: Sequence[int]) -> int:
    """Return the smallest buckets[i] >= raw_size."""
    index = bisect.bisect_left(buckets, raw_size)
    return buckets[index]
```

CUDA graph runner 把 batch size 向上取整到预捕获的 bucket，
原因完全一致：限制 capture 数量。B-padding 是 sglang 里处理「有限 JIT key」的标准做法。

---

### 2.3 Padding 行的处理

```python
n_extra = B_pad - B
if n_extra > 0:
    _z = accepted_steps_i32.new_zeros(n_extra)
    acc_steps_pad = torch.cat([accepted_steps_i32, _z])
    state_idx_pad = torch.cat([state_idx_i32, _z])
    actual_seq_len_pad = B_pad * T
else:
    acc_steps_pad = accepted_steps_i32
    state_idx_pad = state_idx_i32
    actual_seq_len_pad = actual_seq_len
```

**`accepted_steps=0` 的语义**：

recovery kernel 内部的 loop 是 `for step in range(0, T): active = step <= accepted_step`。
当 `accepted_step=0`：step=0 时 `active=True`，step 1..T-1 时 `active=False`。
实际上执行 1 步的「空更新」（k=0, v=0），写回 h_0（初始值）。

**`state_idx=0` 的语义**：

slot 0 是整个 MambaPool 的 reserved dummy slot（`spec_state_size + 1` 里的 `+1`），
实际请求永远不会占用 slot 0。Padding 行写 slot 0，不会污染任何真实请求的状态。

---

### 2.4 Zero-copy stash view

```python
k_bat = stash["k"][0, :actual_seq_len_pad].view(
    B_pad, T, stash["k"].shape[2], stash["k"].shape[3]
)
```

**理解这行的前提**：

stash 在 `gdn_backend.py` 里是这样分配的：

```python
stash_entry = {
    "k": torch.empty(
        (key.shape[0], max_tokens, *key.shape[2:]),   # [1, pool_size*T, H, K]
        dtype=key.dtype, device=key.device,
    ),
    ...
}
```

`max_tokens = pool_size * draft_token_num`，远大于 `B_pad * T`。
存入时：`stash_entry["k"][:, :actual_seq_len].copy_(key)`，
数据是连续的 `[1, B*T, H, K]` layout（dim0=1 是历史遗留的「leading 1」）。

```
stash["k"][0, :actual_seq_len_pad]  →  形状 [B_pad*T, H, K]
.view(B_pad, T, H, K)               →  reshape 成 [B_pad, T, H, K]
```

这是一个**零拷贝 view**（无 GPU kernel 启动），只改变 tensor 的 stride/shape 元数据。
Padding 行 `B..B_pad-1` 的数据是 stash buffer 里的 stale 值，
但因为 `acc_steps_pad[padding_rows] = 0`，kernel 不会读这些行，写回也只写 slot 0。

**为什么不用 `torch.cat([real_data, zeros])`**：

`torch.cat` 会 launch 一个 GPU kernel（内存拷贝），在 side stream 上有额外 overhead。
View 是纯 CPU metadata 操作，零 GPU overhead。
这里的 「zero-copy」 是性能优化，同时也是正确性保证（stale 数据不会被实际使用）。

---

### 2.5 `maybe_init_gdn_recovery_prewarm()`

**调用位置**（`review/gdn-recovery-flashinfer` 当前状态）：

```python
# model_runner.py — init_attention_backends()
if self.device == "cuda" or self.device == "musa":
    self.init_cublas()
    self.init_attention_backend()
    self.maybe_init_gdn_recovery_prewarm()   # ← 在 CUDA graph capture 之前
```

**为什么必须在 `init_cuda_graphs()` 之前**：

JIT 编译过程本身会在 GPU 上分配临时内存、触发 cubin 编译。
如果放在 CUDA graph capture 期间，这些 side-effect 会被录进 graph，replay 时重复触发，导致错误。

**为什么必须在 `init_attention_backend()` 之后**：

`maybe_init_gdn_recovery_prewarm` 需要 `self.attn_backend` 存在，
并且 `attn_backend.linear_attn_backend.kernel_dispatcher.decode_kernel` 存在，
这些都在 `init_attention_backend()` 里创建。

**Early exit guards 逐一解释**：

```python
if self.device != "cuda":
    return
```
FlashInfer SM100 kernel 只在 CUDA 上存在，NPU/ROCm 跳过。

```python
if self.is_draft_worker:
    return
```
Draft model 不做 MTP state recovery（recovery 是 target worker 的职责）。

```python
T_max = self.server_args.speculative_num_draft_tokens
if T_max is None:
    return
```
非 speculative decoding 模式，没有 draft tokens，不需要 recovery。

```python
if not isinstance(self.attn_backend, HybridLinearAttnBackend):
    return
```
只有 hybrid（GDN）模型才有 SSM recovery，纯 attention 模型跳过。

```python
if (decode_kernel is None or
    decode_kernel.__class__.__name__ != "FlashInferGDNKernel" or
    not getattr(decode_kernel, "use_state_pool", False)):
    return
```
非 FlashInfer 路径（Triton 路径）不需要 prewarm，因为 Triton 没有 JIT 冷启动问题。

**Prewarm 主体**：

```python
for B_pad in _PAD_BS:                       # 10 个 B 值
    for T in range(1, T_max + 1):           # 1..T_max
        dummy_idx = torch.zeros(B_pad, dtype=torch.int32, device=dev)
        dummy_acc = torch.zeros(B_pad, dtype=torch.int32, device=dev)
        dummy_k = torch.zeros(B_pad, T, H, K, dtype=ssm_dtype, device=dev)
        ...
        gated_delta_rule_mtp(
            ...
            initial_state_source=state_source,   # 真实的 state pool（shape 对，但只写 slot 0）
            initial_state_indices=dummy_idx,     # 全 0 → 全部写 slot 0
            output_state_indices=dummy_idx,
            accepted_steps=dummy_acc,            # 全 0 → 每行只跑 1 步，无害
            disable_output=True,
        )
```

`state_source = mamba_pool.mamba_cache.temporal[0]`：用第 0 层的 state pool，
形状和 dtype 是真实的。JIT 编译只依赖 shape/dtype/sm 版本，用真实 shape 保证 prewarm 的 kernel 和推理时用的完全一致。

写 slot 0（reserved dummy）：harmless，slot 0 永远不被真实请求使用。

`torch.cuda.synchronize()` 在循环外：等所有编译完成后再继续，
这样 prewarm 结束后所有 `(B_pad, T)` 组合都已编译进 kernel cache。

**codebase 先例**：

- `model_runner.py:1265-1279` — `pre_warm_nccl`：服务启动时预热 NCCL communicator，
  完全相同的「startup prewarm」模式，甚至用了相同的 `time.perf_counter()` + logger.info 打印。
- `model_runner.py:837` — `maybe_init_ngram_embedding()`：条件性 lazy init，
  `maybe_*` 命名约定和 guard 结构与 `maybe_init_gdn_recovery_prewarm` 完全一致。

---

### 2.6 `inference_mode(False)` 用于 stash buffer 分配

```python
# gdn_backend.py — target verify，cache_mode=none 路径
if stash_entry is None or stash_entry["k"].shape[1] < max_tokens:
    with torch.inference_mode(False):
        stash_entry = {
            "k": torch.empty(...),
            "v": torch.empty(...),
            ...
        }
```

**为什么需要 `inference_mode(False)`**：

PyTorch 的 `inference_mode`（推理期间默认开启）会把新创建的 tensor 标记为 inference tensor。
Inference tensor 有一个限制：不能被 inplace 写入——而 stash 的使用方式正是：
```python
stash_entry["k"][:, :actual_seq_len].copy_(key)   # ← inplace 写
```

如果 stash buffer 是 inference tensor，这行 copy_ 会报错。
`inference_mode(False)` 把 buffer 创建成普通 tensor，跳过这个限制。

**codebase 先例**（完全相同的模式 + 相同的原因注释）：

`layers/moe/moe_runner/flashinfer_cutedsl.py:270-275`：

```python
# inference_mode(False) ensures the wrapper's pre-allocated CUDA-graph
# buffers are normal tensors.  This call typically happens inside
# _dummy_run which runs under inference_mode(); inference tensors cannot
# be inplace-updated during later CUDA graph capture (which runs outside
# inference_mode), so we must opt out here.
with torch.inference_mode(False):
    layer._cutedsl_wrapper = CuteDslMoEWrapper(...)
```

原因完全一致：在 inference_mode 下分配的 buffer 需要在后续被 inplace 更新。

---

## 三、总结：这些做法是否被 codebase 允许？

| 技术 | 这两个 commit 的用法 | Codebase 先例 | 结论 |
|---|---|---|---|
| Side stream overlap（`alt_stream.wait_stream` + `current_stream.wait_event`）| recovery kernels 在 `_recovery_stream` 上跑，主 stream wait_event | DeepSeek V2, Qwen3 Next, llama4, Grok, Bailing MoE, Nemotron-H | ✅ 标准模式 |
| `record_stream()` pin tensor lifetime | 保护 `state_idx_i32`/`accepted_steps_i32` 不被 allocator 提前回收 | cache_controller, hisparse_coordinator, deepseek_v2, cohere2_moe, managers/utils | ✅ 标准做法 |
| `is_current_stream_capturing()` guard | CUDA graph capture 期间 inline 跑，否则 fork 到 side stream | lora/moe_overlap（5处）, parallel_state, ernie45_moe_vl | ✅ 标准 guard |
| B-padding to power-of-2 with `bisect` | 限制 FlashInfer JIT key 为 10 个 | `base_cuda_graph_runner._pad_to_bucket()` — 完全相同逻辑 | ✅ 直接先例 |
| Startup prewarm（`maybe_init_*`）| `maybe_init_gdn_recovery_prewarm()` 在 `init_attention_backends()` 里调用 | `pre_warm_nccl`，`maybe_init_ngram_embedding()` | ✅ 直接先例 |
| `inference_mode(False)` 分配可 inplace 的 buffer | stash buffer 分配在 `inference_mode(False)` 上下文里 | `flashinfer_cutedsl.py ensure_cutedsl_wrapper` — 完全相同 + 相同注释 | ✅ 直接先例 |
| Zero-copy view（`stash["k"].view(B_pad, T, ...)`）| 避免 padding 时的 GPU copy kernel | 整个 codebase 到处都是 `tensor.view()` | ✅ 基础操作 |

**没有发现任何明确禁止这些模式的注释、CONTRIBUTING guide 或架构文档。**
相反，所有这些模式在 codebase 中都有直接的先例，且通常有相同的原因注释。

---

## 四、Eager Path vs CUDA Graph Replay Path：从零开始讲清楚

> 这一节专门给第一次接触 sglang / CUDA graph 的读者。先讲概念，再对应到 Qwen 3.5 服务器启动的真实流程。

---

### 4.1 为什么 GPU 推理有「CPU 开销」问题

GPU 跑 `torch.nn.Linear`、`attention` 这些算子，本质上是：
1. Python 调用 `torch.*` 函数
2. PyTorch 的 C++ 层把这个 op 翻译成一个 CUDA kernel 调用
3. 这个 kernel 被推入 GPU 的命令队列（CUDA stream）
4. GPU 异步执行，CPU 立刻继续

每一步都有几 μs 的 CPU overhead（Python 解释 + C++ dispatch + CUDA driver）。
一个 GDN 模型有几十层，每层有十几个 op，所以**每次 forward pass，CPU 要发出几百次 CUDA kernel 调用**。

当 batch size 很小（比如 bs=1, decode 阶段），GPU 跑一个 kernel 只需要几十 μs，但 CPU 提交下一个 kernel 也要几十 μs。这就造成 **CPU bound**：GPU 做完了在等 CPU 发下一条命令，大量 GPU 算力闲置。

---

### 4.2 CUDA Graph 的核心思想

CUDA Graph 解决 CPU overhead 的方法非常直接：

**Phase 1：Capture（录制，服务器启动时只做一次）**

```
CPU 进入 "capture" 模式
→ 正常执行一次 forward pass（dummy 输入）
→ GPU 驱动把所有收到的 kernel 调用录制进一个 "图"（DAG）
  [Linear kernel 1] → [Attention kernel] → [Linear kernel 2] → ...
→ 退出 capture 模式，图录制完毕
```

**Phase 2：Replay（回放，每次推理只需一条命令）**

```
CPU 更新输入 buffer（把新的 token id、seq len 等写进静态内存）
→ GPU 驱动执行 graph.replay()：一条命令，把整个图的所有 kernel 全部发射
→ GPU 执行完整 forward pass
→ CPU 从静态输出 buffer 读结果
```

从 CPU 的角度看：原来要发 300 次 kernel 调用，现在只发 1 次 `graph.replay()`。
CPU overhead 从几百次降到几次，吞吐量大幅提升。

**关键约束**：CUDA graph 要求输入的 **形状（shape）固定**。
- `graph.replay()` 会把之前录制好的 kernel 全部跑一遍
- 这些 kernel 的 grid/block 大小是 capture 时确定的，取决于 tensor 形状
- 如果 batch size 变了，kernel grid size 就变了，之前录制的 graph 就不适用了

这就是为什么 sglang 要预先 capture 多个 bucket size 的 graph：

```
服务器启动时 capture：
  bs=1  → graph_bs1
  bs=2  → graph_bs2
  bs=4  → graph_bs4
  bs=8  → graph_bs8
  bs=12 → graph_bs12
  bs=16 → graph_bs16
  bs=24 → graph_bs24
  ...（数十个 bucket）
```

推理时，如果当前 batch size = 6，就用 `graph_bs8`（最小的 ≥ 6 的 bucket），把 slot 7、8 填 0（padding）。

---

### 4.3 Eager Path 是什么，什么时候用

**Eager path**：就是「没有 CUDA graph」的普通 Python 执行路径。

每次 forward 都从头到尾走一遍 Python 代码，一条一条地发射 CUDA kernel。
- **优点**：完全灵活，可以处理任意形状、动态 Python 控制流
- **缺点**：有 CPU overhead

什么时候用 eager path：

| 场景 | 原因 |
|---|---|
| Prefill（新请求，填充 prompt）| 每个请求的 prompt 长度不同，形状变化大，graph 无法覆盖 |
| Decode batch size 超过最大 captured bucket | graph 没录这个 bs，只能 eager |
| CUDA graph 关闭（`--disable-cuda-graph`）| 全程 eager |
| 模型第一次跑（capture 之前的 warmup）| graph 还没录，只能 eager |

什么时候用 CUDA graph replay：

| 场景 | 原因 |
|---|---|
| Decode，bs ≤ max captured bs | 固定形状，可以 replay |
| Speculative decoding 的 target verify | verify 的 token 数 = bs × draft_token_num，形状固定 |

---

### 4.4 两条路径的代码入口在哪里

```python
# model_runner.py — forward_batch_generation() 的判断逻辑

can_run_graph = bool(
    forward_batch.forward_mode.is_cuda_graph()      # 这个 mode 适合 graph？
    and self.decode_cuda_graph_runner               # graph runner 存在？
    and self.decode_cuda_graph_runner.can_run_graph(forward_batch)  # bs 在范围内？
)

if can_run_graph:
    # ── CUDA Graph Replay Path ──
    ret = self.decode_cuda_graph_runner.execute(forward_batch)
    #   内部：load_batch()（更新静态 buffer）→ graph.replay()（一条命令跑完整 forward）
    return ModelRunnerOutput(logits_output=ret, ...)

else:
    # ── Eager Path ──
    ret = self.eager_runner.execute(forward_batch)
    #   内部：model.forward() → 逐层 Python 执行 → 逐条发射 GPU kernel
    return ...
```

注意：`attn_backend.init_forward_metadata()` 和 `attn_backend.init_forward_metadata_out_graph()` 就是在这两条路径里被调用的两个不同的元数据初始化入口：

| 路径 | 调用的元数据初始化 | 是否包含 `_wait_recovery_if_pending()` |
|---|---|---|
| Eager | `HybridLinearAttnBackend.init_forward_metadata()` | ✅ 是（HOOK 2） |
| CUDA Graph Replay | `HybridLinearAttnBackend.init_forward_metadata_out_graph()` | ❌ 否（本 PR 的已知缺口） |

> **注意**：`init_forward_metadata_replay_cuda_graph()` 这个方法是本 PR 新加的，按注释设计它应该在 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` 开启时由 plan stream 调用，从而通过 `fwd_stream.wait_stream(plan_stream)` 传递 recovery join。但在当前主流程里这个方法还未被调用——CUDA graph path 的 recovery join 是一个已知的未完成项。

---

### 4.5 Qwen 3.5 服务器启动到推理的全流程

现在把上面的概念带入真实情境。假设你运行：

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-... \
    --speculative-algorithm EAGLE \
    --speculative-num-draft-tokens 5 \
    --gdn-mtp-cache-mode none
```

**阶段 1：服务器启动初始化（`init_attention_backends()`）**

```
init_attention_backend()
  └── 创建 HybridLinearAttnBackend
        ├── full_attn_backend = FlashAttentionBackend（处理 attention 层）
        └── linear_attn_backend = GDNBackend（处理 SSM 层）
              └── kernel_dispatcher.decode_kernel = FlashInferGDNKernel（如果 SM100）

maybe_init_gdn_recovery_prewarm()
  └── 检测到 FlashInferGDNKernel + use_state_pool=True
  └── 对 10 个 B_pad 值 × T_max 个 T 值，预编译所有 FlashInfer JIT kernels
  └── 约 10 × 5 = 50 次编译（每次 37s → 等 torch.cuda.synchronize() 全部完成）
  └── 打印：Prewarm GDN recovery kernels done in X.XX s
```

**阶段 2：CUDA Graph Capture（`init_cuda_graphs()`）**

```
capture target verify graph for bs=1：
  1. 准备 dummy ForwardBatch（bs=1, draft_token_num=5, 全零 token ids）
  2. attn_backend.init_forward_metadata_out_graph(fb, in_capture=True)
     ← 设置 attention indices 等，这些 op 不录入 graph
  3. with cuda_graph.capture():
       attn_backend.init_forward_metadata_in_graph(fb)
       model.forward(dummy_input_ids, dummy_positions, fb)
     ← GPU 把这里的所有 kernel（attention、SSM、MLP...）录入 graph
  4. graph_bs1 录制完毕

对 bs=2, 3, 4, 5, 6, 7, 8, 10, 12, ... 重复上述过程

⚠️ 注意：_no_cache_mtp_recompute()（recovery）不在 target verify 的 forward 里
   → recovery kernels 不会被录入 CUDA graph
   → 每次 capture 时 _recovery_event_pending=False（服务器刚启动，没有未完成 recovery）
```

**阶段 3：请求到来，第一个 decode step（Speculative Decoding Step 1）**

```
├── Phase A: Draft（在 draft model 上跑 T=5 步自回归）
│   batch = {req_1, req_2, ...} （比如 bs=100 个请求正在 decode）
│   draft_forward() on draft model（draft model 有自己的 SSM pool，不干扰 target）
│   → 生成 5 个草稿 token per request
│
└── Phase B: Verify（在 target model 上验证）
    
    eagle_prepare_for_verify()（在 plan_stream 上，提前准备）
      └── can_run_cuda_graph? bs=100，在 bucket 里？假设是 → True
      └── decode_cuda_graph_runner.load_batch(verify_forward_batch)
            └── _pad_to_bucket(100, capture_bs) = 最近的 bucket（比如 100 或 104）
            └── 把 draft_tokens、seq_lens 等写进静态 buffer
            └── init_forward_metadata_out_graph(fb)  ← 设置 attention indices

    fwd_stream.wait_stream(plan_stream)  ← 主 forward stream 等 plan stream 准备好

    forward_batch_generation(verify_forward_batch)
      └── can_run_graph=True → decode_cuda_graph_runner.execute()
            └── load_batch()（needs_forward_metadata_init=False，跳过，已经准备好）
            └── graph.replay()
                  GPU 在 fwd_stream 上：
                  ← 读取 SSM pool 里的 h_0（初始状态，或上一步 h_K）
                  ← 运行 5 个 draft token 的 target verify forward
                  ← 输出 logits [bs, 5, vocab_size]

    eagle_sample()：从 logits 里找第一个被拒绝的位置
      → 假设 accept K=3 步（token 1,2,3 接受，token 4 拒绝）

    commit_mamba_states_after_verify()
      └── update_mamba_state_after_mtp_verify(last_correct_step_indices=[3,3,...])
            └── intermediate_state_cache is None（cache_mode=none）
            └── _no_cache_mtp_recompute(accepted_steps=[3,3,...])
                  ←── 这里就是 OVERLAP 的 fork 点！
                  
                  [eager path guard]：is_current_stream_capturing()? NO（不在 capture 里）
                  
                  lazy create: self._recovery_stream = torch.cuda.Stream()
                               self._recovery_event = torch.cuda.Event()
                  
                  self._recovery_stream.wait_stream(fwd_stream)
                  ← GPU side：recovery stream 等 fwd_stream 上的 graph.replay() 完毕
                     （等 stash 写完，等 fwd_stream 的 verify forward 全部结束）
                  
                  with torch.cuda.stream(recovery_stream):
                      for layer_id, stash in stash_per_layer.items():
                          gated_delta_rule_mtp(...)
                          ← GPU 在 recovery_stream 上异步执行：
                             从 h_0（SSM pool）出发，走 3 步，写回 h_3
                  
                  recovery_event.record(recovery_stream)
                  state_idx_i32.record_stream(recovery_stream)
                  accepted_steps_i32.record_stream(recovery_stream)
                  _recovery_event_pending = True
                  
                  CPU 立即返回（不等 GPU recovery 完成）！
                  ← GPU 上 recovery_stream 还在跑，但 CPU 已经往下走了

Step 1 结束，CPU 已经在准备 Draft Phase 2

└── Phase A: Draft Step 2（draft model forward，在 fwd_stream 上）
    → draft model 读 draft model 自己的 SSM pool
    → target model 的 SSM pool（recovery 正在写 h_3）DRAFT 不碰
    → 两者并行，没有 hazard

在 draft forward 运行的同时，GPU 的 recovery_stream 也在悄悄地把 h_3 写入 target SSM pool

├── Phase B: Verify Step 2（target model）

    eagle_prepare_for_verify()（plan stream）
      └── load_batch()
            └── init_forward_metadata_out_graph()
                  ⚠️ 此处：eager path → _wait_recovery_if_pending() 不在这条路
                  （CUDA graph path 的 join 是已知未完成项）

    fwd_stream.wait_stream(plan_stream)

    forward_batch_generation()
      └── can_run_graph=True → graph.replay()
          ← GPU 开始读 SSM pool 里的 h_3...
          ← ⚠️ 如果 recovery 还没写完 h_3，这里就是 RAW hazard

    ... 后续同上
```

> **总结**：在 eager decode path（bs 超过 max captured bucket）时，`_wait_recovery_if_pending()` 在 `init_forward_metadata()` 里被正确调用。在 CUDA graph path 里，当前实现里 join 逻辑还未完全接入，这是本 PR 的已知待完善点。

---

### 4.6 两条路径的元数据初始化：为什么要分开写

这是新手最容易困惑的地方：为什么 `init_forward_metadata` 要拆成 `out_graph`、`in_graph` 两个方法？

原因是 CUDA graph capture 的录制机制：

```
capture 时的 GPU timeline：

─── 准备阶段（NOT recorded）─────── with graph.capture(): ──── (recorded) ──────
  init_forward_metadata_out_graph()  |  init_forward_metadata_in_graph()          |
  设置 Python 侧的 indices、          |  发射 GPU kernel（被录进 graph）             |
  host tensors，CPU 侧逻辑            |  例如：build_sparse_attention_mask()        |
  GPU op 也 OK，但不会被录进 graph    |                                              |
──────────────────────────────────── ──────────────────────────────────────────────
```

replay 时：
- `out_graph` 的事情再做一遍（更新 host 数据）
- `in_graph` 录进去的 kernel 通过 `graph.replay()` 自动触发，不需要再调用

`_wait_recovery_if_pending()` 属于「out_graph」的工作：
- 它只是在 GPU stream 上插一个 `wait_event`（纯 GPU timeline 操作）
- 需要在每次 verify forward 之前做，无论是 eager 还是 replay

这就是为什么 `_wait_recovery_if_pending()` 被放在 `init_forward_metadata()`（eager 路径的入口）和 `init_forward_metadata_replay_cuda_graph()`（CUDA graph 路径的入口）里，而不是放在 `in_graph` 里。

---

### 4.7 一图概括：服务器生命周期里两条路径的使用时机

```
服务器生命周期：

[启动]
  ↓
init_attention_backends()        ← 创建 HybridLinearAttnBackend, FlashInferGDNKernel
  ↓
maybe_init_gdn_recovery_prewarm() ← 预编译 FlashInfer JIT，防止推理时冷启动
  ↓
init_cuda_graphs()               ← 对每个 bucket bs capture target verify graph
  ↓
服务器就绪，接受请求

[每个 request batch]
  ↓
Draft Phase                      ← 总是 eager（draft 形状变化大，或用 draft CUDA graph）
  ↓
Verify Phase
  ├── bs <= max_bs AND in bucket？
  │     ↓ YES
  │   [CUDA Graph Replay Path]   ← load_batch → graph.replay()（一条命令跑完）
  │     ↓
  │   commit_mamba_states_after_verify()
  │     └── _no_cache_mtp_recompute()  ← fork recovery 到 side stream
  │           （recovery 与下一步 draft 并行跑）
  │
  └── bs > max_bs OR 其他情况？
        ↓ NO
      [Eager Path]               ← Python 逐层执行 forward
        ↓
      init_forward_metadata()    ← _wait_recovery_if_pending() 在这里 join
        ↓
      model.forward()
        ↓
      commit_mamba_states_after_verify()
        └── _no_cache_mtp_recompute()  ← fork recovery 到 side stream
```

---

## 五、框架全局设计：Overlap 是什么，为什么要在这些地方加代码

> 这一节专门给新手。前三节是逐行分析，这一节是从零开始讲框架。

---

### 4.1 先搞清楚：GDN 模型的 SSM 状态是什么

GDN（Gated Delta Network）是一种 hybrid 模型，它的每一层不是纯 Attention，而是交替使用：
- **Attention 层**：上下文在 KV cache 里，标准做法
- **SSM 层（Mamba2 风格）**：上下文压缩在一个固定大小的**隐状态 h** 里

SSM 的关键特性：

```
h_0（初始状态）→ 吃入 token_1 → h_1 → 吃入 token_2 → h_2 → ... → h_K（当前状态）
```

每次模型读一个新 token，h 就被更新一次。**h 是有损压缩的**——你只能知道「h_K 之后的 token」是什么，不能从 h_K 恢复之前的所有 token（不像 KV cache 那样可以随机访问历史）。

因此，SSM 模型必须在 persistent memory pool（`mamba_cache.temporal`）里为每个请求保存最新的 h，每次推理步骤结束后更新。

---

### 4.2 Speculative Decoding（投机解码）的基本步骤

sglang 里的投机解码是一个**循环**，每一轮（step）分两个阶段：

```
┌─ Step N ──────────────────────────────────────────────────────────┐
│                                                                     │
│  Phase A: Draft（草稿阶段）                                          │
│    Draft model 根据 h_{N-1}（上一步接受后的 SSM 状态）              │
│    连续自回归地生成 T 个草稿 token：[t₁, t₂, ..., tT]              │
│    （只有 draft model 的 SSM 状态在变，target model 的 h 不动）      │
│                                                                     │
│  Phase B: Verify（验证阶段）                                         │
│    Target model 对 [t₁, ..., tT] 并行 forward 一次                  │
│    → 找到第一个 target 不同意的位置 K                               │
│    → 接受 token [t₁, ..., t_K]，拒绝后面的                          │
│    → 需要把 target SSM 状态从 h_{N-1} 推进到 h_K                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Phase B 结束后，我们需要知道 h_K——因为 Step N+1 的 draft 要从 h_K 开始。

---

### 4.3 为什么 Verify 后需要「Recovery」

Target model 在 verify 时，对 T 个 draft token **并行** forward。
但 SSM 是**顺序**的——并行 forward 期间，target model 的 h 不会自然地停在 h_K 上。

Verify forward 结束后，target 的 SSM pool 里存的是什么？取决于 `gdn_mtp_cache_mode`：

**`cache_mode=full`**（代价高）：
- Verify forward 期间，为每个 draft step 都缓存了中间状态 h_0, h_1, ..., h_T
- 接受 K 步后，直接把 h_K 从缓存里拿出来写进 SSM pool
- 不需要重新计算，但内存开销是 `T × state_size`

**`cache_mode=none`**（这个 PR 的场景，内存省但要额外计算）：
- Verify forward 期间，**不缓存**中间 h
- Verify 结束后，SSM pool 里的状态是 **stale 的**（还是 h_{N-1}）
- 必须**重新跑一遍** SSM recurrence，从 h_{N-1} 出发走 K 步，得到 h_K
- 这个重跑就叫 **Recovery**（对应 `_no_cache_mtp_recompute()`）

**Recovery 的输入**（stash）：
Recovery 需要各层的 weights（A_log, dt_bias）和本次 verify 时的 k/v/a/b 中间激活值。
这些激活值在 verify forward 期间被存进了 `stash_per_layer`（见 `gdn_backend.py`）。

---

### 4.4 Overlap 的核心思想：Recovery 和 Draft 并行

没有 Overlap 之前，每个 step 的时序是：

```
主 stream (GPU timeline):
Step N:   ──[Verify Forward]──[Recovery]──[Draft Forward]──
Step N+1: ──────────────────────────────────────────────────[Verify Forward]──[Recovery]──[Draft Forward]──
          时间轴 ──────────────────────────────────────────────────────────────────────────────────────────►
```

Recovery 是纯 GPU 计算（重跑 SSM kernel），需要时间。在这段时间里，**GPU 的计算资源是空着的**——Draft model 的 forward 在等 Recovery 结束才能开始（因为 Draft 需要 h_K，而 h_K 是 Recovery 写的）。

**关键观察**：Draft model forward 读的是 **Draft model 自己的 SSM 状态**，不是 Target model 的 SSM pool。所以：

- Recovery（写 Target 的 h_K）
- Draft Forward（读 Draft 自己的 h）

**这两件事互不干扰**，可以并行！

有了 Overlap 之后：

```
主 stream (GPU):  ──[Verify Forward]──[Draft Forward]──[Verify Forward]──[Draft Forward]──►
side stream (GPU):                ──[Recovery]──                      ──[Recovery]──
                                                ↑                                   ↑
                                           (wait_event)                        (wait_event)
                                   主 stream 在这里等 Recovery 完              主 stream 在这里等
                                   才开始下一次 Verify Forward                才开始下一次 Verify Forward
```

Recovery 的 GPU 时间被 Draft Forward 的 GPU 时间**遮住（hide）**了，白白多了 Recovery 的 throughput。

---

### 4.5 为什么在这些具体位置加代码

现在用代码调用链来说明每个 hook 点为什么在那里。

#### 调用链总览

```
eagle_worker_v2.verify(batch)
  │
  ├── target_worker.forward_batch_generation(verify_forward_batch)   # Target Verify Forward
  │
  ├── commit_mamba_states_after_verify(...)
  │     └── attn_backend.update_mamba_state_after_mtp_verify(...)
  │             └── _no_cache_mtp_recompute(...)                     # ← [HOOK 1] 在这里 fork 到 side stream
  │                   （或 cache_mode=full 走 scatter 路径，不 fork）
  │
  └── 返回                                                            # verify() 返回，CPU 继续
        ↓
eagle_worker_v2.draft(batch)
  │
  ├── prepare_for_draft(...)
  │
  └── draft_forward(forward_batch)                                   # Draft Forward 在主 stream 上跑
        （Recovery 同时在 side stream 上跑，互不影响）

         ↓ 下一个 step
eagle_worker_v2.verify(batch)
  │
  ├── target_worker.forward_batch_generation(verify_forward_batch)
  │     └── attn_backend.init_forward_metadata(forward_batch)        # ← [HOOK 2] 在这里 join side stream
  │             └── _wait_recovery_if_pending()                      # 保证 h_K 已写入，才能开始 Verify
  │
  └── ...
```

#### HOOK 1：`_no_cache_mtp_recompute()` 里 fork 到 side stream

**位置**：`hybrid_linear_attn_backend.py` → `_no_cache_mtp_recompute()` 末尾

```python
# 不是立即等 Recovery 完，而是把它扔到 side stream 上跑
self._recovery_stream.wait_stream(torch.cuda.current_stream())  # side stream 先等主 stream 写完 stash
with torch.cuda.stream(self._recovery_stream):
    _run_recovery()                                              # Recovery kernels 在 side stream 异步跑
self._recovery_event.record(self._recovery_stream)              # 记录一个"完成标记"
self._recovery_event_pending = True                              # 标记：side stream 上还有未完成的工作
```

**为什么在这里**：这是唯一一个知道「当前 step 的 stash 已经写完、accepted_steps 已知」的时机。
再早一步，stash 还没写；再晚一步，下一步的 verify 可能已经开始覆盖 SSM pool。

**`self._recovery_stream.wait_stream(current_stream)` 的必要性**：
Stash（k/v/a/b 中间激活）是在 Verify Forward 的主 stream 上写的。
如果 side stream 不等主 stream，可能在 stash 写完之前就开始读，造成 **Read-After-Write hazard**（读到旧数据）。

#### HOOK 2：`init_forward_metadata()` 里 join side stream

**位置**：`hybrid_linear_attn_backend.py` → `init_forward_metadata()` 开头

```python
def init_forward_metadata(self, forward_batch):
    if forward_batch.forward_mode.is_draft_extend_v2():
        ...
        return                          # draft 阶段：不读 target SSM pool，跳过 join
    self._wait_recovery_if_pending()    # ← target verify 开始前：先等 Recovery 完
    for attn_backend in self.attn_backend_list:
        attn_backend.init_forward_metadata(forward_batch)
```

**为什么在 `init_forward_metadata`，而不是在 `forward()` 里**：

`init_forward_metadata` 是 target forward 开始之前必然会调用的「准备工作」函数，
在这里插入 join 是最早的安全点——保证 join 在任何 SSM 读取之前执行。

**为什么 `is_draft_extend_v2()` 跳过**：
Draft 阶段调用 `init_forward_metadata` 是为了准备 **Draft model** 的 attention metadata，
它不读 Target model 的 SSM pool，所以 join 是不必要的开销，跳过。

#### HOOK 3：`init_forward_metadata_replay_cuda_graph()` 新增重写

**位置**：`hybrid_linear_attn_backend.py` → 新增方法 `init_forward_metadata_replay_cuda_graph()`

这是 CUDA graph replay 路径的等价 join 点。

**为什么需要单独处理 CUDA graph replay**：

- Eager path：每次 forward 都会调用 `init_forward_metadata`（HOOK 2）
- CUDA graph replay path：forward 的 GPU ops 已经录制在 graph 里，不再逐一调用 Python
  但 **metadata 初始化**（CPU 端的 tensor 绑定、索引准备）仍然通过 `init_forward_metadata_replay_cuda_graph` 调用
- 所以 join 必须在这个 replay-path 等价函数里也插入，否则 CUDA graph 路径没有 join

**CUDA graph capture 期间的特殊处理**（HOOK 1 里的 guard）：

```python
if torch.cuda.is_current_stream_capturing():
    _run_recovery()   # capture 期间：必须在 capture stream 上同步跑，不能 fork
    return
```

CUDA graph capture 时，PyTorch 只录制「当前 capture stream 上的 GPU ops」。
如果此时把 Recovery 扔到 side stream，那些 kernels 不会被录进 graph，replay 时 Recovery 就丢了。
所以 capture 期间必须老老实实在 capture stream 上跑，不做 overlap。

---

### 4.6 两个 stream 之间的 hazard 分析（为什么这样同步就够了）

```
时间 →

主 stream:     [stash write] [verify forward ends] [draft forward]          [verify forward starts]
                              ↑                                                        ↑
side stream:                  [recovery stream.wait(main)] [recovery kernels] [event record]
                              │← 等主 stream 写完 stash                                │
                                                                              主 stream.wait_event()
                                                                              等 event 完成
```

**Hazard 列表**：

| Hazard | 是否存在 | 如何消除 |
|---|---|---|
| side stream 读 stash，主 stream 还没写完（RAW）| 存在 | `recovery_stream.wait_stream(main_stream)` |
| 主 stream 下一步 verify 读 SSM pool，side stream 还没写完 h_K（RAW）| 存在 | `main_stream.wait_event(recovery_event)` |
| side stream 读 `state_idx_i32`，主 stream 函数返回后 allocator 回收内存（use-after-free）| 存在 | `state_idx_i32.record_stream(recovery_stream)` |
| draft forward 读 target SSM pool（WAR）| 不存在 | draft 读的是 **draft model** 自己的 SSM pool，不同对象 |

---

### 4.7 一图总结：每一行代码在 GPU timeline 上的作用

```
GPU timeline:

主 stream:   ╔════════════╗   ╔══════════════════╗   ╔═══════════╗   ╔════════════╗
             ║ Verify Fwd ║   ║  Draft Fwd (Step N)║   ║ init_meta ║   ║ Verify Fwd ║
             ╚════════════╝   ╚══════════════════╝   ╚═══════════╝   ╚════════════╝
                      │               │                     │
                      │               │              wait_event ←─────────────────────────┐
                      ↓               │                     │                             │
side stream:    ╔════════════╗        │                     │              ╔════════════╗ │
                ║  Recovery  ║        │                     │              ║  Recovery  ║ │
                ╚════════════╝        │                     │              ╚════════════╝ │
                      │               │                     │                   └──────── event record ┘
                      └── record_stream(state_idx_i32)      │
                               (防止内存被提前释放)        (join 点：确保 h_K 写完再读)
```

- `_recovery_stream.wait_stream(main)` ← 防止 Recovery 读到没写完的 stash
- `with torch.cuda.stream(recovery_stream)` ← 把 Recovery kernel 放到 side stream 的 queue
- `recovery_event.record(recovery_stream)` ← 在 side stream 末尾插一个「完成标记」
- `state_idx_i32.record_stream(recovery_stream)` ← 防止内存被 allocator 提前回收
- `_recovery_event_pending = True` ← 告诉下一个 step「side stream 上有活还没干完」
- `main_stream.wait_event(recovery_event)` ← 下一个 Verify 开始前，GPU 等 Recovery 完

这就是 Overlap 的完整闭环。
