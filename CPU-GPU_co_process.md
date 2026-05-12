# vLLM CPU-GPU 异构 Decode Attention 改造方案

## 1. 项目概述

### 1.1 背景与目标

本方案旨在将 **vLLM CPU 版本** 的内存管理（CPU Paged KV Cache）与 Decode 阶段 Self-Attention 算子，和 **vLLM GPU 版本** 的其他高性能算子（LayerNorm、MLP、Q/K/V Projection、Sampling 等）进行拼接，构建一套 CPU-GPU 异构处理系统。

核心目标：
- **Prefill 阶段**：完全在 GPU 上执行（包括 Attention），利用 GPU 的高并行计算能力处理长序列。
- **Prefill → Decode 过渡**：Prefill 生成的 KV Cache 立即通过异步 PCIe 搬运到 CPU 主存。
- **Decode 阶段**：
  - 非 Attention 算子（LayerNorm、Linear、MLP）继续在 GPU 执行。
  - Self-Attention 算子（Q × K^T → Softmax → P × V）在 CPU 上执行，使用 vLLM CPU 版本的 Paged Attention Kernel（`ops.cpu_attention_with_kv_cache`）。
  - KV Cache 的**主存储**位于 CPU Pinned Memory 中。

### 1.2 设计原则

1. **提前搬运，Overlap PCIe**：不在 Decode Attention 前临时搬运 KV，而是让 KV Cache 始终驻留在 CPU 上。Prefill 结束后立即启动异步 D2H 拷贝，与 Scheduler 调度重叠。
2. **复用官方代码**：Decode Attention 计算直接复用 vLLM CPU 版本的 `ops.cpu_attention_with_kv_cache` 和 `ops.cpu_attn_reshape_and_cache`，不自行实现 CPU Kernel。
3. **最小侵入**：所有改动集中在 Python 层，不修改任何 C++ / CUDA 源码。
4. **保留 GPU 性能**：Prefill 及 Decode 阶段的非 Attention 算子完全保持原有 GPU 路径不变。

---

## 2. 异构架构设计

### 2.1 整体数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GPU Worker                                   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Prefill Phase                                                 │   │
│  │                                                               │   │
│  │  GPU: Input → Embeddings → DecoderLayer[0..N]                 │   │
│  │         ↓                                                     │   │
│  │  GPU: Q/K/V Projection (Linear)                               │   │
│  │         ↓                                                     │   │
│  │  GPU: unified_kv_cache_update(K, V) → GPU KV Cache (临时)     │   │
│  │         ↓                                                     │   │
│  │  GPU: unified_attention_with_output(Q, K, V, GPU Cache)       │   │
│  │         ↓                                                     │   │
│  │  GPU: MLP → Output                                            │   │
│  │         │                                                     │   │
│  │         └──► Async D2H Stream (non_blocking)                  │   │
│  │              GPU Cache ──permute──► CPU KV Cache (主存)        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Decode Phase (per step, per layer)                            │   │
│  │                                                               │   │
│  │  GPU: LayerNorm                                               │   │
│  │         ↓                                                     │   │
│  │  GPU: Q/K/V Projection                                        │   │
│  │         │                                                     │   │
│  │    K,V ─┼──► CPU (同步拷贝, 1 token)                         │   │
│  │         │      ops.cpu_attn_reshape_and_cache                 │   │
│  │         │      → 写入 CPU KV Cache (主存)                      │   │
│  │         │                                                     │   │
│  │    Q  ──┼──► CPU                                             │   │
│  │         │      ops.cpu_attention_with_kv_cache                │   │
│  │         │      → CPU Paged Attention (读 CPU Cache)           │   │
│  │         │                                                     │   │
│  │ Output◄─┘ 回传 GPU                                           │   │
│  │         ↓                                                     │   │
│  │  GPU: MLP → Sampling                                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 存储架构：双 KV Cache

| Cache 类型 | 位置 | Layout | 用途 | 生命周期 |
|-----------|------|--------|------|---------|
| **GPU KV Cache** | GPU HBM | `[2, num_blocks, block_size, num_kv_heads, head_size]` (NHD) | Prefill Attention 临时存储 | Prefill 期间有效，结束后内容搬至 CPU |
| **CPU KV Cache** | CPU Pinned Memory | `[2, num_blocks, num_kv_heads, block_size, head_size]` (CPU Layout) | Decode Attention 主存储 | 长期有效，随 sequence 增长 |

### 2.3 算子执行位置

| 阶段 | LayerNorm / Projection / MLP / Sampling | KV Cache Update | Attention Compute |
|------|----------------------------------------|-----------------|-------------------|
| **Prefill** | GPU | GPU (`unified_kv_cache_update`) | GPU (`unified_attention_with_output`) |
| **Decode** | GPU | CPU (`ops.cpu_attn_reshape_and_cache`) | CPU (`ops.cpu_attention_with_kv_cache`) |

---

## 3. 修改文件清单

| 序号 | 文件路径 | 修改类型 | 核心功能 |
|------|---------|---------|---------|
| 1 | `vllm/v1/worker/utils.py` | 修改现有函数 | 绑定 CPU KV Cache 到 Attention 层 |
| 2 | `vllm/model_executor/layers/attention/attention.py` | 修改类 + 新增方法 | 异构 Attention 核心逻辑（GPU Prefill + CPU Decode） |
| 3 | `vllm/v1/worker/gpu/model_runner.py` | 修改现有方法 | Pure Decode Batch 强制降级为 Eager 模式（禁用 CUDA Graph） |
| 4 | `vllm/v1/worker/gpu/attn_utils.py` | 可选修改 | 统一初始化 CPU KV Cache（如需要） |

---

## 4. 详细修改方案

### 4.1 文件 1：`vllm/v1/worker/utils.py`

#### 修改位置
函数 `bind_kv_cache()`（约第 457 行）。

#### 修改目的
当前 `bind_kv_cache()` 仅将 GPU KV Cache 绑定到 `Attention.kv_cache`。需要**同时为每个 Attention 层触发 CPU KV Cache 的初始化**，作为 Decode 阶段 Paged Attention 的存储介质。

#### 具体修改
在函数末尾、循环绑定 `forward_context[layer_name].kv_cache = kv_cache` 之后，为每个 `Attention` 层调用其 `init_cpu_kv_cache()` 方法（该方法在文件 2 中实现）。

---

### 4.2 文件 2：`vllm/model_executor/layers/attention/attention.py`

#### 修改位置
`Attention` 类（约第 177 行起），整个类级别的改造。

#### 修改目的
实现整个异构系统的**核心拼接层**。让同一个 `Attention` 层在 Prefill 阶段走 GPU 路径，在 Decode 阶段将 KV 写入 CPU 主存、Q 送到 CPU 执行 Paged Attention、结果回传 GPU。

#### 具体修改（共 7 个部分）

##### ① `__init__`：增加 CPU KV Cache 占位符

在构造函数末尾增加成员变量：
- `self.cpu_kv_cache = None`：延迟到 `bind_kv_cache` 后初始化
- `self._d2h_stream = None`：异步搬运流，延迟到首次使用前初始化

##### ② 新增 `init_cpu_kv_cache()`

**调用时机**：由文件 1 的 `bind_kv_cache()` 在模型加载完成后调用。

**功能**：
- 读取已绑定的 `self.kv_cache`（GPU Tensor，GPU layout）
- 提取 `num_blocks`、`block_size` 等维度信息
- 按 CPU PagedAttention 的 layout 分配 pinned memory：
  - Shape: `(2, num_blocks, num_kv_heads, block_size, head_size)`
  - Device: `cpu`
  - `pin_memory=True`
- 初始化独立的 CUDA Stream：`self._d2h_stream = torch.cuda.Stream(device=gpu_cache.device)`

##### ③ 新增 `_prefill_kv_to_cpu_async()`

**调用时机**：Prefill 阶段 `forward()` 结束后。

**功能**：
- 在 `self._d2h_stream` 上执行 `non_blocking=True` 的异步拷贝
- **Layout 转换**（关键）：
  - GPU K: `[num_blocks, block_size, num_kv_heads, head_size]`
  - CPU K: `[num_blocks, num_kv_heads, block_size, head_size]`
  - 通过 `gpu_k.permute(0, 2, 1, 3)` 完成转换后拷贝
- 只拷贝本次 Prefill 实际涉及的已分配 block，避免全量搬运

##### ④ 新增 `_decode_kv_to_cpu(key, value, attn_metadata)`

**调用时机**：Decode 阶段 `forward()` 开头。

**功能**：
- 将当前 token 的 `key`、`value`（GPU Tensor，shape 为 `[num_tokens, num_kv_heads, head_size]`）同步搬到 CPU
- 将 `attn_metadata.slot_mapping`（GPU Tensor）同步搬到 CPU
- 解包 `self.cpu_kv_cache` 为 `cpu_k_cache`、`cpu_v_cache`
- 调用 `ops.cpu_attn_reshape_and_cache(key_cpu, value_cpu, cpu_k_cache, cpu_v_cache, slot_mapping_cpu, isa="vec")`
  - `isa` 后续可根据 CPU 平台（AMX/NEON/VXE）动态选择

##### ⑤ 新增 `_build_cpu_metadata(attn_metadata)`

**调用时机**：Decode 阶段、调用 CPU Attention kernel 前。

**功能**：
- 从 GPU 的 `attn_metadata` 中提取 CPU kernel 需要的字段
- 将以下 GPU Tensor 同步搬到 CPU：
  - `query_start_loc`
  - `seq_lens`
  - `block_table`
  - `alibi_slopes`（如存在）
  - `scheduler_metadata`（如存在，否则传空 tensor）
- 保持标量参数不变：`causal`、`scale`、`sliding_window`、`softcap`

##### ⑥ 新增 `_cpu_paged_attention(query_cpu, output, cpu_metadata)`

**调用时机**：Decode 阶段、KV 写入完成后。

**功能**：
- 准备 CPU 上的 output buffer（`pin_memory=True`）
- 解包 `self.cpu_kv_cache` 为 `cpu_k_cache`、`cpu_v_cache`
- 调用 `ops.cpu_attention_with_kv_cache(...)`，传入：
  - `query_cpu`（从 GPU 搬来的 Q）
  - `cpu_k_cache`、`cpu_v_cache`（文件 ② 分配的 CPU 主存）
  - 文件 ⑤ 构建的 CPU metadata
- 将 `output_cpu` 同步拷回 GPU，写入 `output`

##### ⑦ 修改 `forward()`：增加 Prefill / Decode 分支

替换原有的统一 GPU Attention 路径，增加条件判断：

| 条件 | 行为 |
|------|------|
| `num_prefill_tokens > 0` | **走原有 GPU 路径**：`unified_kv_cache_update` + `unified_attention_with_output`，结束后调用 `_prefill_kv_to_cpu_async()` |
| `num_prefill_tokens == 0` 且 `query.shape[0] > 0` | **走 CPU 路径**：等待 D2H stream → `_decode_kv_to_cpu()` → Q 搬 CPU → `_build_cpu_metadata()` → `_cpu_paged_attention()` → Output 回 GPU |
| Mixed Batch（同时有 Prefill + Decode） | **Phase 1 暂不处理**，统一降级走原有 **GPU 路径**，避免跨设备拆分 query tensor 的复杂度 |
| CUDA Graph Capture 阶段 | 无论条件如何，**强制走 GPU 路径**（通过 `torch.cuda.is_current_stream_capturing()` 判断），否则 Graph Capture 会失败 |

---

### 4.3 文件 3：`vllm/v1/worker/gpu/model_runner.py`

#### 修改位置
方法 `execute_model()`（约第 955 行），在调用 `model()` 之前的逻辑中。

#### 修改目的
Decode 阶段的 CPU Attention（`ops.cpu_attention_with_kv_cache`、Tensor `.cpu()` 等）**完全不能被 CUDA Graph 捕获**。必须在检测到 Pure Decode Batch 时，**强制降级为 Eager 模式**。

#### 具体修改
1. 在准备 `batch_desc` 后，判断当前 batch 是否为 **Pure Decode**：
   - 判定条件：`scheduler_output.num_prefill_tokens == 0` 且 `batch_desc.num_tokens > 0`
   - （或通过 `input_batch.num_prefill_tokens` 判断）

2. 若 `batch_desc.cg_mode == CUDAGraphMode.FULL` 且为 Pure Decode：
   - 将 `batch_desc.cg_mode` 降级为 `CUDAGraphMode.EAGER`
   - 这样后续代码会走入 `else` 分支（`self.model(**model_inputs)` 的 eager 调用），不会触发 `self.cudagraph_manager.run_fullgraph()`

3. **保留 Prefill Batch 的 CUDA Graph 能力**：当 batch 包含 Prefill 时，不做降级，继续原有 Graph / Piecewise 逻辑。

---

### 4.4 文件 4：`vllm/v1/worker/gpu/attn_utils.py`（可选）

#### 修改位置
函数 `init_kv_cache()`（约第 212 行）。

#### 修改目的
当前 `init_kv_cache()` 仅分配 GPU KV Cache。如果希望 CPU KV Cache 的分配也由统一的初始化流程管理（而非分散在各 `Attention` 层内部），可在此集中分配。

#### 具体修改（可选，Phase 1 建议暂缓）
在 `init_kv_cache()` 末尾增加：
- 遍历 `kv_caches`（GPU cache dict）
- 为每个 layer 计算 CPU shape（permute layout）
- 分配 `torch.empty(cpu_shape, device='cpu', pin_memory=True)`
- 将 CPU cache dict 传给 `bind_kv_cache()`（需同步修改文件 1 的函数签名）

> **建议**：Phase 1 先不改此文件，让 CPU KV Cache 分配集中在文件 2 的 `init_cpu_kv_cache()` 中，降低耦合。后续如需统一内存池管理，再迁移至此文件。

---

## 5. 时序与 Overlap 设计

### 5.1 Prefill → Decode 过渡时序

```
Time →

Prefill Step:
  ├─ [GPU Stream] Forward all layers (LayerNorm → QKV → GPU Attn → MLP)
  │         │
  │         └──► [D2H Stream] _prefill_kv_to_cpu_async() starts
  │              (non_blocking copy + permute)
  │
  └─ [CPU] Scheduler prepares next batch  ←── 与 D2H 重叠!

Decode Step N:
  ├─ [GPU Stream] LayerNorm + Q/K/V Projection
  ├─ [Sync] torch.cuda.current_stream().wait_stream(d2h_stream)
  │         (确保 Prefill KV 已就绪)
  ├─ [GPU→CPU] key/value copy (1 token, ~几 KB)
  ├─ [CPU] ops.cpu_attn_reshape_and_cache
  ├─ [GPU→CPU] query copy (1 token, ~几 KB)
  ├─ [CPU] ops.cpu_attention_with_kv_cache
  │         (读取 CPU Cache 中的历史 KV, 零 PCIe 开销)
  ├─ [CPU→GPU] output copy back
  └─ [GPU Stream] MLP + Sampling
```

### 5.2 Overlap 关键点

- **历史 KV**：已在前序步骤中驻留 CPU Cache，Decode Attention 时**零 PCIe 搬运**。
- **当前 K/V**：仅 1 个 token，数据量极小（`num_kv_heads × head_size × sizeof(dtype)`，通常 < 1KB）。
- **真正的 Overlap**：Prefill 结束后的**大批量 Async D2H** 与 Scheduler 的下一步准备并行。

---

## 6. 边界情况与兼容性

| 场景 | 处理策略 |
|------|---------|
| **Mixed Batch**（同一 batch 中既有 Prefill 又有 Decode） | Phase 1 统一走原有 **GPU 路径**，不走 CPU Attention。避免拆分 query tensor 及同时维护 GPU/CPU 双路径的复杂度。后续 Phase 2 可按 `num_prefill_tokens` 切分处理。 |
| **Dummy Run / Warmup / CUDA Graph Capture** | 在 `Attention.forward()` 中检测 `torch.cuda.is_current_stream_capturing()`。若处于 Capture 阶段，**强制走 GPU 路径**（即使 num_prefill_tokens == 0），否则 CUDA Graph 录制会失败。 |
| **CPU Memory 不足** | CPU KV Cache 大小与 GPU KV Cache 完全对等（`num_blocks` 相同）。需确保系统物理内存 ≥ GPU 显存中 KV Cache 的分配量。对 8B 模型通常占用 < 2GB。 |
| **Tensor Parallel / Pipeline Parallel** | Phase 1 假设单卡（TP=1, PP=1）。多卡场景下 CPU KV Cache 需按 rank 分配，且 `ops.cpu_attention_with_kv_cache` 需适配 TP 切分逻辑，后续扩展。 |
| **Quantized KV Cache (FP8)** | `ops.cpu_attention_with_kv_cache` 不支持 FP8 KV Cache（`cpu_attn.py` 中显式 raise）。需确保配置使用 `bf16` 或 `fp16`。 |
| **CPU 平台差异 (x86/ARM/S390X)** | `ops.cpu_attn_reshape_and_cache` 的 `isa` 参数需根据 `vllm.platforms.current_platform.get_cpu_architecture()` 动态选择（`amx` / `neon` / `vxe` / `vec` / `vec16`）。Phase 1 可硬编码为 `"vec"`，后续优化。 |

---

## 7. 实施阶段（Roadmap）

### Phase 1：最小可运行版本（MVP）
- [ ] 文件 1 改造：`bind_kv_cache` 增加 CPU Cache 初始化触发
- [ ] 文件 2 改造：`Attention` 类实现 `init_cpu_kv_cache()` + `forward()` 双分支
- [ ] 文件 3 改造：`model_runner` 对 Pure Decode 降级 Eager
- [ ] 验证：单卡单请求，Pure Prefill → Pure Decode 完整流程
- [ ] 功能测试：`vllm serve` 能正常启动，输出结果与纯 GPU 版本一致

### Phase 2：性能优化
- [ ] Mixed Batch 支持（同一 batch 内 Prefill 走 GPU、Decode 走 CPU）
- [ ] `isa` 参数根据 CPU 平台自动选择（AMX 加速等）
- [ ] D2H Stream 与 GPU Compute 的细粒度 overlap（层间 overlap）
- [ ] CPU KV Cache 的内存池化管理（避免 per-layer 独立分配碎片）

### Phase 3：扩展性
- [ ] Tensor Parallel 适配（多卡间 CPU KV Cache 的同步机制）
- [ ] Pipeline Parallel 适配
- [ ] 与 vLLM 的 Prefix Caching / Chunked Prefill 深度集成

---

## 8. 风险与假设

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Decode 阶段每层都有 3 次 PCIe 往返（K/V、Q、Output） | Latency 增加 | 单次数据量极小（1 token），且历史 KV 已在 CPU 上；若 latency 不可接受，后续可考虑把 Q/K/V Projection 也搬到 CPU |
| CUDA Graph 禁用导致 Decode 吞吐量下降 10~30% | Throughput 下降 | 这是架构层面的 trade-off，无法避免；优化方向是最大化 CPU Kernel 效率（AMX/AVX512）和 PCIe overlap |
| `ops.cpu_attention_with_kv_cache` 的 metadata 格式与 GPU 版本不完全一致 | Kernel 调用失败 | 严格对照 `cpu_attn.py` 的调用方式构造 metadata，确保所有 tensor 都在 CPU 上且 dtype/shape 正确 |

---

## 附录：关键 API 参考

### GPU 路径（原有，不变）
```python
# KV Cache 写入
torch.ops.vllm.unified_kv_cache_update(key, value, layer_name)

# Attention 计算
torch.ops.vllm.unified_attention_with_output(query, key, value, output, layer_name, ...)
```

### CPU 路径（复用 vLLM CPU 版本）
```python
# KV Cache 写入（CPU Paged Cache）
torch.ops._C.cpu_attn_reshape_and_cache(
    key, value, key_cache, value_cache, slot_mapping, isa
)

# Attention 计算（CPU Paged Attention）
torch.ops._C.cpu_attention_with_kv_cache(
    query, key_cache, value_cache, output,
    query_start_loc, seq_lens, scale, causal,
    alibi_slopes, sliding_window_left, sliding_window_right,
    block_table, softcap, scheduler_metadata, s_aux
)
```

### Layout 对照
| 维度 | GPU Cache (FlashAttention) | CPU Cache (cpu_attention_with_kv_cache) |
|------|---------------------------|----------------------------------------|
| K/V 外层 | `unbind(0)` → `[num_blocks, block_size, num_kv_heads, head_size]` | `unbind(0)` → `[num_blocks, num_kv_heads, block_size, head_size]` |
| 转换 | — | `gpu_k.permute(0, 2, 1, 3)` |
