# CPU-GPU Heter Review

Review date: 2026-05-12

## Scope

静态审查了以下文件：

- `CPU_GPU_Heter_test/修改记录.md`
- `CPU_GPU_Heter_test/迭代记录.md`
- `CPU_GPU_Heter_test/test_module_level.py`
- `CPU_GPU_Heter_test/test_end_to_end.py`
- `vllm/model_executor/layers/attention/attention.py`
- `vllm/v1/worker/utils.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu_model_runner.py`

未修改源码、未修改环境变量、未运行会启动服务或写环境的端到端测试。

## Test Result

- 未重新运行 `CPU_GPU_Heter_test/test_end_to_end.py`：该脚本会启动/终止 `vllm serve`、重写 `CPU_GPU_Heter_test/vllm_serve.log`、删除 `/tmp/vllm_cpu_attention_executed.flag`，并执行 `os.environ.setdefault("no_proxy", ...)`。这不符合本次“禁止修改环境和系统环境”的约束。
- 未重新运行 `CPU_GPU_Heter_test/test_module_level.py`：该测试主要 mock 路径选择，不能覆盖下面列出的 KV cache 时序问题。
- 只读取了现有 `CPU_GPU_Heter_test/vllm_serve.log`。日志显示 CPU fallback 路径确实被执行，但这不能证明 CPU KV cache 中的历史 prefill KV 内容是正确的。

## Findings

### 1. High: Prefill KV 搬运发生在 GPU KV cache update 之前

位置：

- `vllm/model_executor/layers/attention/attention.py:687-692`
- `vllm/model_executor/layers/attention/attention.py:703-705`
- `vllm/model_executor/layers/attention/attention.py:722-724`

当前 `forward()` 在 `num_prefill_tokens > 0` 时先调用 `_prefill_kv_to_cpu_async()`，随后才调用 `unified_kv_cache_update()`。但 FlashAttention 等 GPU backend 的 `forward_includes_kv_cache_update=False`，即 KV cache 写入正是由后面的 `unified_kv_cache_update()` 完成。

结果是：prefill 后搬到 CPU 的 cache 可能是旧值或未更新值，而不是当前 prompt 的 K/V。后续 pure decode 走 CPU attention 时，会从 CPU cache 读取错误的历史 K/V。

现有模块测试没有暴露这个问题，因为它 mock 掉了 `unified_kv_cache_update` 和 attention op，只验证“路径被调用”，没有比较 prefill 后 CPU cache 与 GPU cache 内容。

建议验证方式：做一个最小模块测试，在 prefill forward 后同步 `_d2h_stream`，比较 `cpu_kv_cache[0]` 与 `kv_cache[0].permute(0, 2, 1, 3)`，确认拷贝发生在 KV update 之后。

### 2. High: Mixed batch 中 decode token 的 CPU KV cache 可能不会更新

位置：

- `vllm/model_executor/layers/attention/attention.py:1104-1118`

CPU decode 分支仅在 `num_prefill_tokens == 0` 时执行 `_decode_kv_to_cpu()`。如果 batch 中同时包含 prefill 和 decode 请求，当前逻辑会整体走 GPU attention 路径，decode token 的新 K/V 不会通过 `_decode_kv_to_cpu()` 写入 CPU cache。

这意味着该请求下一步进入 pure decode 后，CPU cache 可能缺少上一轮 mixed batch 中生成的 token K/V。

迭代记录里写了 “Mixed Batch 当前 Prefill+Decode 混合 batch 统一走 GPU 路径，未做拆分”，这作为功能限制可以接受；但如果后续 pure decode 依赖 CPU cache，就需要保证 mixed batch 中 decode token 的 KV 也同步到 CPU，或者明确禁用这种调度场景。

### 3. High: 如果启用原生 CPU kernel，`scheduler_metadata` 可能无效

位置：

- `vllm/model_executor/layers/attention/attention.py:817-820`
- `vllm/model_executor/layers/attention/attention.py:839-856`
- `csrc/cpu/cpu_attn.cpp:147-149`
- `csrc/cpu/cpu_attn.cpp:183`

`_build_cpu_metadata()` 在没有 `attn_metadata.scheduler_metadata` 时使用 `torch.empty(0)`。但原生 `cpu_attention_with_kv_cache` 会把 `scheduler_metadata.data_ptr()` reinterpret 成 `cpu_attention::AttentionMetadata*`，并读取 `input.metadata->isa`。

如果 `torch.ops._C.cpu_attention_with_kv_cache` 可用，这里传空 tensor 有崩溃或未定义行为风险。vLLM CPU backend 的正常做法是在 `CPUAttentionMetadataBuilder` 中通过 `ops.cpu_attn_get_scheduler_metadata(...)` 生成 metadata。

当前 Python fallback 路径不会触发这个问题，但一旦切到真正 C++ CPU kernel，这里需要先补齐 CPU scheduler metadata 构造逻辑。

### 4. Medium: Sliding window 参数和 mask 位置计算不可靠

位置：

- `vllm/model_executor/layers/attention/attention.py:270-284`
- `vllm/model_executor/layers/attention/attention.py:850-851`
- `vllm/model_executor/layers/attention/attention.py:870-871`

`Attention.self.sliding_window` 保存的是原始 `int | None`，而 CPU/FlashAttention impl 内部使用的是转换后的 tuple，例如 `(sliding_window - 1, 0)`。当前 CPU path 判断 `isinstance(self.sliding_window, tuple)`，因此对常见 int sliding window 会传 `-1, -1`，等于关闭 sliding window。

Python fallback 的 sliding window mask 还使用 `q_start + qi` 作为 query 的全局序列位置；在 decode batch 中 `q_start` 是 batch 内 query offset，不是请求内 token position。对 sliding-window 模型会导致 mask 错误。

Llama 3.1 这类非 sliding-window 模型不受此项影响，但泛化到 SWA 模型会有 correctness 风险。

### 5. Low: hot path 中残留 `/tmp` 调试写文件和标记文件

位置：

- `vllm/model_executor/layers/attention/attention.py:205-210`
- `vllm/model_executor/layers/attention/attention.py:830-835`
- `vllm/v1/worker/gpu_model_runner.py:3908-3922`

当前实现会在 attention fallback 和 execute_model 热路径中写 `/tmp/vllm_cpu_attention_executed.flag`、`/tmp/vllm_execute_model_debug.log`。这对验证有帮助，但放在长期运行服务中会引入额外 IO，并且标记文件只能说明进入了路径，不能说明数值正确。

## Summary

当前实现确实把 pure decode attention 接到了 CPU fallback 路径上，现有日志也能看到该路径执行。但我认为 prefill KV 搬运的时序问题是必须优先确认的 correctness 风险：如果 CPU cache 没有拿到已经写入后的 prompt KV，那么后续 CPU attention 的结果不可靠。

## Real Test Update: 2026-05-12 12:09-12:11 UTC

用户放权允许为本地请求设置 `no_proxy/NO_PROXY` 后，进行了端到端实测。

新增日志文件：

- `CPU_GPU_Heter_test/vllm_serve_review_20260512_115059.log`
- `CPU_GPU_Heter_test/vllm_serve_review_20260512_115059_escalated.log`
- `CPU_GPU_Heter_test/request_review_20260512_115059.log`
- `CPU_GPU_Heter_test/request_review_20260512_115059_escalated.log`

### Test commands and result

第一次在 Codex 沙箱内启动失败，原因是沙箱无法访问 GPU/NVML：

- `Can't initialize NVML`
- `RuntimeError: Failed to infer device type`

随后在沙箱外启动同一条 `vllm serve` 命令，服务成功启动：

- model: `/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- port: `8001`
- `--max-num-seqs 4`
- `--gpu-memory-utilization 0.30`
- server ready at `2026-05-12 12:08:05 UTC`
- engine initialized successfully, CUDA graph capture completed

沙箱内 curl 无法访问宿主网络 namespace 中的 server，因此请求也在沙箱外执行。请求结果：

- `GET /v1/models`: HTTP 200
- `POST /v1/chat/completions`: HTTP 200
- prompt: `Hello, what is 2+2?`
- `max_tokens=10`
- output: `The answer to 2+2 is 4`
- usage: `prompt_tokens=44`, `completion_tokens=10`, `total_tokens=54`

服务已在测试后通过 Ctrl-C 正常停止。日志中出现 `destroy_process_group() was not called before program exit` warning，这是 Ctrl-C 关闭时的 NCCL 清理 warning，本次没有看到请求期的 Python traceback、IndexError 或 RuntimeError。

### New Finding: High - Prefill 也进入了 CPU Decode Attention 路径

实测日志显示，首次请求的 44 个 prompt tokens 在所有 32 层都进入了：

```text
Decode CPU Paged Attention — Python Fallback 执行中, layer=model.layers.0.self_attn.attn, query_shape=torch.Size([44, 32, 128])
...
Decode CPU Paged Attention — Python Fallback 执行中, layer=model.layers.31.self_attn.attn, query_shape=torch.Size([44, 32, 128])
```

这说明当前判断并没有把 prefill 保持在 GPU attention 路径上。`vllm/v1/attention/backends/flash_attn.py` 的 `FlashAttentionMetadata` 有 `query_start_loc`、`seq_lens`、`block_table`、`slot_mapping` 等字段，但没有 `num_prefill_tokens` 字段。当前代码用：

```python
num_prefill_tokens = getattr(attn_metadata, 'num_prefill_tokens', 0)
```

因此在 FlashAttentionMetadata 下会得到默认值 `0`，导致 `unified_attention_with_output()` 误判为 pure decode，并在 prompt prefill 阶段直接走 CPU fallback。

这个问题会影响核心设计目标：

- 设计目标是 prefill attention 在 GPU 上执行。
- 实测显示 prompt 长度 44 的 prefill attention 在 CPU fallback 上执行。
- 这也解释了为什么现有请求虽然能返回正确短答案，但不能证明异构路径按预期分阶段执行。

建议优先修正 pure decode 判断，不要依赖 `num_prefill_tokens` 这个 FlashAttentionMetadata 不存在的字段。可以从 `query_start_loc` / `seq_lens` / scheduled token 数等真实 metadata 判断当前是否为每个 request 只有 1 个 decode token，或在 runner 侧显式传入可靠的 prefill/decode 标记。
