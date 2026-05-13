# CPU-GPU 异构 Decode Attention 实验说明

本文档记录当前基于 vLLM 的 CPU-GPU 异构处理系统进展、测试结果、性能结论和后续问题。

## 目标

当前系统目标是在 vLLM GPU 推理链路上，将 **Decode 阶段的 Self-Attention** 放到 CPU 上执行：

```text
Decode Self-Attention: Q x K -> softmax -> P x V
```

整体设计原则：

- Prefill 阶段仍在 GPU 上执行。
- Prefill 生成 KV cache 后，立即将该层完整 KV cache 异步搬到 CPU。
- Decode 阶段每次生成新的 K/V 后，也立即写入 CPU KV cache。
- Decode Self-Attention 使用 CPU 侧 paged KV cache。
- 其他算子仍走 GPU 路径。

这样做的目标是提前 overlap 掉 KV cache 在 PCIe 上移动的开销，同时保留 vLLM paged attention 的内存管理思想。

## 当前实现状态

核心实现位于：

- `vllm/model_executor/layers/attention/attention.py`

当前已经实现：

- CPU KV cache 分配与 GPU KV cache block 数同步。
- Prefill GPU attention 结束后触发 layer-wise KV cache D2H。
- Decode 阶段新 K/V 写入 CPU paged KV cache。
- Decode pure decode step 识别，避免把 prefill 或 mixed prefill-decode 错误路由到 CPU attention。
- CPU attention Python fallback，用于 correctness-first 验证。
- `VLLM_HETER_DISABLE_CPU_ATTENTION=1` 开关，用于在同一工作树中运行纯 GPU baseline。
- `VLLM_HETER_PROFILE=1` 和 `VLLM_HETER_PROFILE_PATH` 开关，用于细粒度 profile。

当前尚未完成：

- 真正接入 vLLM 原生 C++ CPU paged attention kernel。
- 消除 Python fallback 中逐 block 收集/拼接 paged KV 的巨大开销。
- 对 mixed batch 做 prefill/decode 拆分。
- 对 CPU NUMA、线程数、ISA、内存布局做系统优化。

## 文件说明

### 测试脚本

- `test_module_level.py`：模块级测试。
- `test_heter_decode_routing.py`：验证 pure decode 路由、prefill 不误走 CPU decode。
- `test_cpu_attention_correctness.py`：CPU attention fallback 与 PyTorch dense reference 对比。
- `test_end_to_end_correctness.py`：端到端 correctness baseline，对比 GPU baseline 和异构路径。
- `benchmark_heter_vs_gpu.py`：纯 GPU vs 异构路径吞吐 benchmark。
- `profile_heter_breakdown.py`：异构路径细粒度 profile。

### 记录与结果

- `修改记录.md`：人工修改记录。
- `迭代记录.md`：阶段性迭代记录。
- `review.md`：code review 记录。
- `correctness_baseline_记录.md`：correctness baseline 记录。
- `throughput_benchmark_记录.md`：吞吐测试记录。
- `heter_profile_记录.md`：细粒度 profile 记录。
- `correctness_report.json`：correctness 原始结果。
- `throughput_benchmark_results.json`：吞吐原始结果。
- `heter_profile_results.json`：profile 汇总结果。
- `heter_profile_events.jsonl`：profile 原始事件。

### 可视化

- `throughput_benchmark_figure.svg`：GPU baseline vs 异构吞吐图。
- `heter_profile_breakdown.svg`：异构路径主要阶段耗时分解图。
- `heter_profile_cpu_attention_breakdown.svg`：CPU attention Python fallback 内部分解图。

### 日志

- `correctness_gpu_baseline.log`
- `correctness_heter_cpu_decode.log`
- `throughput_gpu.log`
- `throughput_heter.log`
- `heter_profile_server.log`
- `vllm_serve.log`

## 环境

测试使用用户指定的 conda 环境：

```bash
/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python
/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/vllm
```

模型路径：

```bash
/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
```

测试过程中仅在脚本或子进程中临时设置环境变量，没有修改系统持久环境变量：

- `no_proxy=localhost,127.0.0.1`
- `NO_PROXY=localhost,127.0.0.1`
- `VLLM_HETER_DISABLE_CPU_ATTENTION=1`：仅用于纯 GPU baseline 子进程。
- `VLLM_HETER_PROFILE=1`：仅用于 profile 子进程。
- `VLLM_HETER_PROFILE_PATH=...`：仅用于 profile 输出路径。

## Correctness 结果

当前 correctness baseline 已覆盖：

- 短 prompt。
- 更长 prompt。
- 多轮连续请求。
- 并发请求。
- token id / 文本对比。
- logprobs 数值对比。

端到端 correctness 结论：

- GPU baseline 与异构路径 token/text 基本一致。
- logprob 存在小幅数值差异，主要来自 CPU/GPU 浮点实现、softmax 归约顺序和 fallback 实现差异。
- 当前设定下 logprob 差异未导致 token 级输出偏离。

详细记录见：

- `correctness_baseline_记录.md`
- `correctness_report.json`

## 吞吐结果

测试脚本：

```bash
/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python CPU_GPU_Heter_test/benchmark_heter_vs_gpu.py
```

配置：

- context length：`128, 512, 1024`
- batch size：`1, 2`
- max tokens：`8`
- 纯 GPU baseline：`VLLM_HETER_DISABLE_CPU_ATTENTION=1`
- 异构路径：默认开启 CPU decode attention

结果摘要：

| Context | Batch | GPU decode tok/s | Heter decode tok/s | Heter/GPU |
|---:|---:|---:|---:|---:|
| 128 | 1 | 102.001 | 0.291 | 0.286% |
| 128 | 2 | 159.613 | 0.322 | 0.202% |
| 512 | 1 | 86.107 | 0.321 | 0.373% |
| 512 | 2 | 162.569 | 0.341 | 0.210% |
| 1024 | 1 | 88.850 | 0.327 | 0.368% |
| 1024 | 2 | 163.622 | 0.330 | 0.202% |

注意：这里的 `decode tok/s` 是 `completion_tokens / request_wall_time`，wall time 包含端到端请求路径，因此不是严格 isolated decode kernel throughput。

结论：

- 当前异构路径吞吐远低于纯 GPU baseline。
- batch size 从 1 到 2 时，GPU baseline 有明显扩展收益；异构路径几乎没有。
- 当前版本是 correctness-first Python fallback，不能代表最终 C++ CPU kernel 的潜在性能。

详细记录见：

- `throughput_benchmark_记录.md`
- `throughput_benchmark_results.json`
- `throughput_benchmark_figure.svg`

## 细粒度 Profile 结果

测试脚本：

```bash
/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python CPU_GPU_Heter_test/profile_heter_breakdown.py
```

配置：

- batch size：`1`
- max tokens：`8`
- context length：`128, 512`
- profile 开关：`VLLM_HETER_PROFILE=1`

主要阶段耗时：

| Context | Wall time s | Instrumented sum s | CPU attention Python s | Output H2D s | Decode KV D2H s | Metadata D2H s |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 29.507 | 28.126 | 27.923 | 0.030 | 0.017 | 0.017 |
| 512 | 26.818 | 24.984 | 24.795 | 0.031 | 0.016 | 0.016 |

CPU attention Python fallback 内部分解：

| Context | CPU attention total s | Collect/concat KV s | Repeat KV s | QK s | Softmax s | PV s | Collect/total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 27.923 | 25.075 | 0.346 | 0.081 | 0.005 | 0.051 | 89.80% |
| 512 | 24.795 | 21.987 | 0.434 | 0.169 | 0.007 | 0.134 | 88.68% |

结论：

- 几十秒开销不是 prefill GPU 计算主导。
- 也不是提前 KV D2H 或 decode 阶段 Q/K/V/output 搬运主导。
- 最大瓶颈是 Python fallback 中每层每 token 从 paged KV block 中逐 block 取出并 `torch.cat` 拼接 K/V。
- 真正的 QK、softmax、PV 计算占比很小。

这说明当前性能问题来自 fallback 实现方式，而不是异构设计中“KV 提前搬到 CPU”这一点本身。

## 为什么需要 C++ CPU kernel

vLLM 原生 CPU 后端不会在 Python 层逐 block 拼接完整 K/V。它调用：

- `ops.cpu_attn_reshape_and_cache`
- `ops.cpu_attention_with_kv_cache`

对应 C++ 实现会在 kernel/main loop 中按 tile 读取 `block_table`，直接通过 pointer offset 访问 paged KV cache，并在 tile GEMM 中完成 QK/PV。

因此下一阶段首选方案是接入或修正真正的 C++ CPU attention kernel：

- 避免 Python 层逐 block `torch.cat`。
- 避免每层每 token 的 Python 调度和小 tensor 操作。
- 保留 paged KV cache 的 block table 访问方式。
- 使用 vLLM CPU 后端已有 ISA dispatch、tile GEMM 和 online softmax 逻辑。

## 当前最重要的问题

1. **性能瓶颈：Python fallback 的 paged KV 收集/拼接**
   - profile 显示该部分占 CPU attention 总耗时约 `88-90%`。
   - 下一步应优先接入 C++ CPU kernel。

2. **C++ CPU kernel 接口尚未正确接通**
   - 当前代码中虽然尝试调用 `torch.ops._C.cpu_attention_with_kv_cache`，但实际运行仍 fallback 到 Python。
   - 需要确认当前构建环境是否包含 CPU extension、op 是否注册、metadata 是否满足 C++ kernel 要求。

3. **scheduler metadata 问题**
   - vLLM CPU attention kernel 依赖 CPU attention scheduler 生成的 metadata。
   - 当前 GPU runner 的 attention metadata 不一定能直接满足 CPU kernel。

4. **CUDA graph 与 profile/CPU 路径兼容**
   - CPU decode 路径必须避开 CUDA graph capture。
   - profile 事件写入也必须避开 capture，否则会导致 `cudaErrorStreamCaptureInvalidated`。

5. **端到端性能 benchmark 仍不是 isolated decode-only**
   - 当前吞吐测试是 API 端到端请求 wall time。
   - 后续需要补 decode-only microbenchmark 或 kernel-level benchmark。

## 下一步计划

1. 检查当前 `vLLM_GPU` 环境是否编译/注册了 `torch.ops._C.cpu_attention_with_kv_cache` 和 `torch.ops._C.cpu_attn_reshape_and_cache`。
2. 对齐当前异构 CPU KV cache layout 与 vLLM CPU kernel 预期 layout。
3. 研究 `vllm/v1/attention/backends/cpu_attn.py` 和 `csrc/cpu/cpu_attn.cpp` 的 metadata/scheduler 依赖。
4. 将 decode CPU attention 从 Python fallback 切换到 C++ CPU kernel。
5. 重新运行 correctness baseline。
6. 重新运行吞吐 benchmark 与细粒度 profile，比较 Python fallback 和 C++ kernel 的差异。

## 当前状态总结

当前版本已经完成了异构系统的 correctness-first 原型和比较完整的测试记录：

- 数据流已跑通。
- Correctness baseline 基本通过。
- 性能问题已定位。
- 最大瓶颈明确为 Python fallback 的 paged KV 收集/拼接。
- 下一阶段应集中处理 C++ CPU kernel 接入，而不是继续优化 PCIe D2H/H2D。
