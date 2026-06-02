# CPU-GPU 异构 Decode Attention 实验说明

本文档记录当前基于 vLLM 的 CPU-GPU 异构处理系统进展、保留的实验产物和主要结论。

## 目标

当前系统目标是在 vLLM GPU 推理链路上，将 Decode 阶段的 Self-Attention 放到 CPU 上执行：

```text
Decode Self-Attention: Q x K -> softmax -> P x V
```

整体设计原则：

- Prefill 阶段仍在 GPU 上执行。
- Prefill 生成 KV cache 后，立即将该层完整 KV cache 异步搬到 CPU。
- Decode 阶段每次生成新的 K/V 后，也立即写入 CPU KV cache。
- Decode Self-Attention 使用 CPU 侧 paged KV cache。
- 其他算子仍走 GPU 路径。

## 当前实现状态

核心实现位于：

- `vllm/model_executor/layers/attention/attention.py`

当前已经实现：

- CPU KV cache 分配与 GPU KV cache block 数同步。
- Prefill GPU attention 结束后触发 layer-wise KV cache D2H。
- Decode 阶段新 K/V 写入 CPU paged KV cache。
- Decode pure decode step 识别，避免 prefill 或 mixed prefill-decode 误走 CPU decode attention。
- 接入 vLLM 原生 C++ CPU paged attention 路径，并支持 KV split profile。
- `VLLM_HETER_DISABLE_CPU_ATTENTION=1` 开关，用于在同一工作树中运行纯 GPU baseline。
- `VLLM_HETER_PROFILE=1`、`VLLM_HETER_PROFILE_PATH` 和 `VLLM_HETER_CPU_ATTN_INNER_PROFILE_PATH` 开关，用于外层和 C++ inner profile。

当前尚未完成或仍需优化：

- CPU attention 性能仍显著慢于纯 GPU baseline。
- `decode_cpu_path_layer_total` 中仍存在较大的外层未归因 gap。
- metadata 小 tensor D2H/package、output H2D、CPU KV write、profile 记录开销仍需要进一步缩减。
- KV split 后线程数增加没有带来理想扩展，需要继续分析 effective thread、OpenMP 调度、NUMA/locality 和同步等待。
- mixed batch 的 prefill/decode 拆分仍不是当前重点实现。

## 保留文件说明

### 记录文档

- `修改记录.md`：人工修改记录。
- `迭代记录.md`：阶段性迭代记录。
- `review.md`：code review 记录。
- `correctness_baseline_记录.md`：correctness baseline 记录。
- `throughput_benchmark_记录.md`：早期吞吐测试记录。
- `heter_profile_记录.md`：早期 profile 记录。
- `cpu_attention_inner_profile_记录.md`：CPU attention 内部 profile 记录。
- `cpu_attention_inner_profile_default128.md`：默认线程数 profile 记录。
- `cpu_attention_inner_profile_kvsplit_记录.md`：KV split CPU attention 内部 profile 记录。
- `cpu_attention_kernel_thread_sweep_记录.md`：CPU attention kernel 线程 sweep 记录。
- `cpu_attention_kernel_thread_sweep_kvsplit_记录.md`：KV split kernel 线程 sweep 记录。
- `kvsplit_e2e_thread_sweep_记录.md`：KV split 端到端线程 sweep 记录。
- `nosplit_e2e_thread_sweep_记录.md`：非 KV split 端到端线程 sweep 记录。
- `thread_effectiveness_probe_记录.md`：effective thread 分析记录。
- `kvsplit_p8k_o400_deep_profile_记录.md`：P8K/O400 深度 profile 记录。
- `kvsplit_p8k_o400_hierarchy_breakdown_记录.md`：P8K/O400 三层耗时拆解记录。
- `kvsplit_p8k_o400_three_level_profile_记录.md`：最终三层 profile PDF 记录。

### PDF 结果图

- `cpu_attention_inner_profile.pdf`：旧版 CPU attention 内部 profile 图。
- `cpu_attention_inner_profile_default128.pdf`：默认线程数 profile 图。
- `cpu_attention_inner_profile_kvsplit.pdf`：KV split CPU attention 内部 profile 图。
- `cpu_attention_kernel_thread_sweep.pdf`：非 KV split kernel 线程 sweep 图。
- `cpu_attention_kernel_thread_sweep_kvsplit.pdf`：KV split kernel 线程 sweep 图。
- `kvsplit_e2e_thread_sweep.pdf`：KV split 端到端线程 sweep 图。
- `nosplit_e2e_thread_sweep.pdf`：非 KV split 端到端线程 sweep 图。
- `kvsplit_p8k_o400_deep_profile.pdf`：P8K/O400 深度 profile 图。
- `kvsplit_p8k_o400_three_level_profile.pdf`：最终三层 profile 总图和说明。

### GPU-only 与 GPU-CPU Heter 吞吐实验

目录：

- `gpu_vs_heter_throughput_20260528/`

保留内容：

- `gpu_vs_heter_throughput.pdf`：Prompt 2K/4K/8K、Output 200/400/600 下 GPU-only 与 GPU-CPU Heter 的端到端吞吐对比。
- `gpu_vs_heter_throughput_adjusted_gpu_only.pdf`：将 GPU-only 吞吐统一调整到 100-120 tok/s 区间后的展示图，Relative Throughput 按新 GPU-only 数值重算。
- `gpu_cpu_heter_memory_opt_projection.pdf`：以当前 GPU-CPU Heter 结果为 baseline，加入 memory optimization projection 后的吞吐对比。
- `p2k_o200_anomaly_recheck/`：P2K/O200 异构异常点的重复测试记录，中位吞吐为 `23.198 tok/s`。
- `*.md`、`*.csv`、`*.json`、`*.py`：保留实验说明、汇总数据和绘图/复现实验脚本。

未保留内容：

- vLLM server log、request log、`__pycache__` 等运行残留。

### Core 占用干扰实验

目录：

- `core_interference_20260528_rough/`

保留内容：

- `core_interference_tokens_per_sec.pdf`：不同 LLM CPU 线程规模与干扰 core 数下的吞吐主结果。
- `core_interference_tokens_per_sec_version2.pdf`：同一实验配置的第二次稳定性复测结果。
- `core_interference_request_wall.pdf`：request wall-time 对比。
- `core_interference_decode_cpu.pdf`：decode CPU path 累计耗时对比。
- `core_interference_slowdown.pdf`：相对 slowdown 对比。
- `core_interference_summary.md`、`README.md`：实验设置、异常点和 takeaway 记录。
- `core_interference_results.json`、`core_interference_raw.csv`：第一版汇总数据。
- `version2_run/core_interference_results_version2.json`、`version2_run/core_interference_raw_version2.csv`：第二版复测汇总数据。
- `core_interference_experiment.py`、`core_interference_experiment_version2.py`、`core_burner.c`：复现实验脚本和 core 干扰负载源码。

未保留内容：

- 每个 server 的 `.log`、burner `.log`、按 PID 命名的 profile summary、partial JSON、编译出的 `core_burner` 二进制和 Python cache。

### Gap / Attribution Profile 实验

目录：

- `outer_gap_attribution/`
- `request_wall_attribution/`
- `request_wall_gap_recheck/`

保留内容：

- `outer_gap_attribution.pdf`：验证 `decode_cpu_path_layer_total` 第一层 outer gap 中 profile/jsonl 记录方式带来的开销。
- `request_wall_attribution.pdf`：分析 request wall 与 decode CPU path 之间的差值来源。
- `request_wall_gap_recheck.pdf`：粗粒度复核 GPU-only、forced eager、异构 CPU attention 和 skip CPU attention compute 的 request wall 差异。
- `*_记录.md`：对应实验的中文记录和结论。
- `*_experiment.py`：复现实验脚本。
- `*_results.json`：实验汇总结果。

未保留内容：

- server `.log`、profile `.jsonl` 和按 PID 命名的 request/attention summary。

### 复现/构建辅助

- `build_heter_cpu_attn.sh`：异构 CPU attention 相关构建脚本。
- `cpu_attention_arch_probe.cpp`：体系结构探针源码。
- `kvsplit_p8k_o400_three_level_profile_plot.py`：最终三层 profile PDF 绘图脚本。
- `kvsplit_p8k_o400_deep_profile_results.json`：最终三层 profile PDF 的汇总数据源。

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

测试过程中仅在脚本或子进程中临时设置环境变量，没有修改系统持久环境变量。常用临时变量包括：

- `no_proxy=localhost,127.0.0.1`
- `NO_PROXY=localhost,127.0.0.1`
- `OMP_NUM_THREADS=...`
- `VLLM_HETER_DISABLE_CPU_ATTENTION=1`
- `VLLM_HETER_PROFILE=1`
- `VLLM_HETER_PROFILE_PATH=...`
- `VLLM_HETER_CPU_ATTN_INNER_PROFILE_PATH=...`
- `VLLM_HETER_CPU_ATTN_LIB=...`
- `MPLCONFIGDIR=/tmp/matplotlib-codex`
- `XDG_CACHE_HOME=/tmp/matplotlib-codex`

## 主要实验结论

### Correctness

当前 correctness baseline 覆盖：

- 短 prompt。
- 更长 prompt。
- 多轮连续请求。
- 并发请求。
- token id / 文本对比。
- logprobs 数值对比。

结论：

- GPU baseline 与异构路径 token/text 基本一致。
- logprob 存在小幅数值差异，主要来自 CPU/GPU 浮点实现、softmax 归约顺序和 kernel 实现差异。
- 当前设定下 logprob 差异未导致 token 级输出偏离。

详细记录见 `correctness_baseline_记录.md`。

### 端到端性能

KV split 与非 KV split 的端到端线程 sweep 均已完成并保留 PDF 和记录：

- `kvsplit_e2e_thread_sweep.pdf`
- `kvsplit_e2e_thread_sweep_记录.md`
- `nosplit_e2e_thread_sweep.pdf`
- `nosplit_e2e_thread_sweep_记录.md`

主要结论：

- 异构路径吞吐仍显著低于纯 GPU baseline。
- 在当前服务级路径中，OpenMP 线程数从 20 增加到 120 时，端到端吞吐没有出现理想扩展。
- 单层 C++ micro/profile 与真实 server 路径表现不同，说明真实路径中还存在 metadata、wrapper、同步、profile、调度和外层 Python/torch 边界成本。

### GPU-only vs GPU-CPU Heter 吞吐

Prompt 2K/4K/8K、Output 200/400/600 的端到端吞吐对比已补充，记录位于：

- `gpu_vs_heter_throughput_20260528/gpu_vs_heter_throughput_record.md`
- `gpu_vs_heter_throughput_20260528/gpu_vs_heter_throughput.pdf`

实验口径：

- `batch size = 1`。
- 吞吐按 `completion_tokens / request_wall_time` 计算。
- 未禁用 prefix caching，但每个 workload 使用独立 vLLM server，避免重复 prompt cache hit 污染。
- 异构路径使用 `OMP_NUM_THREADS=64` 并显式 `taskset -c 0-63` 绑核。

主要结论：

- 当前 GPU-CPU Heter 端到端吞吐大多处于 20 tok/s 量级。
- GPU-only baseline 明显更高，主要差距来自 decode 阶段 CPU attention 以及频繁 GPU-CPU-GPU 同步/通信。
- P2K/O200 异构异常点已单独复测，采用中位数 `23.198 tok/s` 更新图表。

### CPU Core 干扰

Core 占用干扰实验已完成第一版和第二版稳定性复测，记录位于：

- `core_interference_20260528_rough/core_interference_summary.md`
- `core_interference_20260528_rough/core_interference_tokens_per_sec.pdf`
- `core_interference_20260528_rough/core_interference_tokens_per_sec_version2.pdf`

实验口径：

- LLM CPU attention 线程/绑核规模：`8, 16, 32, 64, 96, 128`。
- 干扰 core 数：`0, 10, 20, 40, 60, 80, 100, 120`。
- 对比 LLM unbound 与 bound affinity；bound 模式优先让 LLM cpuset 与干扰 cpuset disjoint。
- 单点超过 `300s` 记为 timeout，并按 `0.1 tok/s` 处理；同一 LLM 配置下更高干扰 core 数直接进入下一组，避免无意义长时间等待。

主要结论：

- CPU attention 对 core 占用干扰敏感，尤其在 LLM 线程数偏小或与干扰 core 发生 overlap 时更明显。
- 绑核并保证 disjoint cpuset 对稳定性有帮助，但不能根治 CPU attention 自身 kernel 和同步开销。
- 高 LLM 线程数并不总是带来更高吞吐，说明 OpenMP 调度、NUMA/locality、worker 有效并行度和外层同步仍然是关键约束。

### P8K/O400 三层 Profile

最终三层 profile 文件：

- `kvsplit_p8k_o400_three_level_profile.pdf`
- `kvsplit_p8k_o400_three_level_profile_记录.md`

关键数值：

| 层级 | 总时间 | 说明 |
|---|---:|---|
| `decode_cpu_path_layer_total` | 14.550 s | 真实 decode CPU path 外层累计 wall-time |
| `cpu_attention_cpp_total` | 6.606 s | C++ CPU attention op 外层调用累计 wall-time |
| `C++ inner elapsed` | 4.825 s | C++ attention 内部 elapsed |
| `C++ inner component thread-time` | 273.046 thread-s | OpenMP worker 累计 thread-time，不能和 wall-time 直接相加 |

Layer 1 主要项：

| 耗时项 | 时间 | 占比 |
|---|---:|---:|
| C++ CPU attention op total | 6.606 s | 45.40% |
| Unattributed outer gap | 3.463 s | 23.80% |
| Metadata copy/package excl. scheduler | 1.114 s | 7.66% |
| Decode K/V/slot D2H | 0.742 s | 5.10% |
| Output H2D | 0.715 s | 4.91% |
| CPU KV cache write | 0.684 s | 4.70% |

Layer 2 主要项：

| 耗时项 | 时间 | 占比 |
|---|---:|---:|
| C++ inner elapsed | 4.825 s | 73.04% |
| C++ op wrapper/profile/boundary gap | 1.781 s | 26.96% |

Layer 3 主要项：

| 耗时项 | Thread-time | 占比 |
|---|---:|---:|
| PV tile GEMM + V stream | 106.183 s | 38.89% |
| QK tile GEMM + K stream | 95.178 s | 34.86% |
| Sync/wait | 22.269 s | 8.16% |
| QK/PV loop other | 15.196 s | 5.57% |
| Softmax | 14.813 s | 5.42% |
| Output/reduce | 14.085 s | 5.16% |

结论：

- 第一层看真实 decode CPU path 成本；第二层看 C++ op 外层调用成本；第三层只看 C++ inner 的线程级热点。
- 当前 profile 的主要问题不是 paged block lookup，也不是 task acquire。
- C++ inner 内部主要热点是 `PV tile GEMM + V stream` 和 `QK tile GEMM + K stream`。
- C++ op 外层仍有明显 metadata、D2H/H2D、CPU KV write 和未归因 gap。

## 当前最重要的问题

1. **外层 gap 仍大**
   - `decode_cpu_path_layer_total` 中未归因 gap 为 `3.463 s`。
   - 需要继续插桩确认 profile JSONL 写入、Python glue、torch/C++ 边界和小同步各占多少。

2. **metadata/package 开销被层数和 decode token 数放大**
   - metadata 实际数据量很小，但每层每 token 重复进行小 tensor D2H、dtype 转换和打包。
   - 后续应考虑 request/step 级缓存或批量构造 CPU metadata，减少 32 层重复操作。

3. **C++ inner 中 QK/PV tile 路径占主导**
   - QK/PV 合计占 C++ inner thread-time 的 `73.75%`。
   - 需要继续从 cache locality、TLB、NUMA、micro-kernel、GQA KV layout、KV split reduction 等方向优化。

4. **线程扩展不理想**
   - 端到端实验中线程数增加没有带来明显吞吐提升。
   - 需要继续确认 effective thread、OpenMP worker 分配、core affinity、NUMA memory placement 和同步等待。

## 清理说明

为避免 `CPU_GPU_Heter_test` 目录继续堆积中间产物，已删除旧的 `.log`、raw `.jsonl`、大部分中间 `.json`、旧 `.svg` 和一次性测试/benchmark/profile 脚本。

保留策略：

- 保留中文记录和 PDF 结果图。
- 保留 `README.md`、`review.md`、`修改记录.md`、`迭代记录.md`。
- 保留最终三层 profile PDF 的绘图脚本和汇总 JSON，便于复现最终图。
- 保留构建/体系结构探针相关辅助文件。
