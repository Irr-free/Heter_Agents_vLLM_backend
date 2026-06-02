# CPU Attention 内部 profile 记录

- 时间：`2026-05-19 08:25:57 GMT`
- 模型：`/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- batch size：`1`
- prompt：`[4000, 6000, 8000]`
- output：`[100, 300, 400]`（包含 400 作为上界补点）
- PDF：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/cpu_attention_inner_profile.pdf`
- 汇总 JSON：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/cpu_attention_inner_profile_results.json`

## 临时环境变量

以下变量只在测试子进程内设置，未修改系统持久环境变量：

- `VLLM_HETER_CPU_ATTN_LIB=/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/vllm/libs/libvllm_heter_cpu_attn.so`
- `VLLM_HETER_PROFILE=1`
- `VLLM_HETER_PROFILE_PATH=/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/cpu_attention_inner_outer_profile_events.jsonl`
- `VLLM_HETER_CPU_ATTN_INNER_PROFILE_PATH=/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/cpu_attention_inner_profile_events.jsonl`
- `no_proxy=localhost,127.0.0.1`
- `NO_PROXY=localhost,127.0.0.1`

## 架构探针

| 指标 | 数值 |
|---|---:|
| OpenMP threads | 128 |
| 顺序读带宽 | 80.72 GiB/s |
| 随机 pointer chase 延迟 | 50.49 ns/access |
| OpenMP barrier 延迟 | 46.70 us/barrier |
| 竞争 atomic fetch_add | 66.46 ns/op |

## 端到端与 CPU attention wall time

| Prompt | Output | Request wall s | CPU attention wall s | CPU attention share | Completion tokens | Inner events |
|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 100 | 18.380 | 12.192 | 66.33% | 100 | 3168 |
| 4000 | 300 | 61.147 | 41.280 | 67.51% | 300 | 9600 |
| 4000 | 400 | 80.749 | 54.547 | 67.55% | 400 | 12800 |
| 6000 | 100 | 24.402 | 18.252 | 74.80% | 100 | 3168 |
| 6000 | 300 | 75.263 | 56.327 | 74.84% | 300 | 9600 |
| 6000 | 400 | 98.731 | 74.210 | 75.16% | 400 | 12800 |
| 8000 | 100 | 30.513 | 24.139 | 79.11% | 100 | 3168 |
| 8000 | 300 | 93.095 | 74.493 | 80.02% | 300 | 9600 |
| 8000 | 400 | 128.613 | 103.176 | 80.22% | 400 | 12800 |

## CPU attention 内部 thread-time 占比

该表使用 OpenMP 线程累计时间，因此用于判断 CPU 核内热点，不等同于 wall time。

| Prompt | Output | QK tile | PV tile | Softmax | Mask | Q copy | Output/reduce | Block lookup | Task acquire | Sync/wait | Loop other |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 100 | 28.47% | 49.61% | 10.60% | 0.01% | 0.04% | 0.03% | 1.68% | 0.01% | 0.00% | 9.54% |
| 4000 | 300 | 28.27% | 49.68% | 10.67% | 0.01% | 0.04% | 0.03% | 1.69% | 0.01% | 0.00% | 9.60% |
| 4000 | 400 | 28.32% | 49.74% | 10.61% | 0.01% | 0.04% | 0.03% | 1.68% | 0.01% | 0.00% | 9.55% |
| 6000 | 100 | 28.50% | 49.78% | 10.47% | 0.01% | 0.03% | 0.02% | 1.68% | 0.01% | 0.00% | 9.50% |
| 6000 | 300 | 28.46% | 49.81% | 10.47% | 0.01% | 0.03% | 0.02% | 1.68% | 0.01% | 0.00% | 9.51% |
| 6000 | 400 | 28.48% | 49.64% | 10.56% | 0.01% | 0.03% | 0.02% | 1.69% | 0.01% | 0.00% | 9.55% |
| 8000 | 100 | 28.50% | 49.84% | 10.48% | 0.01% | 0.02% | 0.02% | 1.67% | 0.01% | 0.00% | 9.46% |
| 8000 | 300 | 28.39% | 50.30% | 10.30% | 0.01% | 0.02% | 0.02% | 1.64% | 0.01% | 0.00% | 9.31% |
| 8000 | 400 | 28.21% | 50.49% | 10.30% | 0.01% | 0.02% | 0.02% | 1.64% | 0.01% | 0.00% | 9.31% |

## 总体热点排序

| Stage | Thread-time s | Share |
|---|---:|---:|
| PV tile GEMM + V stream | 455.658 | 49.99% |
| QK tile GEMM + K stream | 258.549 | 28.37% |
| Softmax | 95.331 | 10.46% |
| QK/PV loop other | 86.129 | 9.45% |
| Paged block lookup | 15.195 | 1.67% |
| Q copy/scale | 0.261 | 0.03% |
| Output/reduce | 0.179 | 0.02% |
| Task acquire/atomic | 0.076 | 0.01% |
| Mask | 0.074 | 0.01% |
| Sync/wait | 0.000 | 0.00% |

## 结论

- 本 profile 已进入 vLLM 原生 C++ CPU attention；内部事件由 `VLLM_HETER_CPU_ATTN_INNER_PROFILE_PATH` 触发。
- `QK tile GEMM + K stream` 和 `PV tile GEMM + V stream` 同时包含 SIMD/AMX 计算与对 CPU KV cache 的流式读取，属于当前最关键的核内路径。
- `Paged block lookup` 反映 paged KV 的 block table 间接寻址成本；如果占比很低，说明主要瓶颈不是页表查找，而是 K/V 数据流与矩阵微内核。
- `Task acquire/atomic`、`Sync/wait` 用来观察 OpenMP 任务分发、split reduction 等同步成本；如果占比高，需要优先看线程数、split 策略和 NUMA 绑定。
- 原始 JSONL 和 server log 已在脚本结束时清理，只保留汇总 JSON、markdown 和 PDF。
