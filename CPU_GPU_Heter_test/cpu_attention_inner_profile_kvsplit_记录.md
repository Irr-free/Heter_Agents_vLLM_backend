# CPU Attention 内部 profile 记录

- 时间：`2026-05-19 09:45:10 GMT`
- 模型：`/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- batch size：`1`
- prompt：`[4000, 6000, 8000]`
- output：`[100, 300, 400]`（包含 400 作为上界补点）
- PDF：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/cpu_attention_inner_profile_kvsplit.pdf`
- 汇总 JSON：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/cpu_attention_inner_profile_kvsplit_results.json`

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
| 顺序读带宽 | 16.92 GiB/s |
| 随机 pointer chase 延迟 | 79.64 ns/access |
| OpenMP barrier 延迟 | 337.13 us/barrier |
| 竞争 atomic fetch_add | 59.43 ns/op |

## 端到端与 CPU attention wall time

| Prompt | Output | Request wall s | CPU attention wall s | CPU attention share | Completion tokens | Inner events |
|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 100 | 22.671 | 15.208 | 67.08% | 100 | 3168 |
| 4000 | 300 | 61.456 | 41.587 | 67.67% | 300 | 9600 |
| 4000 | 400 | 82.243 | 56.024 | 68.12% | 400 | 12800 |
| 6000 | 100 | 25.186 | 18.820 | 74.72% | 100 | 3168 |
| 6000 | 300 | 81.685 | 61.097 | 74.80% | 300 | 9600 |
| 6000 | 400 | 106.871 | 81.033 | 75.82% | 400 | 12800 |
| 8000 | 100 | 32.586 | 26.087 | 80.06% | 100 | 3168 |
| 8000 | 300 | 102.345 | 82.239 | 80.35% | 300 | 9600 |
| 8000 | 400 | 157.454 | 126.391 | 80.27% | 400 | 12800 |

## CPU attention 内部 thread-time 占比

该表使用 OpenMP 线程累计时间，因此用于判断 CPU 核内热点，不等同于 wall time。

| Prompt | Output | QK tile | PV tile | Softmax | Mask | Q copy | Output/reduce | Block lookup | Task acquire | Sync/wait | Loop other |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 100 | 28.01% | 50.04% | 10.67% | 0.01% | 0.04% | 0.03% | 1.64% | 0.01% | 0.00% | 9.55% |
| 4000 | 300 | 28.24% | 49.62% | 10.64% | 0.01% | 0.04% | 0.03% | 1.66% | 0.01% | 0.00% | 9.74% |
| 4000 | 400 | 28.25% | 49.40% | 10.71% | 0.01% | 0.04% | 0.03% | 1.69% | 0.01% | 0.00% | 9.86% |
| 6000 | 100 | 28.34% | 49.38% | 10.68% | 0.01% | 0.03% | 0.02% | 1.69% | 0.01% | 0.00% | 9.85% |
| 6000 | 300 | 28.22% | 49.46% | 10.73% | 0.01% | 0.03% | 0.02% | 1.68% | 0.01% | 0.00% | 9.84% |
| 6000 | 400 | 28.25% | 49.80% | 10.51% | 0.01% | 0.03% | 0.02% | 1.66% | 0.01% | 0.00% | 9.71% |
| 8000 | 100 | 28.08% | 50.42% | 10.36% | 0.01% | 0.02% | 0.02% | 1.62% | 0.01% | 0.00% | 9.46% |
| 8000 | 300 | 28.09% | 50.46% | 10.33% | 0.01% | 0.02% | 0.02% | 1.62% | 0.01% | 0.00% | 9.45% |
| 8000 | 400 | 27.80% | 50.45% | 10.57% | 0.01% | 0.02% | 0.02% | 1.62% | 0.01% | 0.00% | 9.51% |

## 总体热点排序

| Stage | Thread-time s | Share |
|---|---:|---:|
| PV tile GEMM + V stream | 501.388 | 49.99% |
| QK tile GEMM + K stream | 281.773 | 28.09% |
| Softmax | 105.892 | 10.56% |
| QK/PV loop other | 96.697 | 9.64% |
| Paged block lookup | 16.532 | 1.65% |
| Q copy/scale | 0.290 | 0.03% |
| Output/reduce | 0.210 | 0.02% |
| Task acquire/atomic | 0.095 | 0.01% |
| Mask | 0.079 | 0.01% |
| Sync/wait | 0.000 | 0.00% |

## 结论

- 本 profile 已进入 vLLM 原生 C++ CPU attention；内部事件由 `VLLM_HETER_CPU_ATTN_INNER_PROFILE_PATH` 触发。
- `QK tile GEMM + K stream` 和 `PV tile GEMM + V stream` 同时包含 SIMD/AMX 计算与对 CPU KV cache 的流式读取，属于当前最关键的核内路径。
- `Paged block lookup` 反映 paged KV 的 block table 间接寻址成本；如果占比很低，说明主要瓶颈不是页表查找，而是 K/V 数据流与矩阵微内核。
- `Task acquire/atomic`、`Sync/wait` 用来观察 OpenMP 任务分发、split reduction 等同步成本；如果占比高，需要优先看线程数、split 策略和 NUMA 绑定。
- 原始 JSONL 和 server log 已在脚本结束时清理，只保留汇总 JSON、markdown 和 PDF。
