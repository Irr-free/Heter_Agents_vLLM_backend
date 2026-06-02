# KV split P8K/O400 CPU Attention 深度 Profile

- 时间：`2026-05-19 15:22:39 UTC`
- 模型：`/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- workload：`prompt=8000`，`output=400`，`batch=1`
- OpenMP threads：`120`
- CPU affinity：`0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123`
- PDF：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/kvsplit_p8k_o400_deep_profile.pdf`
- JSON：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/kvsplit_p8k_o400_deep_profile_results.json`

## 临时环境变量

- 仅在测试子进程内设置：`OMP_NUM_THREADS=120`。
- 仅在测试子进程内设置：`VLLM_HETER_PROFILE=1`、`VLLM_HETER_PROFILE_PATH`、`VLLM_HETER_CPU_ATTN_INNER_PROFILE_PATH`。
- 仅在测试子进程内设置：`VLLM_HETER_CPU_ATTN_LIB`。
- 仅在测试子进程内设置：`no_proxy=localhost,127.0.0.1`、`NO_PROXY=localhost,127.0.0.1`。
- 使用 `taskset` 做进程级 affinity；未修改系统持久环境变量，未修改系统超线程或 CPU online 设置。

## 架构探针

| 指标 | 数值 |
|---|---:|
| OpenMP threads | 120 |
| 顺序读+FP reduction 吞吐（非峰值带宽） | 65.56 GiB/s |
| 随机 pointer chase 延迟 | 41.90 ns/access |
| OpenMP barrier 延迟 | 78.36 us/barrier |
| 竞争 atomic fetch_add | 68.66 ns/op |

## CPU attention wall time 与逻辑 KV 带宽

| Profile | CPU attention wall s | Request/total wall s | Events | Logical KV GiB | Logical KV GiB/s |
|---|---:|---:|---:|---:|---:|
| 真实 vLLM server 32 层 | 4.825 | 25.434 | 12768 | 400.42 | 82.99 |
| 直接 C++ one-layer reuse | 0.310 | 0.317 | 400 | 12.51 | 41.71 |
| 直接 C++ 32-layer working set | 7.188 | 7.221 | 12800 | 400.42 | 59.00 |

## 真实 vLLM server 外层事件

| Event | Count | Total s | Avg ms |
|---|---:|---:|---:|
| decode_cpu_path_layer_total | 12768 | 14.550 | 1.140 |
| cpu_attention_cpp_total | 12768 | 6.606 | 0.517 |
| decode_query_d2h | 12768 | 0.247 | 0.019 |
| decode_kv_d2h | 12768 | 0.742 | 0.058 |
| decode_metadata_d2h | 12768 | 1.424 | 0.112 |
| decode_output_h2d | 12768 | 0.715 | 0.056 |
| decode_cpu_scheduler_metadata | 12768 | 0.310 | 0.024 |
| decode_wait_prefill_d2h | 12768 | 0.530 | 0.042 |
| decode_kv_cache_write_total | 12768 | 0.684 | 0.054 |

## CPU attention 内部 thread-time 占比

注意：这里是 OpenMP worker 累计 thread-time，占比用于定位核内热点，不等同于 wall time。

| Profile | QK tile | PV tile | Softmax | Output/reduce | Block lookup | Task acquire | Sync/wait | Loop other | Q copy | Mask |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 真实 vLLM server 32 层 | 34.86% | 38.89% | 5.42% | 5.16% | 0.98% | 0.82% | 8.16% | 5.57% | 0.13% | 0.02% |
| 直接 C++ one-layer reuse | 5.96% | 7.77% | 2.71% | 1.73% | 0.44% | 0.93% | 77.85% | 2.50% | 0.10% | 0.01% |
| 直接 C++ 32-layer working set | 23.90% | 22.53% | 3.72% | 4.68% | 0.55% | 1.07% | 40.36% | 3.09% | 0.09% | 0.01% |

## 调度元数据

| Profile | thread_num | effective_thread_num | workitem_group_num | reduction_split_num | reduction_item_num |
|---|---:|---:|---:|---:|---:|
| 直接 C++ one-layer reuse | 120 | 13 | 13 | 13 | 1 |
| 直接 C++ 32-layer working set | 120 | 13 | 13 | 13 | 1 |

## 结论

1. 真实 server request wall time 为 `25.434s`；外层 `decode_cpu_path_layer_total` 累计 `14.550s`（约占 request wall 的 `57.21%`），`cpu_attention_cpp_total` 累计 `6.606s`（约 `25.97%`），C++ 内层 kernel elapsed 累计 `4.825s`（约 `18.97%`）。
2. 逻辑 KV 读取量约 `400.42 GiB`；按 C++ 内层 elapsed 折算为 `82.99 GiB/s`。这个数是逻辑访问量除以 kernel elapsed，不等于 PCM/IMC 实际 DRAM 带宽。
3. 架构探针中的 `65.56 GiB/s` 是带 FP reduction 的顺序读吞吐，不是机器峰值内存带宽，因此不能用它作为带宽上限；它只说明当前核内代码路径很容易被计算/reduction/同步混合开销限制。
4. 直接 32-layer working set 的逻辑 KV 带宽为 `59.00 GiB/s`，约为读+reduction 探针的 `89.99%`；它比 one-layer reuse 更接近真实 server，因为 32 层 KV working set 超过 LLC 容量。
5. 真实 server 的 thread-time 热点为 PV tile `38.89%`、QK tile `34.86%`、Softmax `5.42%`。
6. `Paged block lookup` 占比 `0.98%`，说明 paged block table 查找不是主要矛盾。
7. `Task acquire` 占比 `0.82%`，`Sync/wait` 占比 `8.16%`；同步等待不是最大项，但在 KV split 下已经不可忽略。
8. 综合判断：本 case 没有证据表明已经打满整机 DRAM 峰值带宽；更像是 K/V 数据流、QK/PV 微内核、online softmax/reduction、cache/TLB locality 与 KV split 同步共同构成的低效率路径。

## 原始文件

- server inner JSONL：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/kvsplit_p8k_o400_server_inner_events.jsonl`
- server outer JSONL：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/kvsplit_p8k_o400_server_outer_events.jsonl`
- server log：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/kvsplit_p8k_o400_server.log`