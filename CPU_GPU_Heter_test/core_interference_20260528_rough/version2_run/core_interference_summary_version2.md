# Core 占用干扰粗粒度实验记录

- 时间：`2026-05-28 23:58:37 UTC`
- workload：`prompt=8000`，`output=400`，`batch=1`
- LLM 核数/线程数：`[8, 16, 32, 64, 96, 128]`
- 干扰核数：`[0, 10, 20, 40, 60, 80, 100, 120]`
- 重复次数：`1`
- 超时规则：请求超过 `300s` 未完成则停止该点，吞吐按 `0.1 tok/s` 记录。
- 跳过规则：同一 LLM 核数/亲和性下，一旦某个干扰核数 timeout，后续更高干扰核数直接标记为 synthetic timeout，不再实际运行。
- vLLM 参数：`--gpu-memory-utilization 0.25`，`--max-model-len 10000`
- Prefix cache：显式设置 `--no-enable-prefix-caching`，避免同一 server 内连续请求复用 8K prompt。
- JSON：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/core_interference_20260528_rough/version2_run/core_interference_results_version2.json`
- CSV：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/core_interference_20260528_rough/version2_run/core_interference_raw_version2.csv`

## 输出 PDF

- `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/core_interference_20260528_rough/version2_run/core_interference_request_wall.pdf`
- `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/core_interference_20260528_rough/version2_run/core_interference_tokens_per_sec.pdf`
- `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/core_interference_20260528_rough/version2_run/core_interference_decode_cpu.pdf`
- `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/core_interference_20260528_rough/version2_run/core_interference_slowdown.pdf`

## 实验设计

- `unbound`：vLLM server 不使用 `taskset` 绑定 CPU affinity，但设置 `OMP_NUM_THREADS=LLM核数`。
- `bound`：vLLM server 使用 `taskset` 绑定到指定 LLM cpuset，同时设置 `OMP_NUM_THREADS=LLM核数`。
- bound 模式下优先让干扰 cpuset 与 LLM cpuset disjoint；机器资源不足时记录 logical/physical overlap。
- 干扰程序 `core_burner` 只做寄存器内算术循环，避免制造大规模 DRAM streaming。
- profile 使用 memory summary 前后差分，不启用逐事件 JSONL。
- 早期一次试跑发现 prefix cache hit rate 会随重复请求升高，因此已中止无效试跑并从本结果中排除。
- 同一 LLM 配置下 timeout 后的更高干扰点按用户指定规则直接跳过并记为 timeout。

## 汇总结果

| mode | LLM cores | interference cores | wall median | tok/s median | decode CPU median | decode slowdown | timeout | physical disjoint | physical overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| bound | 8 | 0 | 25.717 | 15.554 | 16.403 | 1.000 | 0 | True | 0 |
| bound | 8 | 10 | 27.535 | 14.527 | 18.048 | 1.100 | 0 | True | 0 |
| bound | 8 | 20 | 29.193 | 13.702 | 18.923 | 1.154 | 0 | True | 0 |
| bound | 8 | 40 | 29.142 | 13.726 | 18.856 | 1.150 | 0 | True | 0 |
| bound | 8 | 60 | 34.035 | 11.753 | 21.727 | 1.325 | 0 | False | 4 |
| bound | 8 | 80 | 35.604 | 11.235 | 23.379 | 1.425 | 0 | False | 8 |
| bound | 8 | 100 | 35.826 | 11.165 | 23.092 | 1.408 | 0 | False | 8 |
| bound | 8 | 120 | 36.429 | 10.980 | 24.004 | 1.463 | 0 | False | 8 |
| bound | 16 | 0 | 21.275 | 18.801 | 11.836 | 1.000 | 0 | True | 0 |
| bound | 16 | 10 | 22.920 | 17.452 | 12.398 | 1.048 | 0 | True | 0 |
| bound | 16 | 20 | 22.907 | 17.462 | 12.841 | 1.085 | 0 | True | 0 |
| bound | 16 | 40 | 23.304 | 17.164 | 13.387 | 1.131 | 0 | True | 0 |
| bound | 16 | 60 | 25.007 | 15.996 | 14.054 | 1.187 | 0 | False | 12 |
| bound | 16 | 80 | 28.263 | 14.153 | 16.174 | 1.367 | 0 | False | 16 |
| bound | 16 | 100 | 28.343 | 14.113 | 15.714 | 1.328 | 0 | False | 16 |
| bound | 16 | 120 | 300.000 | 0.100 | 288.758 | 24.397 | 1 | False | 16 |
| bound | 32 | 0 | 20.203 | 19.799 | 10.036 | 1.000 | 0 | True | 0 |
| bound | 32 | 10 | 20.040 | 19.961 | 9.630 | 0.959 | 0 | True | 0 |
| bound | 32 | 20 | 19.949 | 20.051 | 10.013 | 0.998 | 0 | True | 0 |
| bound | 32 | 40 | 21.069 | 18.985 | 11.276 | 1.124 | 0 | False | 8 |
| bound | 32 | 60 | 21.122 | 18.938 | 10.862 | 1.082 | 0 | False | 28 |
| bound | 32 | 80 | 24.035 | 16.642 | 11.535 | 1.149 | 0 | False | 32 |
| bound | 32 | 100 | 300.000 | 0.100 | 287.837 | 28.679 | 1 | False | 32 |
| bound | 32 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 32 |
| bound | 64 | 0 | 20.441 | 19.569 | 10.429 | 1.000 | 0 | True | 0 |
| bound | 64 | 10 | 25.567 | 15.645 | 15.591 | 1.495 | 0 | False | 10 |
| bound | 64 | 20 | 40.007 | 9.998 | 29.727 | 2.850 | 0 | False | 20 |
| bound | 64 | 40 | 26.113 | 15.318 | 15.844 | 1.519 | 0 | False | 40 |
| bound | 64 | 60 | 23.776 | 16.824 | 12.937 | 1.240 | 0 | False | 60 |
| bound | 64 | 80 | 300.000 | 0.100 | 291.320 | 27.933 | 1 | False | 64 |
| bound | 64 | 100 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| bound | 64 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| bound | 96 | 0 | 20.537 | 19.477 | 10.398 | 1.000 | 0 | True | 0 |
| bound | 96 | 10 | 22.844 | 17.510 | 11.608 | 1.116 | 0 | False | 10 |
| bound | 96 | 20 | 23.862 | 16.763 | 12.121 | 1.166 | 0 | False | 20 |
| bound | 96 | 40 | 23.688 | 16.886 | 12.018 | 1.156 | 0 | False | 40 |
| bound | 96 | 60 | 37.755 | 10.595 | 25.525 | 2.455 | 0 | False | 60 |
| bound | 96 | 80 | 300.000 | 0.100 | 291.715 | 28.056 | 1 | False | 64 |
| bound | 96 | 100 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| bound | 96 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| bound | 128 | 0 | 21.328 | 18.755 | 10.733 | 1.000 | 0 | True | 0 |
| bound | 128 | 10 | 23.077 | 17.334 | 11.843 | 1.103 | 0 | False | 10 |
| bound | 128 | 20 | 22.182 | 18.032 | 11.239 | 1.047 | 0 | False | 20 |
| bound | 128 | 40 | 23.428 | 17.074 | 11.496 | 1.071 | 0 | False | 40 |
| bound | 128 | 60 | 23.473 | 17.041 | 11.897 | 1.108 | 0 | False | 60 |
| bound | 128 | 80 | 300.000 | 0.100 | 290.617 | 27.077 | 1 | False | 64 |
| bound | 128 | 100 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| bound | 128 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| unbound | 8 | 0 | 31.190 | 12.825 | 21.590 | 1.000 | 0 | True | 0 |
| unbound | 8 | 10 | 28.826 | 13.876 | 18.816 | 0.872 | 0 | False | 10 |
| unbound | 8 | 20 | 27.832 | 14.372 | 18.241 | 0.845 | 0 | False | 20 |
| unbound | 8 | 40 | 28.742 | 13.917 | 18.894 | 0.875 | 0 | False | 40 |
| unbound | 8 | 60 | 40.279 | 9.931 | 26.583 | 1.231 | 0 | False | 60 |
| unbound | 8 | 80 | 42.047 | 9.513 | 23.320 | 1.080 | 0 | False | 64 |
| unbound | 8 | 100 | 41.990 | 9.526 | 23.724 | 1.099 | 0 | False | 64 |
| unbound | 8 | 120 | 55.469 | 7.211 | 39.514 | 1.830 | 0 | False | 64 |
| unbound | 16 | 0 | 24.397 | 16.396 | 14.437 | 1.000 | 0 | True | 0 |
| unbound | 16 | 10 | 24.710 | 16.188 | 14.639 | 1.014 | 0 | False | 10 |
| unbound | 16 | 20 | 25.309 | 15.805 | 15.404 | 1.067 | 0 | False | 20 |
| unbound | 16 | 40 | 34.431 | 11.618 | 17.455 | 1.209 | 0 | False | 40 |
| unbound | 16 | 60 | 36.287 | 11.023 | 18.364 | 1.272 | 0 | False | 60 |
| unbound | 16 | 80 | 36.904 | 10.839 | 18.424 | 1.276 | 0 | False | 64 |
| unbound | 16 | 100 | 36.252 | 11.034 | 19.247 | 1.333 | 0 | False | 64 |
| unbound | 16 | 120 | 300.000 | 0.100 | 287.134 | 19.889 | 1 | False | 64 |
| unbound | 32 | 0 | 28.167 | 14.201 | 12.304 | 1.000 | 0 | True | 0 |
| unbound | 32 | 10 | 27.310 | 14.647 | 12.142 | 0.987 | 0 | False | 10 |
| unbound | 32 | 20 | 29.031 | 13.779 | 14.540 | 1.182 | 0 | False | 20 |
| unbound | 32 | 40 | 29.559 | 13.532 | 15.428 | 1.254 | 0 | False | 40 |
| unbound | 32 | 60 | 29.771 | 13.436 | 15.078 | 1.225 | 0 | False | 60 |
| unbound | 32 | 80 | 29.543 | 13.539 | 15.248 | 1.239 | 0 | False | 64 |
| unbound | 32 | 100 | 300.000 | 0.100 | 288.762 | 23.468 | 1 | False | 64 |
| unbound | 32 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| unbound | 64 | 0 | 26.646 | 15.012 | 12.059 | 1.000 | 0 | True | 0 |
| unbound | 64 | 10 | 28.156 | 14.206 | 13.079 | 1.085 | 0 | False | 10 |
| unbound | 64 | 20 | 25.887 | 15.452 | 12.100 | 1.003 | 0 | False | 20 |
| unbound | 64 | 40 | 27.136 | 14.741 | 13.800 | 1.144 | 0 | False | 40 |
| unbound | 64 | 60 | 300.000 | 0.100 | 286.432 | 23.752 | 1 | False | 60 |
| unbound | 64 | 80 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| unbound | 64 | 100 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| unbound | 64 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| unbound | 96 | 0 | 27.510 | 14.540 | 12.299 | 1.000 | 0 | True | 0 |
| unbound | 96 | 10 | 26.559 | 15.061 | 12.628 | 1.027 | 0 | False | 10 |
| unbound | 96 | 20 | 27.501 | 14.545 | 13.330 | 1.084 | 0 | False | 20 |
| unbound | 96 | 40 | 26.287 | 15.216 | 13.170 | 1.071 | 0 | False | 40 |
| unbound | 96 | 60 | 25.207 | 15.868 | 12.803 | 1.041 | 0 | False | 60 |
| unbound | 96 | 80 | 300.000 | 0.100 | 290.969 | 23.658 | 1 | False | 64 |
| unbound | 96 | 100 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| unbound | 96 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| unbound | 128 | 0 | 21.610 | 18.510 | 10.664 | 1.000 | 0 | True | 0 |
| unbound | 128 | 10 | 23.727 | 16.858 | 11.820 | 1.108 | 0 | False | 10 |
| unbound | 128 | 20 | 23.870 | 16.757 | 11.827 | 1.109 | 0 | False | 20 |
| unbound | 128 | 40 | 23.692 | 16.883 | 12.065 | 1.131 | 0 | False | 40 |
| unbound | 128 | 60 | 25.530 | 15.668 | 13.553 | 1.271 | 0 | False | 60 |
| unbound | 128 | 80 | 300.000 | 0.100 | 292.006 | 27.383 | 1 | False | 64 |
| unbound | 128 | 100 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| unbound | 128 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |

## 临时环境变量

- 仅在 vLLM server 子进程内设置 `OMP_NUM_THREADS`。
- 仅在 vLLM server 子进程内设置 `VLLM_HETER_CPU_ATTN_LIB`。
- 仅在 vLLM server 子进程内设置 `VLLM_HETER_PROFILE*` 和 `VLLM_HETER_REQUEST_PROFILE*`。
- 仅在 vLLM server 子进程内设置 `no_proxy/NO_PROXY`。
- 绘图建议使用 `MPLCONFIGDIR=/tmp/matplotlib-codex` 和 `XDG_CACHE_HOME=/tmp/matplotlib-codex`，不修改持久系统环境变量。
- 未修改系统设置。
