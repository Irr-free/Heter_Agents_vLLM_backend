# Core 占用干扰粗粒度实验记录

- 时间：`2026-05-28 09:14:16 UTC`
- workload：`prompt=8000`，`output=400`，`batch=1`
- LLM 核数/线程数：`[8, 16, 32, 64, 96, 128]`
- 干扰核数：`[0, 10, 20, 40, 60, 80, 100, 120]`
- 重复次数：`1`
- 超时规则：请求超过 `300s` 未完成则停止该点，吞吐按 `0.1 tok/s` 记录。
- 跳过规则：同一 LLM 核数/亲和性下，一旦某个干扰核数 timeout，后续更高干扰核数直接标记为 synthetic timeout，不再实际运行。
- vLLM 参数：`--gpu-memory-utilization 0.25`，`--max-model-len 10000`
- Prefix cache：显式设置 `--no-enable-prefix-caching`，避免同一 server 内连续请求复用 8K prompt。
- JSON：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/core_interference_20260528_rough/core_interference_results.json`
- CSV：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/core_interference_20260528_rough/core_interference_raw.csv`

## 输出 PDF

- `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/core_interference_20260528_rough/core_interference_request_wall.pdf`
- `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/core_interference_20260528_rough/core_interference_tokens_per_sec.pdf`
- `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/core_interference_20260528_rough/core_interference_decode_cpu.pdf`
- `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/core_interference_20260528_rough/core_interference_slowdown.pdf`

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
| bound | 8 | 0 | 27.098 | 14.761 | 17.677 | 1.000 | 0 | True | 0 |
| bound | 8 | 10 | 28.344 | 14.112 | 18.273 | 1.034 | 0 | True | 0 |
| bound | 8 | 20 | 28.812 | 13.883 | 18.400 | 1.041 | 0 | True | 0 |
| bound | 8 | 40 | 29.114 | 13.739 | 19.380 | 1.096 | 0 | True | 0 |
| bound | 8 | 60 | 32.932 | 12.146 | 21.414 | 1.211 | 0 | False | 4 |
| bound | 8 | 80 | 34.895 | 11.463 | 22.917 | 1.296 | 0 | False | 8 |
| bound | 8 | 100 | 36.028 | 11.102 | 23.169 | 1.311 | 0 | False | 8 |
| bound | 8 | 120 | 38.303 | 10.443 | 26.109 | 1.477 | 0 | False | 8 |
| bound | 16 | 0 | 21.199 | 18.869 | 11.902 | 1.000 | 0 | True | 0 |
| bound | 16 | 10 | 22.383 | 17.871 | 12.514 | 1.051 | 0 | True | 0 |
| bound | 16 | 20 | 22.823 | 17.526 | 12.577 | 1.057 | 0 | True | 0 |
| bound | 16 | 40 | 23.873 | 16.755 | 13.031 | 1.095 | 0 | True | 0 |
| bound | 16 | 60 | 26.248 | 15.239 | 15.010 | 1.261 | 0 | False | 12 |
| bound | 16 | 80 | 28.499 | 14.036 | 15.741 | 1.323 | 0 | False | 16 |
| bound | 16 | 100 | 28.395 | 14.087 | 15.876 | 1.334 | 0 | False | 16 |
| bound | 16 | 120 | 300.000 | 0.100 | 290.580 | 24.415 | 1 | False | 16 |
| bound | 32 | 0 | 20.331 | 19.674 | 10.054 | 1.000 | 0 | True | 0 |
| bound | 32 | 10 | 20.141 | 19.860 | 10.242 | 1.019 | 0 | True | 0 |
| bound | 32 | 20 | 20.080 | 19.921 | 10.081 | 1.003 | 0 | True | 0 |
| bound | 32 | 40 | 20.293 | 19.711 | 10.267 | 1.021 | 0 | False | 8 |
| bound | 32 | 60 | 21.534 | 18.575 | 10.851 | 1.079 | 0 | False | 28 |
| bound | 32 | 80 | 24.304 | 16.458 | 12.154 | 1.209 | 0 | False | 32 |
| bound | 32 | 100 | 300.000 | 0.100 | 288.074 | 28.654 | 1 | False | 32 |
| bound | 32 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 32 |
| bound | 64 | 0 | 21.289 | 18.789 | 10.889 | 1.000 | 0 | True | 0 |
| bound | 64 | 10 | 21.979 | 18.200 | 11.143 | 1.023 | 0 | False | 10 |
| bound | 64 | 20 | 20.614 | 19.404 | 11.162 | 1.025 | 0 | False | 20 |
| bound | 64 | 40 | 108.217 | 3.696 | 95.601 | 8.779 | 0 | False | 40 |
| bound | 64 | 60 | 25.376 | 15.763 | 13.500 | 1.240 | 0 | False | 60 |
| bound | 64 | 80 | 300.000 | 0.100 | 291.649 | 26.783 | 1 | False | 64 |
| bound | 64 | 100 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| bound | 64 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| bound | 96 | 0 | 20.750 | 19.277 | 10.232 | 1.000 | 0 | True | 0 |
| bound | 96 | 10 | 20.639 | 19.381 | 10.670 | 1.043 | 0 | False | 10 |
| bound | 96 | 20 | 23.596 | 16.952 | 11.675 | 1.141 | 0 | False | 20 |
| bound | 96 | 40 | 23.867 | 16.759 | 11.569 | 1.131 | 0 | False | 40 |
| bound | 96 | 60 | 25.275 | 15.826 | 12.913 | 1.262 | 0 | False | 60 |
| bound | 96 | 80 | 300.000 | 0.100 | 291.879 | 28.525 | 1 | False | 64 |
| bound | 96 | 100 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| bound | 96 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| bound | 128 | 0 | 19.960 | 20.040 | 9.784 | 1.000 | 0 | True | 0 |
| bound | 128 | 10 | 23.148 | 17.280 | 11.999 | 1.226 | 0 | False | 10 |
| bound | 128 | 20 | 23.994 | 16.671 | 11.706 | 1.196 | 0 | False | 20 |
| bound | 128 | 40 | 24.529 | 16.307 | 12.527 | 1.280 | 0 | False | 40 |
| bound | 128 | 60 | 33.608 | 11.902 | 21.137 | 2.160 | 0 | False | 60 |
| bound | 128 | 80 | 300.000 | 0.100 | 291.325 | 29.776 | 1 | False | 64 |
| bound | 128 | 100 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| bound | 128 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| unbound | 8 | 0 | 29.409 | 13.601 | 20.216 | 1.000 | 0 | True | 0 |
| unbound | 8 | 10 | 27.301 | 14.651 | 18.117 | 0.896 | 0 | False | 10 |
| unbound | 8 | 20 | 28.002 | 14.285 | 17.942 | 0.888 | 0 | False | 20 |
| unbound | 8 | 40 | 30.467 | 13.129 | 20.968 | 1.037 | 0 | False | 40 |
| unbound | 8 | 60 | 32.106 | 12.459 | 22.069 | 1.092 | 0 | False | 60 |
| unbound | 8 | 80 | 34.718 | 11.521 | 22.375 | 1.107 | 0 | False | 64 |
| unbound | 8 | 100 | 34.727 | 11.518 | 22.785 | 1.127 | 0 | False | 64 |
| unbound | 8 | 120 | 39.497 | 10.127 | 27.124 | 1.342 | 0 | False | 64 |
| unbound | 16 | 0 | 23.424 | 17.077 | 14.025 | 1.000 | 0 | True | 0 |
| unbound | 16 | 10 | 23.521 | 17.006 | 14.036 | 1.001 | 0 | False | 10 |
| unbound | 16 | 20 | 21.773 | 18.371 | 12.204 | 0.870 | 0 | False | 20 |
| unbound | 16 | 40 | 25.442 | 15.722 | 15.889 | 1.133 | 0 | False | 40 |
| unbound | 16 | 60 | 26.236 | 15.246 | 16.362 | 1.167 | 0 | False | 60 |
| unbound | 16 | 80 | 27.215 | 14.698 | 15.460 | 1.102 | 0 | False | 64 |
| unbound | 16 | 100 | 27.234 | 14.687 | 15.434 | 1.100 | 0 | False | 64 |
| unbound | 16 | 120 | 300.000 | 0.100 | 403.598 | 28.778 | 1 | False | 64 |
| unbound | 32 | 0 | 20.619 | 19.400 | 10.935 | 1.000 | 0 | True | 0 |
| unbound | 32 | 10 | 20.666 | 19.356 | 11.272 | 1.031 | 0 | False | 10 |
| unbound | 32 | 20 | 21.023 | 19.027 | 11.142 | 1.019 | 0 | False | 20 |
| unbound | 32 | 40 | 22.785 | 17.555 | 12.087 | 1.105 | 0 | False | 40 |
| unbound | 32 | 60 | 24.212 | 16.520 | 13.208 | 1.208 | 0 | False | 60 |
| unbound | 32 | 80 | 25.014 | 15.991 | 12.793 | 1.170 | 0 | False | 64 |
| unbound | 32 | 100 | 300.000 | 0.100 | 287.002 | 26.247 | 1 | False | 64 |
| unbound | 32 | 120 | 300.000 | 0.100 | 280.440 | 25.647 | 1 | False | 64 |
| unbound | 64 | 0 | 69.150 | 5.785 | 19.442 | 1.000 | 0 | True | 0 |
| unbound | 64 | 10 | 65.592 | 6.098 | 16.502 | 0.849 | 0 | False | 10 |
| unbound | 64 | 20 | 66.259 | 6.037 | 16.970 | 0.873 | 0 | False | 20 |
| unbound | 64 | 40 | 63.137 | 6.335 | 13.692 | 0.704 | 0 | False | 40 |
| unbound | 64 | 60 | 89.738 | 4.457 | 38.065 | 1.958 | 0 | False | 60 |
| unbound | 64 | 80 | 300.000 | 0.100 | 284.282 | 14.622 | 1 | False | 64 |
| unbound | 64 | 100 | 300.000 | 0.100 | 284.747 | 14.646 | 1 | False | 64 |
| unbound | 64 | 120 | 300.000 | 0.100 | 282.396 | 14.525 | 1 | False | 64 |
| unbound | 96 | 0 | 68.740 | 5.819 | 15.465 | 1.000 | 0 | True | 0 |
| unbound | 96 | 10 | 69.843 | 5.727 | 17.561 | 1.136 | 0 | False | 10 |
| unbound | 96 | 20 | 33.786 | 11.839 | 13.075 | 0.845 | 0 | False | 20 |
| unbound | 96 | 40 | 24.184 | 16.540 | 12.431 | 0.804 | 0 | False | 40 |
| unbound | 96 | 60 | 49.270 | 8.118 | 36.523 | 2.362 | 0 | False | 60 |
| unbound | 96 | 80 | 300.000 | 0.100 | 291.828 | 18.871 | 1 | False | 64 |
| unbound | 96 | 100 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| unbound | 96 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| unbound | 128 | 0 | 20.277 | 19.727 | 10.303 | 1.000 | 0 | True | 0 |
| unbound | 128 | 10 | 22.104 | 18.096 | 11.362 | 1.103 | 0 | False | 10 |
| unbound | 128 | 20 | 20.234 | 19.769 | 10.618 | 1.031 | 0 | False | 20 |
| unbound | 128 | 40 | 23.484 | 17.033 | 11.875 | 1.153 | 0 | False | 40 |
| unbound | 128 | 60 | 26.958 | 14.838 | 13.105 | 1.272 | 0 | False | 60 |
| unbound | 128 | 80 | 300.000 | 0.100 | 291.709 | 28.312 | 1 | False | 64 |
| unbound | 128 | 100 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |
| unbound | 128 | 120 | 300.000 | 0.100 | 0.000 | 0.000 | 1 | False | 64 |

## 临时环境变量

- 仅在 vLLM server 子进程内设置 `OMP_NUM_THREADS`。
- 仅在 vLLM server 子进程内设置 `VLLM_HETER_CPU_ATTN_LIB`。
- 仅在 vLLM server 子进程内设置 `VLLM_HETER_PROFILE*` 和 `VLLM_HETER_REQUEST_PROFILE*`。
- 仅在 vLLM server 子进程内设置 `no_proxy/NO_PROXY`。
- 绘图建议使用 `MPLCONFIGDIR=/tmp/matplotlib-codex` 和 `XDG_CACHE_HOME=/tmp/matplotlib-codex`，不修改持久系统环境变量。
- 未修改系统设置。
