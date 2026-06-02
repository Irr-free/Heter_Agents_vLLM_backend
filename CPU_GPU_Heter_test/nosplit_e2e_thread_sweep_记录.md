# No-split 端到端线程数 sweep 记录

- 时间：`2026-05-19 13:00:55 UTC`
- 模型：`/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- batch size：`1`
- prompt：`[4000, 6000, 8000]`
- output：`[100, 300, 400]`
- PDF：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/nosplit_e2e_thread_sweep.pdf`
- 汇总 JSON：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/nosplit_e2e_thread_sweep_results.json`

## 临时环境变量

- GPU baseline 子进程：`VLLM_HETER_DISABLE_CPU_ATTENTION=1`。
- No-split 异构子进程：`VLLM_HETER_CPU_ATTN_LIB=/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/vllm/libs/libvllm_heter_cpu_attn.so`。
- No-split 异构子进程分别设置：`OMP_NUM_THREADS=20/40/60/80/120`。
- No-split 异构子进程使用 `taskset` 进程级 CPU affinity，策略：`taskset_balanced_socket_primary_cores_then_smt_v2`。
- 所有服务子进程设置：`no_proxy=localhost,127.0.0.1`、`NO_PROXY=localhost,127.0.0.1`。
- 未修改系统持久环境变量，未修改系统级 CPU/超线程设置。

## 端到端输出吞吐 tok/s

| Case | GPU | OMP20 | OMP40 | OMP60 | OMP80 | OMP120 | Best no-split | Best/GPU |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| P4K/O100 | 97.380 | 16.846 | 16.451 | 14.374 | 16.098 | 17.125 | OMP120 | 17.59% |
| P4K/O300 | 86.470 | 18.031 | 16.266 | 18.255 | 17.477 | 17.067 | OMP60 | 21.11% |
| P4K/O400 | 95.982 | 15.101 | 16.066 | 18.026 | 17.613 | 16.977 | OMP60 | 18.78% |
| P6K/O100 | 84.890 | 13.539 | 14.024 | 15.683 | 15.488 | 14.857 | OMP60 | 18.47% |
| P6K/O300 | 83.959 | 12.886 | 13.981 | 16.168 | 14.487 | 15.488 | OMP60 | 19.26% |
| P6K/O400 | 70.409 | 11.850 | 14.037 | 16.318 | 14.309 | 15.158 | OMP60 | 23.18% |
| P8K/O100 | 80.172 | 10.511 | 14.015 | 13.343 | 13.133 | 12.264 | OMP40 | 17.48% |
| P8K/O300 | 67.256 | 10.906 | 12.139 | 13.934 | 13.059 | 12.712 | OMP60 | 20.72% |
| P8K/O400 | 93.335 | 11.023 | 13.116 | 14.348 | 13.896 | 12.862 | OMP60 | 15.37% |

## 端到端 request wall time s

| Case | GPU | OMP20 | OMP40 | OMP60 | OMP80 | OMP120 | Best no-split |
|---|---:|---:|---:|---:|---:|---:|---|
| P4K/O100 | 1.027 | 5.936 | 6.079 | 6.957 | 6.212 | 5.839 | OMP120 |
| P4K/O300 | 3.469 | 16.638 | 18.444 | 16.434 | 17.166 | 17.577 | OMP60 |
| P4K/O400 | 4.167 | 26.488 | 24.898 | 22.190 | 22.711 | 23.561 | OMP60 |
| P6K/O100 | 1.178 | 7.386 | 7.131 | 6.376 | 6.457 | 6.731 | OMP60 |
| P6K/O300 | 3.573 | 23.281 | 21.458 | 18.555 | 20.709 | 19.370 | OMP60 |
| P6K/O400 | 5.681 | 33.756 | 28.495 | 24.513 | 27.955 | 26.388 | OMP60 |
| P8K/O100 | 1.247 | 9.514 | 7.135 | 7.495 | 7.615 | 8.154 | OMP40 |
| P8K/O300 | 4.461 | 27.507 | 24.713 | 21.531 | 22.972 | 23.600 | OMP60 |
| P8K/O400 | 4.286 | 36.288 | 30.498 | 27.879 | 28.785 | 31.100 | OMP60 |

## 按线程数汇总

| Variant | Mean tok/s | Geomean tok/s | Mean vs GPU | Best cases |
|---|---:|---:|---:|---:|
| OMP20 | 13.410 | 13.178 | 15.91% | 0/9 |
| OMP40 | 14.455 | 14.387 | 17.24% | 1/9 |
| OMP60 | 15.605 | 15.519 | 18.70% | 7/9 |
| OMP80 | 15.062 | 14.977 | 17.96% | 0/9 |
| OMP120 | 14.946 | 14.830 | 17.83% | 1/9 |

## 结论

- 本测试是端到端 request wall time 口径，包含 prefill 和 decode；吞吐按 `completion_tokens / request_wall_s` 计算。
- GPU baseline 通过临时 `VLLM_HETER_DISABLE_CPU_ATTENTION=1` 禁用异构 CPU attention。
- No-split 异构路径使用 `enable_kv_split=False`，并只改变服务子进程的 `OMP_NUM_THREADS`。
- 详细原始数据见 JSON；图见 PDF。
