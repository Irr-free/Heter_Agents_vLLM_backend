# KV split 端到端线程数 sweep 记录

- 时间：`2026-05-19 11:52:59 UTC`
- 模型：`/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- batch size：`1`
- prompt：`[4000, 6000, 8000]`
- output：`[100, 300, 400]`
- PDF：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/kvsplit_e2e_thread_sweep.pdf`
- 汇总 JSON：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/kvsplit_e2e_thread_sweep_results.json`

## 临时环境变量

- GPU baseline 子进程：`VLLM_HETER_DISABLE_CPU_ATTENTION=1`。
- KV split 异构子进程：`VLLM_HETER_CPU_ATTN_LIB=/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/vllm/libs/libvllm_heter_cpu_attn.so`。
- KV split 异构子进程分别设置：`OMP_NUM_THREADS=20/40/60/80/100/120`。
- KV split 异构子进程使用 `taskset` 进程级 CPU affinity，策略：`taskset_balanced_socket_primary_cores_then_smt_v2`。
- 所有服务子进程设置：`no_proxy=localhost,127.0.0.1`、`NO_PROXY=localhost,127.0.0.1`。
- 未修改系统持久环境变量，未修改系统级 CPU/超线程设置。

## 端到端输出吞吐 tok/s

| Case | GPU | OMP20 | OMP40 | OMP60 | OMP80 | OMP100 | OMP120 | Best KV split | Best/GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| P4K/O100 | 88.414 | 7.779 | 7.068 | 23.240 | 22.479 | 21.277 | 22.106 | OMP60 | 26.29% |
| P4K/O300 | 102.924 | 14.170 | 11.710 | 22.082 | 20.986 | 21.697 | 22.749 | OMP120 | 22.10% |
| P4K/O400 | 103.326 | 11.582 | 11.276 | 22.936 | 21.788 | 21.945 | 23.310 | OMP120 | 22.56% |
| P6K/O100 | 95.581 | 11.888 | 20.833 | 18.629 | 17.959 | 22.228 | 17.434 | OMP100 | 23.26% |
| P6K/O300 | 102.446 | 9.093 | 18.131 | 14.741 | 19.136 | 22.354 | 18.792 | OMP100 | 21.82% |
| P6K/O400 | 102.611 | 11.425 | 19.083 | 15.162 | 19.129 | 18.587 | 18.317 | OMP80 | 18.64% |
| P8K/O100 | 95.184 | 9.442 | 18.866 | 18.589 | 17.970 | 19.366 | 16.188 | OMP100 | 20.35% |
| P8K/O300 | 102.767 | 8.486 | 17.221 | 13.334 | 19.035 | 19.587 | 18.462 | OMP100 | 19.06% |
| P8K/O400 | 102.512 | 12.980 | 16.005 | 5.061 | 20.689 | 18.960 | 19.257 | OMP80 | 20.18% |

## 端到端 request wall time s

| Case | GPU | OMP20 | OMP40 | OMP60 | OMP80 | OMP100 | OMP120 | Best KV split |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P4K/O100 | 1.131 | 12.855 | 14.148 | 4.303 | 4.449 | 4.700 | 4.524 | OMP60 |
| P4K/O300 | 2.915 | 21.171 | 25.618 | 13.585 | 14.295 | 13.827 | 13.188 | OMP120 |
| P4K/O400 | 3.871 | 34.536 | 35.473 | 17.440 | 18.359 | 18.228 | 17.160 | OMP120 |
| P6K/O100 | 1.046 | 8.412 | 4.800 | 5.368 | 5.568 | 4.499 | 5.736 | OMP100 |
| P6K/O300 | 2.928 | 32.992 | 16.546 | 20.352 | 15.677 | 13.420 | 15.964 | OMP100 |
| P6K/O400 | 3.898 | 35.010 | 20.961 | 26.382 | 20.911 | 21.520 | 21.837 | OMP80 |
| P8K/O100 | 1.051 | 10.591 | 5.301 | 5.380 | 5.565 | 5.164 | 6.177 | OMP100 |
| P8K/O300 | 2.919 | 35.352 | 17.421 | 22.498 | 15.761 | 15.317 | 16.250 | OMP100 |
| P8K/O400 | 3.902 | 30.818 | 24.993 | 79.030 | 19.334 | 21.097 | 20.771 | OMP80 |

## 按线程数汇总

| Variant | Mean tok/s | Geomean tok/s | Mean vs GPU | Best cases |
|---|---:|---:|---:|---:|
| OMP20 | 10.761 | 10.565 | 10.78% | 0/9 |
| OMP40 | 15.577 | 14.844 | 15.62% | 0/9 |
| OMP60 | 17.086 | 15.803 | 17.34% | 1/9 |
| OMP80 | 19.908 | 19.849 | 20.07% | 2/9 |
| OMP100 | 20.667 | 20.617 | 20.83% | 4/9 |
| OMP120 | 19.624 | 19.485 | 19.76% | 2/9 |

## 结论

- 本测试是端到端 request wall time 口径，包含 prefill 和 decode；吞吐按 `completion_tokens / request_wall_s` 计算。
- GPU baseline 通过临时 `VLLM_HETER_DISABLE_CPU_ATTENTION=1` 禁用异构 CPU attention。
- KV split 异构路径使用当前代码中的 `enable_kv_split=True`，并只改变服务子进程的 `OMP_NUM_THREADS`。
- 详细原始数据见 JSON；图见 PDF。
