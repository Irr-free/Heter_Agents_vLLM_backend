# CPU Attention Kernel 线程数 sweep 记录

实验时间：`2026-05-19 09:12:46 UTC`

## 测试口径

- 直接调用 `libvllm_heter_cpu_attn.so` 中的 vLLM 原生 C++ CPU attention，不启动 vLLM server。
- shape 按 Llama-3.1-8B decode attention 构造：`num_heads=32`、`num_kv_heads=8`、`head_dim=128`、`block_size=16`、`dtype=bfloat16`。
- KV cache 使用 pinned CPU memory，并通过原生 `reshape_and_cache` 打包，尽量贴近异构路径中的 CPU KV cache 布局。
- 每个 output token 调用一次 one-layer CPU attention；表中的 `32-layer tok/s` 是按 32 层线性外推的 CPU attention-only 吞吐。
- no-SMT 不是修改系统超线程状态，而是用 `taskset` 只选择每个物理核的第一个 logical CPU。
- `enable_kv_split=False`，与当前异构路径 `_build_cpu_metadata()` 调用保持一致。

## 临时环境变量

- 子进程临时设置：`no_proxy=localhost,127.0.0.1`、`NO_PROXY=localhost,127.0.0.1`。
- 20/40 线程子进程临时设置：`OMP_NUM_THREADS`。
- profile 子进程临时设置：`VLLM_HETER_CPU_ATTN_INNER_PROFILE_PATH`。
- 未修改系统持久环境变量，未修改系统级超线程/CPU online 设置。

## 线程设置

| Variant | OMP threads | CPU affinity | no-SMT |
|---|---:|---|---|
| OMP 120 | 120 | `not pinned` | False |
| OMP 20 no-SMT | 20 | `0-9,32-41` | True |
| OMP 40 no-SMT | 40 | `0-19,32-51` | True |

## 结果：32-layer CPU attention-only 吞吐

| Case | OMP120 tok/s | 20 no-SMT tok/s | 40 no-SMT tok/s | 20/OMP120 | 40/OMP120 |
|---|---:|---:|---:|---:|---:|
| P4K/O100 | 22.748 | 38.091 | 34.354 | 167.44% | 151.02% |
| P4K/O300 | 39.093 | 37.041 | 31.567 | 94.75% | 80.75% |
| P4K/O400 | 19.484 | 38.647 | 34.889 | 198.36% | 179.07% |
| P6K/O100 | 4.416 | 24.392 | 26.794 | 552.30% | 606.69% |
| P6K/O300 | 4.436 | 24.546 | 23.968 | 553.32% | 540.31% |
| P6K/O400 | 5.397 | 24.205 | 26.278 | 448.50% | 486.92% |
| P8K/O100 | 5.268 | 22.000 | 22.742 | 417.58% | 431.67% |
| P8K/O300 | 3.830 | 21.357 | 23.081 | 557.60% | 602.61% |
| P8K/O400 | 4.077 | 20.011 | 22.062 | 490.79% | 541.08% |

## 结果：one-layer CPU attention 时间

| Case | OMP120 s | 20 no-SMT s | 40 no-SMT s |
|---|---:|---:|---:|
| P4K/O100 | 0.137 | 0.082 | 0.091 |
| P4K/O300 | 0.240 | 0.253 | 0.297 |
| P4K/O400 | 0.642 | 0.323 | 0.358 |
| P6K/O100 | 0.708 | 0.128 | 0.117 |
| P6K/O300 | 2.113 | 0.382 | 0.391 |
| P6K/O400 | 2.316 | 0.516 | 0.476 |
| P8K/O100 | 0.593 | 0.142 | 0.137 |
| P8K/O300 | 2.448 | 0.439 | 0.406 |
| P8K/O400 | 3.066 | 0.625 | 0.567 |

## 调度元数据观察

| Variant | thread_num | effective_thread_num | workitem_group_num | reduction_split_num |
|---|---:|---:|---:|---:|
| OMP 120 | 120 | 1 | 1 | 0 |
| OMP 20 no-SMT | 20 | 1 | 1 | 0 |
| OMP 40 no-SMT | 40 | 1 | 1 | 0 |

## 文件

- JSON：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/cpu_attention_kernel_thread_sweep_results.json`
- PDF：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/cpu_attention_kernel_thread_sweep.pdf`
