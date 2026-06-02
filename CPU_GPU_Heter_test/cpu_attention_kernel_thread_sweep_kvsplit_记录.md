# CPU Attention Kernel 线程数 sweep 记录

实验时间：`2026-05-19 09:26:40 UTC`

## 测试口径

- 直接调用 `libvllm_heter_cpu_attn.so` 中的 vLLM 原生 C++ CPU attention，不启动 vLLM server。
- shape 按 Llama-3.1-8B decode attention 构造：`num_heads=32`、`num_kv_heads=8`、`head_dim=128`、`block_size=16`、`dtype=bfloat16`。
- KV cache 使用 pinned CPU memory，并通过原生 `reshape_and_cache` 打包，尽量贴近异构路径中的 CPU KV cache 布局。
- 每个 output token 调用一次 one-layer CPU attention；表中的 `32-layer tok/s` 是按 32 层线性外推的 CPU attention-only 吞吐。
- no-SMT 不是修改系统超线程状态，而是用 `taskset` 只选择每个物理核的第一个 logical CPU。
- `enable_kv_split=True`，与当前异构路径 `_build_cpu_metadata()` 调用保持一致。

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
| P4K/O100 | 3.286 | 3.160 | 12.271 | 96.16% | 373.44% |
| P4K/O300 | 12.466 | 32.451 | 178.900 | 260.32% | 1435.09% |
| P4K/O400 | 32.726 | 84.998 | 183.282 | 259.73% | 560.06% |
| P6K/O100 | 4.900 | 65.451 | 136.536 | 1335.61% | 2786.19% |
| P6K/O300 | 3.911 | 66.574 | 137.288 | 1702.03% | 3509.92% |
| P6K/O400 | 2.000 | 65.306 | 137.320 | 3265.27% | 6865.91% |
| P8K/O100 | 125.801 | 51.314 | 107.051 | 40.79% | 85.10% |
| P8K/O300 | 44.422 | 50.653 | 107.941 | 114.03% | 242.99% |
| P8K/O400 | 4.832 | 49.068 | 108.184 | 1015.42% | 2238.77% |

## 结果：one-layer CPU attention 时间

| Case | OMP120 s | 20 no-SMT s | 40 no-SMT s |
|---|---:|---:|---:|
| P4K/O100 | 0.951 | 0.989 | 0.255 |
| P4K/O300 | 0.752 | 0.289 | 0.052 |
| P4K/O400 | 0.382 | 0.147 | 0.068 |
| P6K/O100 | 0.638 | 0.048 | 0.023 |
| P6K/O300 | 2.397 | 0.141 | 0.068 |
| P6K/O400 | 6.250 | 0.191 | 0.091 |
| P8K/O100 | 0.025 | 0.061 | 0.029 |
| P8K/O300 | 0.211 | 0.185 | 0.087 |
| P8K/O400 | 2.587 | 0.255 | 0.116 |

## 调度元数据观察

| Variant | thread_num | effective_thread_num | workitem_group_num | reduction_split_num |
|---|---:|---:|---:|---:|
| OMP 120 | 120 | 11 | 11 | 11 |
| OMP 20 no-SMT | 20 | 3 | 3 | 3 |
| OMP 40 no-SMT | 40 | 5 | 5 | 5 |

## 文件

- JSON：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/cpu_attention_kernel_thread_sweep_kvsplit_results.json`
- PDF：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/cpu_attention_kernel_thread_sweep_kvsplit.pdf`
