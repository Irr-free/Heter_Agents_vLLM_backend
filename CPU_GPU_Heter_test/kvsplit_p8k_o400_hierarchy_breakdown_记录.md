# KV split P8K/O400 三层耗时拆解

- 数据来源：`CPU_GPU_Heter_test/kvsplit_p8k_o400_deep_profile_results.json`
- 原始事件：
  - `CPU_GPU_Heter_test/kvsplit_p8k_o400_server_outer_events.jsonl`
  - `CPU_GPU_Heter_test/kvsplit_p8k_o400_server_inner_events.jsonl`
- workload：`prompt=8000`，`output=400`，`batch=1`
- OpenMP：`OMP_NUM_THREADS=120`
- 说明：外两层是 wall-time 累计；C++ inner 内部阶段是 OpenMP worker 累计 thread-time，因此不能和 wall time 直接相加。

## 总览

| 层级 | 时间 | 含义 |
|---|---:|---|
| request wall | 25.434 s | HTTP 请求到完整响应的端到端时间 |
| decode_cpu_path_layer_total | 14.550 s | 每层 decode attention 走 CPU 路径的完整累计时间 |
| cpu_attention_cpp_total | 6.606 s | `attention_with_kv_cache(...)` C++ op 调用累计时间 |
| C++ inner elapsed | 4.825 s | C++ attention 内部计时事件累计 elapsed |

## 第一层：decode_cpu_path_layer_total 内部

`decode_cpu_path_layer_total = 14.550 s`。其中 `cpu_attention_cpp_total = 6.606 s`，占 `45.40%`；CPU attention C++ op 之外的外层成本为：

```text
14.550 - 6.606 = 7.944 s
```

| 操作 | 时间 | 占 decode_cpu_path_layer_total | 平均每层每 token |
|---|---:|---:|---:|
| C++ CPU attention op total | 6.606 s | 45.40% | 0.517 ms |
| Metadata D2H/copy/package 总计 | 1.424 s | 9.79% | 0.112 ms |
| Decode K/V/slot D2H | 0.742 s | 5.10% | 0.058 ms |
| Output H2D | 0.715 s | 4.91% | 0.056 ms |
| CPU KV reshape/cache write | 0.684 s | 4.70% | 0.054 ms |
| Wait prefill D2H stream | 0.530 s | 3.64% | 0.042 ms |
| Query D2H | 0.247 s | 1.70% | 0.019 ms |
| CPU output alloc | 0.140 s | 0.96% | 0.011 ms |
| 未单独归因的外层 gap | 3.463 s | 23.80% | 0.271 ms |

Metadata 总计 `1.424 s` 里包含：

| Metadata 子项 | 时间 | 占 decode_cpu_path_layer_total |
|---|---:|---:|
| CPU scheduler metadata | 0.310 s | 2.13% |
| metadata tensor copy/package 其余部分 | 1.114 s | 7.66% |

第一层主要结论：

- 最大项仍是 `cpu_attention_cpp_total`，但它只占 `45.40%`。
- CPU attention C++ op 之外有 `7.944 s`，比 C++ inner elapsed `4.825 s` 还大。
- 外层非 C++ op 的主要可见成本是 metadata、K/V D2H、output H2D、CPU KV write。
- `未单独归因 gap = 3.463 s`，主要来自当前没有单独插桩的 Python/torch 边界、profile 事件写入、函数间隙、对象/字典处理等外层开销。

## 第二层：cpu_attention_cpp_total 到 C++ inner elapsed

`cpu_attention_cpp_total = 6.606 s`，其中 C++ inner elapsed 为 `4.825 s`。

| 项 | 时间 | 占 cpu_attention_cpp_total | 占 decode_cpu_path_layer_total |
|---|---:|---:|---:|
| C++ inner elapsed | 4.825 s | 73.04% | 33.16% |
| C++ op wrapper / profile / 未覆盖边界 | 1.781 s | 26.96% | 12.24% |

第二层主要结论：

- C++ op 调用内部约 `73%` 是被 C++ inner profile 覆盖到的 kernel elapsed。
- 仍有 `1.781 s` 没落到 C++ inner elapsed 中，可能包含 torch custom op dispatch、C++ wrapper 参数处理、profile 聚合/JSONL 写入、op 返回边界等成本。
- 因此只优化 QK/PV kernel 不能覆盖 `cpu_attention_cpp_total` 的全部成本。

## 第三层：C++ inner elapsed 内部

注意：下表是 OpenMP worker 累计 thread-time，总和为 `273.046 s`，大于 C++ inner elapsed `4.825 s`。这表示平均约 `56.59` 个 worker-equivalent 在并行消耗时间。

| C++ inner 操作 | Thread-time | 占 C++ inner thread-time |
|---|---:|---:|
| PV tile GEMM + V stream | 106.183 s | 38.89% |
| QK tile GEMM + K stream | 95.178 s | 34.86% |
| Sync/wait | 22.269 s | 8.16% |
| QK/PV loop other | 15.196 s | 5.57% |
| Softmax | 14.813 s | 5.42% |
| Output/reduce | 14.085 s | 5.16% |
| Paged block lookup | 2.667 s | 0.98% |
| Task acquire/atomic | 2.244 s | 0.82% |
| Q copy/scale | 0.367 s | 0.13% |
| Mask | 0.045 s | 0.02% |

细分观察：

- `PV tile GEMM + V stream` 和 `QK tile GEMM + K stream` 合计 `73.75%`，是最主要热点。
- `Sync/wait` 为 `8.16%`，不是最大项，但 KV split 下已经不可忽略。
- `Output/reduce` 为 `5.16%`，主要来自 split reduction，`reduce_splits` 原始 stage 为 `13.937 s` thread-time。
- `Paged block lookup` 只有 `0.98%`，说明 block table 查找不是主要瓶颈。
- `Task acquire/atomic` 只有 `0.82%`，说明 atomic 抢任务不是主瓶颈。

## 自上而下结论

1. 现在最外层应优先关注 `decode_cpu_path_layer_total`，因为它是 attention 被替换到 CPU 后的真实单层路径成本。
2. 在 `decode_cpu_path_layer_total` 内，`cpu_attention_cpp_total` 是最大单项，但只占 `45.40%`；C++ op 外层还有 `7.944 s`。
3. C++ op 外层的主要可见成本是 metadata、K/V D2H、output H2D、CPU KV write，另外还有 `3.463 s` 未细分外层 gap，需要后续继续插桩。
4. 在 `cpu_attention_cpp_total` 内，C++ inner elapsed 占 `73.04%`，剩余 `26.96%` 是 custom op/wrapper/profile/边界成本。
5. 在 C++ inner 内部，真正的大头是 PV 和 QK 两条路径，合计 `73.75%`；这对应 V/K cache 读取与微内核计算的混合成本。
6. 当前证据不支持 paged block lookup 是主瓶颈，也不支持 task acquire 是主瓶颈。
