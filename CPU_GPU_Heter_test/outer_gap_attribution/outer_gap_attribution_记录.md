# Outer Gap Attribution 实验记录

- 时间：`2026-05-27 04:30:48 UTC`
- workload：`prompt=8000`，`output=400`，`batch=1`
- OpenMP：`OMP_NUM_THREADS=120`
- PDF：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/outer_gap_attribution/outer_gap_attribution.pdf`
- JSON：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/outer_gap_attribution/outer_gap_attribution_results.json`

## 实验组

| 组别 | profile 行为 | request wall | decode_cpu_path_layer_total | outer gap | gap 占比 |
|---|---|---:|---:|---:|---:|
| `A_current_jsonl` | 旧行为：每个 profile event 逐条写 JSONL。 | 26.443 s | 15.039 s | 3.794 s | 25.23% |
| `B_memory_aggregate` | 仍记录所有外层事件，但只在内存聚合并周期性写 summary。 | 22.804 s | 11.702 s | 0.823 s | 7.04% |
| `C_outer_only` | 只记录 decode_cpu_path_layer_total，用于估计子项计时/记录扰动。 | 21.565 s | 10.490 s | 10.490 s | 100.00% |
| `D_profile_off` | 关闭 heter profile，只测 request wall。 | 22.558 s | 0.000 s | 0.000 s | 0.00% |
| `E_fine_memory` | 内存聚合，并增加 outer gap/metadata/C++ wrapper 细粒度计时。 | 115.284 s | 27.198 s | 1.369 s | 5.03% |

## 关键解释

- A 到 B 的 outer gap 差值为 `2.970 s`，用于估计逐 event JSONL 写入对 gap 的影响。
- 旧版 `outer gap` 中约 `78.30%` 来自逐 event JSONL 写入/记录扰动。
- 去掉 JSONL 后，B 组剩余 `outer gap` 为 `0.823 s`，占 `decode_cpu_path_layer_total` 的 `7.04%`。
- A 相比 D 的 request wall 差值为 `3.885 s`，用于估计当前 profile 总体观察者效应。
- B 相比 D 的 request wall 差值为 `0.246 s`，说明内存聚合模式的观察者效应远小于旧 JSONL 模式。
- C 组只记录 `decode_cpu_path_layer_total`，所以 `outer gap=decode_cpu_path_layer_total` 是实验设计导致的数学结果，不能解释为真实 gap。
- E 组 fine timer 明显扰动运行，因此 E 组细项不能直接按真实运行比例分摊，只能用于定位进一步插桩时哪些边界会引入明显观察者效应。

## E 组细粒度 timer

注意：E 组因为每层每 token 增加大量 fine timer，运行时间严重膨胀。下表只用于判断哪些新增插桩点开销/覆盖范围大，不作为真实系统耗时比例。

| event | total s | count | avg ms |
|---|---:|---:|---:|
| `profile_record_self_time` | 0.459641 | 383168 | 0.001200 |
| `decode_gap_after_wait_before_kv` | 0.921997 | 12768 | 0.072212 |
| `decode_gap_after_kv_before_query` | 1.939263 | 12768 | 0.151885 |
| `decode_gap_after_query_before_alloc` | 0.316988 | 12768 | 0.024827 |
| `decode_gap_after_alloc_before_metadata` | 0.183014 | 12768 | 0.014334 |
| `decode_gap_after_metadata_before_cpp` | 1.680526 | 12768 | 0.131620 |
| `decode_gap_after_cpp_before_h2d` | 7.589302 | 12768 | 0.594400 |
| `decode_gap_after_h2d_before_total` | 14.218712 | 12768 | 1.113621 |
| `metadata_query_start_loc_cpu_to_i32` | 0.271399 | 12768 | 0.021256 |
| `metadata_seq_lens_cpu_to_i32` | 0.216781 | 12768 | 0.016978 |
| `metadata_block_table_cpu_to_i32` | 0.230572 | 12768 | 0.018059 |
| `metadata_base_dict_package` | 0.012518 | 12768 | 0.000980 |
| `metadata_alibi_package` | 0.009217 | 12768 | 0.000722 |
| `metadata_ops_lookup` | 0.081854 | 12768 | 0.006411 |
| `metadata_get_scheduler_attr` | 0.008706 | 12768 | 0.000682 |
| `metadata_scheduler_arg_prepare` | 0.032482 | 12768 | 0.002544 |
| `cpu_attention_marker_file` | 1.220294 | 12768 | 0.095574 |
| `cpu_attention_kv_cache_unbind` | 0.103946 | 12768 | 0.008141 |
| `cpu_attention_metadata_get` | 0.048263 | 12768 | 0.003780 |
| `cpu_attention_ops_lookup` | 0.095815 | 12768 | 0.007504 |
| `cpu_attention_arg_prepare` | 0.066477 | 12768 | 0.005207 |

## 临时环境变量

- 仅在测试子进程内设置 `OMP_NUM_THREADS`。
- 仅在测试子进程内设置 `VLLM_HETER_PROFILE`、`VLLM_HETER_PROFILE_MODE`、`VLLM_HETER_PROFILE_PATH`、`VLLM_HETER_PROFILE_SUMMARY_PATH`。
- 仅在测试子进程内设置 `VLLM_HETER_CPU_ATTN_LIB`。
- 仅在测试子进程内设置 `no_proxy/NO_PROXY`。
- 未修改系统持久环境变量，未修改系统超线程或 CPU online 设置。

## 结论

1. 之前三层图里的 `outer gap` 主要不是 CPU attention 算子，也不是 PCIe 传输，而是旧 profile 方式造成的大量逐 event JSONL 写入/记录扰动。
2. 后续 profile 应优先使用内存聚合/summary 模式，不应再用逐 event JSONL 作为性能口径。
3. 若要继续拆剩余 gap，需要重新设计低扰动 fine timer，避免像 E 组这样在每层每 token 上记录过多事件。
