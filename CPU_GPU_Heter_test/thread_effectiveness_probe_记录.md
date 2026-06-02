# Thread effectiveness 探针记录

- 时间：`2026-05-19`
- 脚本：`CPU_GPU_Heter_test/verify_thread_effectiveness.py`
- 结果 JSON：`CPU_GPU_Heter_test/thread_effectiveness_probe_results.json`
- 口径：不启动 vLLM server，直接调用 `libvllm_heter_cpu_attn.so` 的 `get_scheduler_metadata()` 和 `attention_with_kv_cache()`。
- shape：Llama-3.1-8B decode attention，`num_heads=32`、`num_kv_heads=8`、`head_dim=128`、`block_size=16`、`dtype=bfloat16`。
- case：`Prompt=8000`，连续模拟 `20` 个 decode step。
- 线程：`OMP_NUM_THREADS=20/40/60/80/120`，并用 `taskset_balanced_socket_primary_cores_then_smt_v2` 做进程级 affinity。
- 未修改系统持久环境变量，未修改系统超线程或系统设置。

## 关键结果

| Mode | OMP threads | thread_num | effective_thread_num | reduction_split_num | avg kernel ms |
|---|---:|---:|---:|---:|---:|
| no-split | 20 | 20 | 1 | 0 | 0.992 |
| no-split | 40 | 40 | 1 | 0 | 1.727 |
| no-split | 60 | 60 | 1 | 0 | 1.271 |
| no-split | 80 | 80 | 1 | 0 | 1.413 |
| no-split | 120 | 120 | 1 | 0 | 1.316 |
| KV split | 20 | 20 | 3 | 3 | 7.926 |
| KV split | 40 | 40 | 5 | 5 | 21.737 |
| KV split | 60 | 60 | 7 | 7 | 9.519 |
| KV split | 80 | 80 | 9 | 9 | 25.441 |
| KV split | 120 | 120 | 13 | 13 | 14.805 |

## 结论

1. `OMP_NUM_THREADS` 确实生效：metadata 中的 `thread_num` 会随 20/40/60/80/120 改变。
2. no-split 路径没有把单 token decode attention 拆给多个 worker：`effective_thread_num=1`、`reduction_split_num=0` 恒定。
3. KV split 路径确实拆分了 KV work，但有效并行度不是 OMP 线程数，而是 3/5/7/9/13 个 split。
4. KV split 引入了 barrier、partial output、reduce 等额外开销；在单请求、batch=1、单 token decode 口径下，增加 OMP 线程数并不等价于提高有效并行度，也不保证端到端吞吐提升。
5. 之前将 direct kernel sweep 中的部分 case 解释为“线程数/绑核能显著改变性能”是 kernel 层面的局部现象，不应直接外推到真实 vLLM 端到端吞吐。
