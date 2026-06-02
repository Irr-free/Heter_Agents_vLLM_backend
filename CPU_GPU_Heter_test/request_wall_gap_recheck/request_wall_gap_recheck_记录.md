# Request Wall Gap Recheck 实验记录

- 时间：`2026-05-27 07:41:24 UTC`
- workload：`prompt=8000`，`output=400`，`batch=1`
- OpenMP：`OMP_NUM_THREADS=120`
- vLLM 参数：`--gpu-memory-utilization 0.25`，`--max-model-len 10000`
- PDF：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/request_wall_gap_recheck/request_wall_gap_recheck.pdf`
- JSON：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/request_wall_gap_recheck/request_wall_gap_recheck_results.json`

## 为什么重测

- 之前把 `gpu_model_runner_model_forward - decode_cpu_path_layer_total` 直接解释成 GPU 其他算子耗时，口径过粗。
- 当前代码里 pure decode 会为了 CPU attention 强制关闭 CUDA Graph；如果 GPU baseline 没有同口径控制，就不能直接拿 4s baseline 去相减。
- 本次修正了 baseline 开关：`VLLM_HETER_DISABLE_CPU_ATTENTION=1` 时不再触发异构 eager 降级；另用 `VLLM_HETER_FORCE_DECODE_EAGER=1` 构造 GPU eager 对照。

## 四组结果

| 组别 | 说明 | request wall | model forward | decode CPU attention | wall - decode CPU |
|---|---|---:|---:|---:|---:|
| `gpu_default_graph` | 禁用 CPU attention，保留默认 CUDA Graph。 | 3.412 s | 0.031 s | 0.000 s | 3.412 s |
| `gpu_forced_eager` | 禁用 CPU attention，但强制 pure decode 走 eager，用于量化 CUDA Graph 被关闭后的非 attention 开销。 | 4.528 s | 3.588 s | 0.000 s | 4.528 s |
| `heter_cpu_attention` | 启用 CPU decode attention，并强制 pure decode 走 eager。 | 21.858 s | 20.796 s | 11.038 s | 10.819 s |
| `heter_skip_cpu_attention_compute` | 保留异构 D2H/H2D、metadata、CPU KV 写入和同步，但跳过 C++ CPU attention 计算。 | 12.488 s | 11.699 s | 3.877 s | 8.611 s |

## 关键差分

- `Heter - GPU default = 18.446 s`。
- `Heter - GPU forced eager = 17.330 s`。
- `Heter wall - decode CPU - GPU default wall = 7.408 s`。
- `Heter wall - decode CPU - GPU eager wall = 6.292 s`。
- `Heter - Heter skip CPU compute = 9.370 s`。
- `Heter skip CPU compute - GPU eager = 7.960 s`。
- `GPU forced eager - GPU default = 1.116 s`。

## 互斥拆解

| 组别 | Decode CPU attention | Model forward without CPU attention | Preprocess | Postprocess/logits | Sampling/bookkeeping | Output processor | API/scheduler other |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gpu_default_graph` | 0.000 s | 0.031 s | 0.349 s | 0.260 s | 0.101 s | 0.027 s | 2.644 s |
| `gpu_forced_eager` | 0.000 s | 3.588 s | 0.367 s | 0.251 s | 0.110 s | 0.018 s | 0.194 s |
| `heter_cpu_attention` | 11.038 s | 9.757 s | 0.532 s | 0.085 s | 0.145 s | 0.054 s | 0.247 s |
| `heter_skip_cpu_attention_compute` | 3.877 s | 7.823 s | 0.381 s | 0.057 s | 0.105 s | 0.020 s | 0.224 s |

## 初步结论

- 你的判断是对的：`Heter wall - decode_cpu_path_layer_total` 不应被简单解释成 GPU 原本那部分计算。正常异构中该值为 `10.819 s`，明显高于 GPU eager 的 `4.528 s`。
- 关闭 CUDA Graph 本身只解释约 `1.116 s`：也就是 `GPU forced eager - GPU default`。
- 更大的问题来自异构路径本身的 per-layer 同步/往返。skip-compute 组已经跳过 C++ CPU attention 计算，但 request wall 仍为 `12.488 s`，比 GPU eager 多 `7.960 s`。
- 正常异构比 skip-compute 多 `9.370 s`，这部分近似对应 C++ CPU attention 计算以及依赖真实 attention 输出后继续执行 GPU 后续算子的增量。
- `Heter skip CPU compute` 是性能归因消融组，不用于 correctness；它保留异构同步/搬运/metadata/KV 写入，只跳过 C++ CPU attention 计算。
- 本实验只用 memory summary profile；未启用逐 event JSONL，避免把记录开销纳入性能结论。

## 临时环境变量

- 仅在测试子进程内设置 `OMP_NUM_THREADS`。
- 仅在测试子进程内设置 `VLLM_HETER_DISABLE_CPU_ATTENTION`、`VLLM_HETER_FORCE_DECODE_EAGER`、`VLLM_HETER_CPU_ATTN_LIB`。
- 仅在 skip-compute 消融子进程内设置 `VLLM_HETER_SKIP_CPU_ATTN_COMPUTE=1`。
- 仅在测试子进程内设置 `VLLM_HETER_REQUEST_PROFILE*` 和 `VLLM_HETER_PROFILE*`。
- 仅在测试子进程内设置 `no_proxy/NO_PROXY`。
- 仅在绘图子进程内设置 `MPLCONFIGDIR` 和 `XDG_CACHE_HOME` 到 `/tmp/matplotlib-codex`。
- 未修改系统持久环境变量，未修改系统设置。
