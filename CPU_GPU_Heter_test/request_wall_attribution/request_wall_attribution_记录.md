# Request Wall Attribution 实验记录

- 时间：`2026-05-27 06:30:56 UTC`
- workload：`prompt=8000`，`output=400`，`batch=1`
- OpenMP：`OMP_NUM_THREADS=120`
- PDF：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/request_wall_attribution/request_wall_attribution.pdf`
- JSON：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/request_wall_attribution/request_wall_attribution_results.json`

## 12768 的来源

- 上一轮数据中的 `12768` 来自 `decode_cpu_path_layer_total.count`。
- 本次请求 `completion_tokens=400`，第 1 个输出 token 来自 prefill forward 后的采样，不走 decode CPU attention。
- 后续 decode step 数为 `400 - 1 = 399`。
- Llama-3.1-8B 有 32 个 decoder layers，因此 decode CPU attention 调用次数为 `399 * 32 = 12768`。

## 实验组

| 组别 | request profile | request wall | decode_cpu_path_layer_total | wall - decode |
|---|---|---:|---:|---:|
| `A_baseline_no_request_profile` | 低扰动 attention memory profile；关闭 request profile。 | 21.695 s | 11.292 s | 10.403 s |
| `B_request_memory` | request profile 内存聚合；attention profile 也使用 memory。 | 23.224 s | 11.681 s | 11.543 s |
| `C_request_jsonl` | request profile 高频 JSONL；仅用于定位，不作为原始耗时口径。 | 21.327 s | 10.647 s | 10.680 s |

## 低扰动互斥拆解

该表用 B 组 request memory profile 做互斥拆解，避免把嵌套事件重复相加。

| component | total s | request wall 占比 | 说明 |
|---|---:|---:|---|
| `Decode CPU attention path` | 11.681011 | 50.30% | `decode_cpu_path_layer_total`，也就是异构 decode attention 外层路径。 |
| `Other model forward` | 10.350079 | 44.57% | `gpu_model_runner_model_forward - decode_cpu_path_layer_total`，主要对应 prefill forward 以及 decode 中除 self-attention 之外的 GPU 算子，例如 QKV/O projection、MLP、norm/residual、logits 前隐藏状态计算等。 |
| `Model runner preprocess` | 0.619660 | 2.67% | 更新 batch 状态、准备输入、slot mapping、attention metadata 等。 |
| `Sampling/bookkeeping` | 0.156016 | 0.67% | 采样、token 状态更新、异步输出准备等。 |
| `Postprocess/logits` | 0.090572 | 0.39% | model forward 之后的 hidden state 选择、logits 计算和 ModelRunner postprocess。 |
| `API/scheduler/output other` | 0.326593 | 1.41% | 上述项之外的 API server、scheduler、output processor、计时误差和未覆盖边界。 |

## 高频 JSONL 的 prefill/decode 定位

该表只用于定位，因为 C 组包含 JSONL 记录开销；但它携带 `scheduled_tokens` 元数据，可以区分 prefill 和 decode。

| item | total s | count/来源 |
|---|---:|---|
| Prefill model forward | 0.429383 | `scheduled_tokens=8001`，1 次 |
| Decode model forward | 19.314021 | `scheduled_tokens=1`，399 次 |
| Decode CPU attention path | 10.646896 | attention profile，12768 次 layer 调用 |
| Decode other model forward | 8.667125 | `decode model forward - decode CPU attention path` |

关键解释：`request wall - decode_cpu_path_layer_total` 大，并不是因为还有一个同量级的未知同步空洞；主要是因为完整 request wall 还包含 prefill forward，以及每个 decode step 中 self-attention 之外的 GPU 计算路径。

## 低扰动 request profile 主要事件

| event | total s | count | avg ms |
|---|---:|---:|---:|
| `openai_completion_create_total` | 23.213211 | 1 | 23213.210922 |
| `openai_completion_generate_iteration` | 23.190648 | 1 | 23190.648050 |
| `engine_core_step_enqueue_total` | 23.109876 | 402 | 57.487254 |
| `engine_core_execute_model_submit` | 22.860830 | 402 | 56.867737 |
| `gpu_worker_execute_model_total` | 22.838044 | 402 | 56.811054 |
| `gpu_worker_model_runner_execute_model` | 22.827797 | 402 | 56.785565 |
| `gpu_model_runner_execute_model_total` | 22.809973 | 400 | 57.024932 |
| `gpu_model_runner_model_forward` | 22.031090 | 400 | 55.077726 |
| `gpu_model_runner_preprocess_total` | 0.619660 | 400 | 1.549150 |
| `engine_core_sample_tokens_call_submit` | 0.188384 | 400 | 0.470960 |
| `gpu_worker_sample_tokens_total` | 0.163575 | 400 | 0.408938 |
| `gpu_model_runner_sample_tokens_total` | 0.156016 | 400 | 0.390039 |
| `gpu_model_runner_postprocess_total` | 0.090572 | 400 | 0.226430 |
| `gpu_model_runner_sample_kernel_total` | 0.049402 | 400 | 0.123506 |
| `gpu_model_runner_debug_log_write` | 0.048362 | 799 | 0.060528 |
| `engine_core_schedule` | 0.047333 | 402 | 0.117744 |
| `output_processor_process_outputs_total` | 0.041627 | 400 | 0.104067 |
| `gpu_model_runner_bookkeeping_sync` | 0.035827 | 400 | 0.089569 |
| `engine_core_scheduler_update_from_output` | 0.033707 | 402 | 0.083848 |
| `output_processor_process_one_output` | 0.029855 | 400 | 0.074638 |
| `openai_completion_render_request` | 0.021656 | 1 | 21.656116 |
| `gpu_model_runner_build_output` | 0.003636 | 400 | 0.009090 |
| `engine_core_wait_model_or_sample_future` | 0.001642 | 402 | 0.004083 |
| `openai_completion_build_response` | 0.000262 | 1 | 0.261820 |

## 高频 JSONL 诊断事件

注意：本组包含逐 event JSONL 写入观察者开销，不能作为原始系统耗时比例，只用于定位细粒度阶段。

| event | total s | count | avg ms |
|---|---:|---:|---:|
| `openai_completion_create_total` | 21.315927 | 1 | 21315.927200 |
| `openai_completion_generate_iteration` | 21.295295 | 1 | 21295.294890 |
| `engine_core_step_enqueue_total` | 21.165988 | 402 | 52.651711 |
| `engine_core_execute_model_submit` | 20.676429 | 402 | 51.433902 |
| `gpu_worker_execute_model_total` | 20.642762 | 402 | 51.350153 |
| `gpu_worker_model_runner_execute_model` | 20.602575 | 402 | 51.250187 |
| `gpu_model_runner_execute_model_total` | 20.573841 | 400 | 51.434601 |
| `gpu_model_runner_model_forward` | 19.743404 | 400 | 49.358510 |
| `gpu_model_runner_preprocess_total` | 0.536775 | 400 | 1.341938 |
| `engine_core_sample_tokens_call_submit` | 0.307921 | 400 | 0.769801 |
| `gpu_worker_sample_tokens_total` | 0.273072 | 400 | 0.682681 |
| `gpu_model_runner_sample_tokens_total` | 0.254665 | 400 | 0.636663 |
| `output_processor_process_outputs_total` | 0.090979 | 400 | 0.227447 |
| `gpu_model_runner_postprocess_total` | 0.087203 | 400 | 0.218007 |
| `gpu_model_runner_sample_kernel_total` | 0.048783 | 400 | 0.121957 |
| `engine_core_schedule` | 0.042257 | 402 | 0.105117 |
| `gpu_model_runner_debug_log_write` | 0.037709 | 799 | 0.047195 |
| `gpu_model_runner_bookkeeping_sync` | 0.032330 | 400 | 0.080825 |
| `engine_core_scheduler_update_from_output` | 0.031173 | 402 | 0.077544 |
| `output_processor_process_one_output` | 0.029628 | 400 | 0.074070 |
| `openai_completion_render_request` | 0.019488 | 1 | 19.487614 |
| `gpu_model_runner_build_output` | 0.003306 | 400 | 0.008264 |
| `engine_core_wait_model_or_sample_future` | 0.001845 | 402 | 0.004591 |
| `openai_completion_build_response` | 0.000239 | 1 | 0.239469 |

## 初步解释口径

- baseline 中 `request wall - decode_cpu_path_layer_total = 10.403 s`。
- 低扰动 request profile 相比 baseline 的 request wall 差值为 `1.529 s`，用于估计 request profile 观察者效应。
- 低扰动 request profile 中 `request wall - decode_cpu_path_layer_total = 11.543 s`。
- B 组中，`Other model forward = 10.350 s`，占 request wall 的 `44.57%`；这是 wall-decode 差值的主要来源。
- C 组 JSONL 的 scheduled-token 拆分显示：prefill model forward 约 `0.429 s`，399 次 decode model forward 约 `19.314 s`，其中 decode CPU attention path 约 `10.647 s`。
- 结论应优先基于 baseline 与 memory profile；JSONL 组只辅助定位具体操作，不进入原始开销结论。

## 临时环境变量

- 仅在测试子进程内设置 `OMP_NUM_THREADS`。
- 仅在测试子进程内设置 `VLLM_HETER_PROFILE*` 和 `VLLM_HETER_REQUEST_PROFILE*`。
- 仅在测试子进程内设置 `VLLM_HETER_CPU_ATTN_LIB`。
- 仅在测试子进程内设置 `no_proxy/NO_PROXY`。
- 未修改系统持久环境变量，未修改系统超线程或 CPU online 设置。
