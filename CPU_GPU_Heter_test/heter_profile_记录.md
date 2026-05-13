# 异构系统细粒度 Profile 记录

实验时间：2026-05-13 15:32:30 GMT

## 配置

- 模型：`/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- batch size：`1`
- max_tokens：`8`
- context length：`128, 512`
- profile 开关：`VLLM_HETER_PROFILE=1`
- profile 原始事件：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/heter_profile_events.jsonl`
- 可视化图：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/heter_profile_breakdown.svg`
- CPU attention 内部分解图：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/heter_profile_cpu_attention_breakdown.svg`

## 端到端与主要阶段

| Context | Wall time s | Completion tokens | Instrumented sum s | CPU attention C++ s | CPU attention Python s | Output H2D s | Decode KV D2H s | Metadata D2H s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.485 | 8 | 0.161 | 0.058 | 0.000 | 0.014 | 0.016 | 0.033 |
| 512 | 0.553 | 8 | 0.221 | 0.116 | 0.000 | 0.016 | 0.015 | 0.029 |

## CPU Attention Python Fallback 内部分解

| Context | CPU attention total s | Collect/concat KV s | Repeat KV s | QK s | Softmax s | PV s | Collect/total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00% |
| 512 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00% |

## 解释

本 profile 的 `wall time` 是 API 端到端请求时间。`Instrumented sum` 是 attention 内部事件的累计和，包含逐层、逐 token 事件，因此用于定位热点，而不是严格等同于端到端 wall time。

如果 `cpu_attention_cpp_total` 非 0 且 `cpu_attention_python_total` 为 0，则 Decode Self-Attention 已进入 vLLM 原生 C++ CPU attention。此时 Python fallback 的逐 block KV 收集/拼接事件应消失。

如果仍出现 `cpu_attention_python_total`，则说明运行时没有成功加载或调用 C++ 扩展，热点仍来自 Python fallback 每层每 token 从 paged KV block 中逐块取出、`torch.cat` 拼接 K/V，再做 GQA repeat。

本轮 profile 用于区分这两种路径，并记录 C++ kernel 接入后的实际端到端耗时。
