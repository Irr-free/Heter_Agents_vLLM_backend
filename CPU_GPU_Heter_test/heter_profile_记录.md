# 异构系统细粒度 Profile 记录

实验时间：2026-05-13 05:31:17 GMT

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

| Context | Wall time s | Completion tokens | Instrumented sum s | CPU attention Python s | Output H2D s | Decode KV D2H s | Metadata D2H s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 29.507 | 8 | 28.126 | 27.923 | 0.030 | 0.017 | 0.017 |
| 512 | 26.818 | 8 | 24.984 | 24.795 | 0.031 | 0.016 | 0.016 |

## CPU Attention Python Fallback 内部分解

| Context | CPU attention total s | Collect/concat KV s | Repeat KV s | QK s | Softmax s | PV s | Collect/total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 27.923 | 25.075 | 0.346 | 0.081 | 0.005 | 0.051 | 89.80% |
| 512 | 24.795 | 21.987 | 0.434 | 0.169 | 0.007 | 0.134 | 88.68% |

## 解释

本 profile 的 `wall time` 是 API 端到端请求时间。`Instrumented sum` 是 attention 内部事件的累计和，包含逐层、逐 token 事件，因此用于定位热点，而不是严格等同于端到端 wall time。

本次结果中 `cpu_attention_python_total` 占主导，并且其内部主要耗时来自 `cpu_attention_collect_kv`，即 Python fallback 每层每 token 都从 paged KV block 中逐块取出、`torch.cat` 拼接 K/V，再做 GQA repeat。真正的 QK、softmax、PV 计算占比很小。

因此，当前几十秒开销不是由 prefill GPU 计算主导，也不是由 KV D2H/H2D 主导，而是由 Python fallback 的 paged KV 收集/拼接路径主导。
