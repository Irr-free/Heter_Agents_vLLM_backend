# 吞吐实验记录：纯 GPU vs CPU-GPU 异构 Decode Attention

## 实验时间

2026-05-13 04:55:13 UTC

## 实验目的

对比当前同一份代码中两种运行模式的 decode 吞吐：

- **纯 GPU baseline**：通过临时环境变量 `VLLM_HETER_DISABLE_CPU_ATTENTION=1` 禁用 CPU decode attention 路径。
- **CPU-GPU 异构路径**：使用默认路径，Prefill/其他算子在 GPU，Decode Self-Attention 在 CPU fallback 中执行。

## 实验配置

- 工作目录：`/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm`
- Python/conda 环境：`/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python`
- vLLM 命令：`/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/vllm serve`
- 模型：`/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- 服务端口：`8001`
- `--max-num-seqs 2`
- `--gpu-memory-utilization 0.30`
- `--generation-config vllm`
- context length：`128, 512, 1024`
- batch size：`1, 2`
- 每个请求 decode token 数：`max_tokens=8`
- 每种模式正式计时前执行一次 warmup：`context_len=128, batch_size=1`

## 临时环境变量

仅在测试脚本/子进程中临时设置，未修改系统持久环境变量：

- `no_proxy=localhost,127.0.0.1`
- `NO_PROXY=localhost,127.0.0.1`
- 纯 GPU baseline 子进程额外设置：`VLLM_HETER_DISABLE_CPU_ATTENTION=1`

## 输出文件

- 测试脚本：`CPU_GPU_Heter_test/benchmark_heter_vs_gpu.py`
- 原始结果：`CPU_GPU_Heter_test/throughput_benchmark_results.json`
- 可视化图：`CPU_GPU_Heter_test/throughput_benchmark_figure.svg`
- GPU 服务日志：`CPU_GPU_Heter_test/throughput_gpu.log`
- 异构服务日志：`CPU_GPU_Heter_test/throughput_heter.log`

## 实验结果

| Context | Batch | GPU decode tok/s | Heter decode tok/s | Heter/GPU | GPU avg latency s | Heter avg latency s |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 102.001 | 0.291 | 0.286% | 0.078 | 27.461 |
| 128 | 2 | 159.613 | 0.322 | 0.202% | 0.099 | 49.720 |
| 512 | 1 | 86.107 | 0.321 | 0.373% | 0.092 | 24.904 |
| 512 | 2 | 162.569 | 0.341 | 0.210% | 0.096 | 46.863 |
| 1024 | 1 | 88.850 | 0.327 | 0.368% | 0.089 | 24.436 |
| 1024 | 2 | 163.622 | 0.330 | 0.202% | 0.095 | 48.486 |

## 结论

1. 当前异构路径的 decode 吞吐约为 `0.29-0.34 tok/s`，明显低于纯 GPU baseline 的 `86-164 tok/s`。
2. batch size 从 1 增加到 2 时，GPU baseline 的吞吐基本接近翻倍；异构路径没有获得同等 batch 扩展收益，说明当前 CPU fallback 路径存在严重串行开销或 Python 调度开销。
3. context length 从 128 增至 1024 时，异构吞吐变化不大，这说明当前瓶颈主要不是单次 CPU attention 的理论计算量，而更可能是 Python fallback、逐层同步、张量搬运/reshape、CPU/GPU 边界调度等固定开销。
4. 该结果不能代表最终 C++ CPU kernel 的潜在性能，只能说明当前 correctness-first Python fallback 版本还不具备性能竞争力。

## 备注

`throughput_benchmark_figure.svg` 使用上半部分 log-scale 绝对吞吐柱状图，下半部分显示异构吞吐相对 GPU baseline 的百分比。这样可以避免线性坐标下异构柱状图几乎不可见的问题。
