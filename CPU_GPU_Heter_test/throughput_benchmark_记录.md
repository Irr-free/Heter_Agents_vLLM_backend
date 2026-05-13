# 吞吐实验记录：纯 GPU vs CPU-GPU 异构 Decode Attention

## 实验时间

2026-05-13 15:37 UTC

## 实验目的

对比当前同一份代码中两种运行模式的 decode 吞吐：

- **纯 GPU baseline**：通过临时环境变量 `VLLM_HETER_DISABLE_CPU_ATTENTION=1` 禁用 CPU decode attention 路径。
- **CPU-GPU 异构路径**：使用默认路径，Prefill/其他算子在 GPU，Decode Self-Attention 在 vLLM 原生 C++ CPU attention 中执行。

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
- 异构子进程额外设置：`VLLM_HETER_CPU_ATTN_LIB`

## 输出文件

- 测试脚本：`CPU_GPU_Heter_test/benchmark_heter_vs_gpu.py`
- 原始结果：`CPU_GPU_Heter_test/throughput_benchmark_results.json`
- 可视化图：`CPU_GPU_Heter_test/throughput_benchmark_figure.svg`
- GPU 服务日志：`CPU_GPU_Heter_test/throughput_gpu.log`
- 异构服务日志：`CPU_GPU_Heter_test/throughput_heter.log`

## 实验结果

| Context | Batch | GPU decode tok/s | Heter decode tok/s | Heter/GPU | GPU avg latency s | Heter avg latency s |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 89.827 | 21.923 | 24.4% | 0.089 | 0.365 |
| 128 | 2 | 157.213 | 38.231 | 24.3% | 0.102 | 0.419 |
| 512 | 1 | 84.575 | 18.374 | 21.7% | 0.095 | 0.435 |
| 512 | 2 | 147.179 | 31.164 | 21.2% | 0.109 | 0.513 |
| 1024 | 1 | 78.247 | 14.996 | 19.2% | 0.102 | 0.533 |
| 1024 | 2 | 150.079 | 22.571 | 15.0% | 0.107 | 0.709 |

## 结论

1. 当前异构路径的 decode 吞吐约为 `15-38 tok/s`，已经明显高于旧 Python fallback，但仍低于纯 GPU baseline 的 `78-157 tok/s`。
2. batch size 从 1 增加到 2 时，GPU baseline 的吞吐基本接近翻倍；异构路径有扩展，但增益不如 GPU，说明 CPU attention 和 GPU/CPU 搬运仍是主要瓶颈。
3. context length 从 128 增至 1024 时，异构吞吐继续下降，说明当前瓶颈已经从 Python 拼接 KV 转移到 C++ attention、metadata 构造和跨设备往返开销。
4. 这组结果代表的是当前原生 C++ CPU attention 接入后的真实性能，不再是 Python fallback 的下界。

## 备注

`throughput_benchmark_figure.svg` 使用上半部分 log-scale 绝对吞吐柱状图，下半部分显示异构吞吐相对 GPU baseline 的百分比。这样可以避免线性坐标下异构柱状图几乎不可见的问题。
