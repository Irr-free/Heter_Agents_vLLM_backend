# KV split P8K/O400 三层 Profile PDF 记录

- 时间：`2026-05-27 02:36:53 UTC`
- 数据来源：`CPU_GPU_Heter_test/kvsplit_p8k_o400_deep_profile_results.json`
- 原始事件：
  - `CPU_GPU_Heter_test/kvsplit_p8k_o400_server_outer_events.jsonl`
  - `CPU_GPU_Heter_test/kvsplit_p8k_o400_server_inner_events.jsonl`
- 绘图脚本：`CPU_GPU_Heter_test/kvsplit_p8k_o400_three_level_profile_plot.py`
- 输出 PDF：`CPU_GPU_Heter_test/kvsplit_p8k_o400_three_level_profile.pdf`
- workload：`prompt=8000`，`output=400`，`batch=1`
- OpenMP：`OMP_NUM_THREADS=120`

## 生成方式

本次没有重新启动 vLLM 服务，没有重新运行端到端请求，只使用已有 profile JSON/JSONL 汇总结果生成 PDF。

绘图使用 `vLLM_GPU` conda 环境中的 `matplotlib 3.10.9`。为了避免 matplotlib/fontconfig 尝试写入用户目录，本次绘图命令只在子进程内临时设置：

```bash
MPLCONFIGDIR=/tmp/matplotlib-codex
XDG_CACHE_HOME=/tmp/matplotlib-codex
```

未修改持久系统环境变量，未修改 shell 配置，未修改 vLLM 执行代码。

## PDF 内容

PDF 共 6 页：

| 页码 | 内容 |
|---:|---|
| 1 | 三层 profile 总图：`decode_cpu_path_layer_total`、`cpu_attention_cpp_total`、`C++ inner` |
| 2 | Layer 1 条形图和读图说明 |
| 3 | Layer 1 每个耗时项的具体含义 |
| 4 | Layer 2 条形图、表格和说明 |
| 5 | Layer 3 条形图和读图说明 |
| 6 | Layer 3 每个耗时项的具体含义和总体结论 |

## 三层数值

### Layer 1：decode_cpu_path_layer_total

`decode_cpu_path_layer_total = 14.550 s`，事件数 `12768`。

| 耗时项 | 时间 | 占比 | 含义 |
|---|---:|---:|---|
| C++ CPU attention op total | 6.606 s | 45.40% | 调用 vLLM 原生 CPU paged attention C++ op 的外层累计时间 |
| Unattributed outer gap | 3.463 s | 23.80% | 外层 timer 覆盖但未单独插桩的边界开销 |
| Metadata copy/package excl. scheduler | 1.114 s | 7.66% | metadata 小 tensor D2H、dtype 转换、CPU tensor 分配和 Python 打包，不含 scheduler 子项 |
| Decode K/V/slot D2H | 0.742 s | 5.10% | decode 新生成 K/V 和 slot_mapping 等小 tensor 从 GPU 搬到 CPU |
| Output H2D | 0.715 s | 4.91% | CPU attention 输出从 CPU 搬回 GPU output tensor |
| CPU KV cache write | 0.684 s | 4.70% | decode 新 token K/V reshape 后写入 CPU paged KV cache |
| Wait prefill D2H stream | 0.530 s | 3.64% | 等待该层 prefill KV cache 异步 D2H 完成 |
| CPU scheduler metadata | 0.310 s | 2.13% | 调用 C++ `get_scheduler_metadata` 生成调度元数据 |
| Query D2H | 0.247 s | 1.70% | decode query 从 GPU 搬到 CPU |
| CPU output alloc | 0.140 s | 0.96% | 为 CPU attention 输出分配 CPU pinned tensor |

说明：`decode_cpu_scheduler_metadata` 是 `decode_metadata_d2h` 的子项。PDF 中将 `decode_metadata_d2h` 拆成 `metadata copy/package excl. scheduler` 和 `CPU scheduler metadata` 两个非重叠部分，避免重复计数。

### Layer 2：cpu_attention_cpp_total

`cpu_attention_cpp_total = 6.606 s`。

| 耗时项 | 时间 | 占比 | 含义 |
|---|---:|---:|---|
| C++ inner elapsed | 4.825 s | 73.04% | C++ CPU attention 内部 wall-time 计时，覆盖 worker 并行区主要 attention 执行路径 |
| C++ op wrapper/profile/boundary gap | 1.781 s | 26.96% | C++ inner elapsed 之外的 custom op dispatch、wrapper 参数处理、profile 写入和返回边界等 |

### Layer 3：C++ inner

`C++ inner elapsed = 4.825 s`。下表为 OpenMP worker 累计 thread-time，总和 `273.046 thread-s`，约 `56.6 worker-equivalent`。

| 耗时项 | Thread-time | 占比 | 含义 |
|---|---:|---:|---|
| PV tile GEMM + V stream | 106.183 s | 38.89% | P×V tile 计算及 V cache 流式读取 |
| QK tile GEMM + K stream | 95.178 s | 34.86% | Q×K tile 计算及 K cache 流式读取 |
| Sync/wait | 22.269 s | 8.16% | KV split 并行区中的 split barrier 和 reduce flag 等同步等待 |
| QK/PV loop other | 15.196 s | 5.57% | QK/PV 主循环中未归到 tile GEMM、block lookup 的循环控制和边界处理 |
| Softmax | 14.813 s | 5.42% | online softmax，包括 max/sum 维护和概率归一化 |
| Output/reduce | 14.085 s | 5.16% | partial output 写入、split reduction 和 final output 写回 |
| Paged block lookup | 2.667 s | 0.98% | 根据 block_table 查找 paged KV cache block 地址/索引 |
| Task acquire/atomic | 2.244 s | 0.82% | OpenMP worker 通过 atomic/fetch-add 等方式领取 work item |
| Q copy/scale | 0.367 s | 0.13% | Q 拷贝/缩放到 CPU kernel 使用的局部格式 |
| Mask | 0.045 s | 0.02% | causal/sliding-window mask 等 score 屏蔽操作 |

## 校验

- 使用 Ghostscript 将 PDF 渲染为 PNG，6 页均成功渲染。
- 渲染分辨率：`1980x1320`。
- 人工抽查：总图、Layer 1 表格、Layer 2 页面、Layer 3 表格均未发现明显文字缺失、截断或重合。

## 结论

1. 第一层看真实 decode CPU path 成本；第二层看 C++ op 外层调用成本；第三层只看 C++ inner 的线程级热点。
2. 当前 profile 的主要结论是：外层 C++ op 之外仍有明显 metadata、D2H/H2D、CPU KV write 和未归因 gap。
3. C++ inner 内部主要热点是 `PV tile GEMM + V stream` 和 `QK tile GEMM + K stream`，二者合计占 `73.75%`。
4. `Paged block lookup` 和 `Task acquire/atomic` 占比较小，不是当前 profile 的主瓶颈。
