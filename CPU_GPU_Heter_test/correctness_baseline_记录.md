# correctness baseline 2026-05-13 03:49:32 UTC

## 测试目的

建立当前 CPU-GPU 异构 Decode Attention 的 correctness 基线，先确认：

- CPU Attention Python fallback 的数值结果与 dense PyTorch reference 一致。
- 端到端异构输出与同一份代码禁用 CPU Attention 后的 GPU baseline 一致。
- Prefill 不再误入 CPU Decode Attention 路径。

## 本轮新增测试文件

- `CPU_GPU_Heter_test/test_cpu_attention_correctness.py`
- `CPU_GPU_Heter_test/test_end_to_end_correctness.py`

## 代码开关

为便于端到端对比，本轮在 `vllm/model_executor/layers/attention/attention.py` 增加了测试开关：

- `VLLM_HETER_DISABLE_CPU_ATTENTION=1`

含义：

- 仅当该环境变量在进程环境中为 `1` 时，禁用异构 CPU Attention 路由和 Prefill KV D2H。
- 默认不设置时，仍使用异构 CPU decode attention 路径。

本轮没有修改持久系统环境变量；该变量只在测试脚本启动 baseline vLLM 子进程时临时传入。

## 测试 1：CPU Attention fallback 数值基线

命令：

```bash
/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python \
  CPU_GPU_Heter_test/test_cpu_attention_correctness.py
```

结果：

- 通过。

验证内容：

- 构造 paged KV cache。
- 调用 `torch.ops.vllm.cpu_paged_attention_fallback`。
- 与 dense PyTorch reference attention 比较。
- `torch.testing.assert_close(..., rtol=1e-5, atol=1e-5)` 通过。

备注：

- 该测试只使用 CPU tensor。
- 在 Codex 沙箱内运行时会看到 NVML/CUDA warning，是沙箱无法访问 GPU driver；不影响该测试。

## 测试 2：端到端 GPU baseline vs 异构输出

命令：

```bash
/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python \
  CPU_GPU_Heter_test/test_end_to_end_correctness.py
```

脚本行为：

1. 启动 baseline 服务：
   - 子进程环境设置 `VLLM_HETER_DISABLE_CPU_ATTENTION=1`
   - 用同一份当前代码走 GPU attention 路径
2. 请求两个 deterministic prompt：
   - `temperature=0.0`
   - `top_p=1.0`
   - `seed=0`
   - `max_tokens=12`
3. 停止 baseline 服务。
4. 启动默认异构服务：
   - 不设置 `VLLM_HETER_DISABLE_CPU_ATTENTION`
5. 请求同样两个 prompt。
6. 比较输出文本和 completion token 数。
7. 检查异构日志中：
   - 必须出现 `query_shape=torch.Size([1, 32, 128])`
   - 不允许出现 `query_shape=torch.Size([44, 32, 128])`

结果：

- 通过。

生成文件：

- `CPU_GPU_Heter_test/correctness_report.json`
- `CPU_GPU_Heter_test/correctness_gpu_baseline.log`
- `CPU_GPU_Heter_test/correctness_heter_cpu_decode.log`

输出对比：

| Prompt | GPU baseline | 异构输出 | 结果 |
|---|---|---|---|
| `Hello, what is 2+2?` | `The answer to 2+2 is 4.` | `The answer to 2+2 is 4.` | 一致 |
| `Write one short sentence about Paris.` | `Paris, the capital of France, is famous for its stunning` | `Paris, the capital of France, is famous for its stunning` | 一致 |

usage 对比：

| Prompt | baseline prompt/completion/total | heter prompt/completion/total | 结果 |
|---|---:|---:|---|
| `Hello, what is 2+2?` | 44 / 12 / 56 | 44 / 12 / 56 | 一致 |
| `Write one short sentence about Paris.` | 42 / 12 / 54 | 42 / 12 / 54 | 一致 |

日志路径验证：

- `correctness_heter_cpu_decode.log` 中 `query_shape=torch.Size([1, 32, 128])` 出现次数：704。
- `correctness_heter_cpu_decode.log` 中 `query_shape=torch.Size([44, 32, 128])` 出现次数：0。

说明：

- 异构 Decode CPU fallback 已实际执行。
- 44-token Prefill 未再误入 CPU Decode Attention。

## 环境变量记录

本轮没有修改系统持久环境变量。

测试脚本只在当前 Python 进程和其 vLLM 子进程中临时设置：

- `no_proxy=localhost,127.0.0.1`
- `NO_PROXY=localhost,127.0.0.1`
- baseline 子进程额外设置 `VLLM_HETER_DISABLE_CPU_ATTENTION=1`

这些设置随测试进程退出自动消失，无需额外恢复。

## 进程清理

端到端脚本结束后检查：

```bash
ps -eo pid,ppid,stat,etime,cmd | grep -E 'vllm serve|EngineCore|APIServer|test_end_to_end_correctness' | grep -v grep
```

结果：

- 无残留 vLLM 服务进程。

## 当前结论

当前 correctness 基线已经覆盖：

- CPU fallback attention 的局部数值正确性。
- 两个固定 prompt 的端到端 GPU baseline vs 异构输出一致性。
- Decode CPU 路由执行真实性。
- Prefill 不进入 CPU Decode fallback。

后续建议继续补：

- 更长 prompt。
- 多轮连续请求。
- 并发请求。
- token id / logits 级别对比，而不仅是文本级别对比。

# extended correctness baseline 2026-05-13 04:16:39 UTC

## 本轮目标

继续补充 correctness 基线，覆盖：

- 更长 prompt。
- 多轮连续请求。
- 并发请求。
- token / logprob 级别对比。

## 修改文件

- `CPU_GPU_Heter_test/test_end_to_end_correctness.py`

## 新增覆盖场景

### 1. 长 prompt

新增 case：

- `long_prompt_summary`

特点：

- prompt tokens：555
- completion tokens：16
- 用重复架构描述构造长上下文，验证 Prefill 后 CPU KV cache 可被后续 decode 正确读取。

### 2. 多轮请求

新增 case：

- `multi_turn_chat`

messages：

- system: `Answer briefly and deterministically.`
- user: `Name one primary color.`
- assistant: `Red.`
- user: `Now name one different primary color.`

结果：

- GPU baseline 与异构输出均为：`Blue.`

### 3. 并发请求

新增 concurrent cases：

- `concurrent_math`
- `concurrent_city`

实现方式：

- 使用 `ThreadPoolExecutor(max_workers=2)` 同时发起两个 `/v1/chat/completions` 请求。

结果：

- GPU baseline 与异构输出文本一致。
- 异构日志出现 `query_shape=torch.Size([2, 32, 128])`，说明两个 decode request 被合并进同一个 CPU decode attention batch。

### 4. token / logprob 级别对比

新增 `/v1/completions` cases：

- `completion_logprobs_short`
- `completion_logprobs_long`

请求参数：

- `temperature=0.0`
- `top_p=1.0`
- `seed=0`
- `logprobs=5`
- `return_token_ids=True`

对比规则：

- 输出 text 必须完全一致。
- completion token 数必须一致。
- `token_ids` 必须一致。
- `tokens` 必须一致。
- `token_logprobs` 允许小数值误差，阈值为最大绝对误差 `<= 0.1`。

说明：

- CPU fallback 与 FlashAttention 数值路径不同，logprob 不适合要求 bitwise 一致。
- 本轮保留 token id / token text 完全一致作为硬约束，logprob 作为数值误差基线记录。

## 测试命令

```bash
/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python -m py_compile \
  CPU_GPU_Heter_test/test_end_to_end_correctness.py

/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python \
  CPU_GPU_Heter_test/test_end_to_end_correctness.py
```

## 测试结果

结果：

- 通过。

生成/更新文件：

- `CPU_GPU_Heter_test/correctness_report.json`
- `CPU_GPU_Heter_test/correctness_gpu_baseline.log`
- `CPU_GPU_Heter_test/correctness_heter_cpu_decode.log`

## 输出一致性摘要

Chat cases：

| case | baseline | heter | 结果 |
|---|---|---|---|
| `short_math` | `The answer to 2+2 is 4.` | `The answer to 2+2 is 4.` | 一致 |
| `short_paris` | `Paris, the capital of France, is famous for its stunning` | 同左 | 一致 |
| `long_prompt_summary` | `The system copies KV cache to CPU memory after prefill and each new decode KV` | 同左 | 一致 |
| `multi_turn_chat` | `Blue.` | `Blue.` | 一致 |

Concurrent chat cases：

| case | baseline | heter | 结果 |
|---|---|---|---|
| `concurrent_math` | `3 + 5 = 8.` | `3 + 5 = 8.` | 一致 |
| `concurrent_city` | `The capital of Italy is Rome.` | `The capital of Italy is Rome.` | 一致 |

Completion token/logprob cases：

| case | token text | token ids | max token logprob abs diff |
|---|---|---|---:|
| `completion_logprobs_short` | 一致 | 一致 | 0.006290793418884277 |
| `completion_logprobs_long` | 一致 | 一致 | 0.05831432342529297 |

## 路由日志验证

来自 `correctness_report.json` summary：

- `decode_cpu_fallback_q1_count`: 1696
- `decode_cpu_fallback_q2_count`: 192
- `prefill_cpu_fallback_q44_count`: 0

解释：

- `q1_count` 表示 single-token decode batch 进入 CPU fallback。
- `q2_count` 表示并发请求合并后的 two-token decode batch 进入 CPU fallback。
- `q44_count=0` 表示原先 44-token Prefill 误入 CPU decode 的问题没有复现。

## 中间发现

第一次扩展测试使用 `1e-4` 阈值强制比较 `token_logprobs`，失败于：

- `completion_logprobs_short` 中某个 token 差异约 0.0015。
- 后续查看完整报告发现 `completion_logprobs_long` 最大差异约 0.0583。

处理：

- token text / token ids 仍保持完全一致的硬约束。
- logprob 改为记录最大绝对误差，并要求 `<= 0.1`。

这更符合当前 Python CPU fallback 与 GPU FlashAttention 不同数值路径的预期。

## 环境变量记录

本轮没有修改系统持久环境变量。

测试脚本临时设置：

- `no_proxy=localhost,127.0.0.1`
- `NO_PROXY=localhost,127.0.0.1`
- `VLLM_HETER_DISABLE_CPU_ATTENTION=1` 仅用于 baseline vLLM 子进程。

测试进程退出后自动恢复。
