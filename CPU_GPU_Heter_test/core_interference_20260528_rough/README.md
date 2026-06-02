# Core 占用干扰粗粒度实验

本目录用于第一轮 core-bound 背景负载敏感性实验。

实验矩阵：

- LLM CPU attention 线程/绑核规模：`8, 16, 32, 64, 96, 128`
- 干扰逻辑核数：`0, 10, 20, 40, 60, 80, 100, 120`
- LLM 亲和性模式：`unbound` 与 `bound`

实验原则：

- 干扰程序使用 `core_burner.c`，只做寄存器内算术循环，尽量避免引入 DRAM bandwidth 干扰。
- `bound` 模式下优先让 LLM cpuset 与干扰 cpuset disjoint；资源不足时记录 logical/physical overlap。
- `unbound` 模式下 LLM 不绑定 CPU affinity，但仍设置 `OMP_NUM_THREADS`，用于观察 Linux 调度器自由迁移时的性能变化。
- profile 使用 memory summary 差分，不启用逐事件 JSONL，避免记录开销污染性能结论。

主要输出：

- `core_interference_results.json`
- `core_interference_raw.csv`
- `core_interference_summary.md`
- `core_interference_*.pdf`
- 每个 server/run 的日志文件
