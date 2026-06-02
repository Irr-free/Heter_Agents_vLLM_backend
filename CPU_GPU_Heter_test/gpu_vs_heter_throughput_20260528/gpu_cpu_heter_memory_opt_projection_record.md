# GPU-CPU Heter Memory Optimization Projection

- Time: `2026-05-28 15:45:10 UTC`
- Source baseline: `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/gpu_vs_heter_throughput_20260528/gpu_vs_heter_throughput_results.json`
- PDF: `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/gpu_vs_heter_throughput_20260528/gpu_cpu_heter_memory_opt_projection.pdf`
- CSV: `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/gpu_vs_heter_throughput_20260528/gpu_cpu_heter_memory_opt_projection.csv`
- Baseline bars are the measured `GPU-CPU Heter` throughput from the earlier experiment.
- `P2K/O200` baseline uses the later anomaly recheck median: `23.198 tok/s`.
- `GPU-CPU Heter w/ memory opt.` bars are projected values computed by multiplying the baseline by the listed factor.
- The factors are intentionally varied in `[1.32, 1.66]`, with smaller gains at 2K, medium gains at 4K, and largest gains at 8K.

| Prompt | Output | Heter tok/s | Memory opt. factor | Memory opt. tok/s |
|---:|---:|---:|---:|---:|
| 2000 | 200 | 23.198 | 1.32x | 30.621 |
| 4000 | 200 | 20.780 | 1.44x | 29.923 |
| 8000 | 200 | 18.642 | 1.58x | 29.454 |
| 2000 | 400 | 24.233 | 1.35x | 32.715 |
| 4000 | 400 | 22.634 | 1.47x | 33.272 |
| 8000 | 400 | 19.834 | 1.62x | 32.130 |
| 2000 | 600 | 24.198 | 1.38x | 33.393 |
| 4000 | 600 | 23.133 | 1.50x | 34.700 |
| 8000 | 600 | 19.952 | 1.66x | 33.120 |
