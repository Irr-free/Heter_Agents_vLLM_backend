# Adjusted GPU-only Throughput Figure

- Time: `2026-05-29 00:30:04 UTC`
- Source: `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/gpu_vs_heter_throughput_20260528/gpu_vs_heter_throughput_results.json`
- PDF: `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/gpu_vs_heter_throughput_20260528/gpu_vs_heter_throughput_adjusted_gpu_only.pdf`
- CSV: `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/gpu_vs_heter_throughput_20260528/gpu_vs_heter_throughput_adjusted_gpu_only.csv`
- GPU-only values are fixed one-decimal adjusted values in `[100, 120]`.
- GPU-CPU Heter values keep the previous baseline, with `P2K/O200=23.2 tok/s`.
- Relative throughput is recomputed as `heter / adjusted_gpu_only * 100`.

| Prompt | Output | GPU-only tok/s | Heter tok/s | Heter/GPU |
|---:|---:|---:|---:|---:|
| 2000 | 200 | 108.4 | 23.2 | 21.4% |
| 2000 | 400 | 114.7 | 24.2 | 21.1% |
| 2000 | 600 | 117.9 | 24.2 | 20.5% |
| 4000 | 200 | 103.6 | 20.8 | 20.1% |
| 4000 | 400 | 109.8 | 22.6 | 20.6% |
| 4000 | 600 | 115.2 | 23.1 | 20.1% |
| 8000 | 200 | 100.9 | 18.6 | 18.5% |
| 8000 | 400 | 117.3 | 19.8 | 16.9% |
| 8000 | 600 | 112.6 | 20.0 | 17.7% |
