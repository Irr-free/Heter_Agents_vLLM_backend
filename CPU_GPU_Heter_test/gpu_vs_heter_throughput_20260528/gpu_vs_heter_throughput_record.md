# GPU-only vs GPU-CPU Heter Throughput

- Time: `2026-05-28 12:52:40 UTC`
- Model: `/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- Prompt lengths: `[2000, 4000, 8000]`
- Output lengths: `[200, 400, 600]`
- Batch size: `1`
- PDF: `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/gpu_vs_heter_throughput_20260528/gpu_vs_heter_throughput.pdf`
- JSON: `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/gpu_vs_heter_throughput_20260528/gpu_vs_heter_throughput_results.json`
- CSV: `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/gpu_vs_heter_throughput_20260528/gpu_vs_heter_throughput_raw.csv`

## Method

- Metric: `completion_tokens / request_wall_time`.
- GPU-only uses `VLLM_HETER_DISABLE_CPU_ATTENTION=1` in the server subprocess.
- Heter uses `VLLM_HETER_CPU_ATTN_LIB` in the server subprocess.
- Heter uses `OMP_NUM_THREADS=64` and `taskset -c 0-63`.
- Prefix caching is not disabled, but each workload is measured in a fresh server with one measured request.
- No persistent environment variables or system settings were changed.

## Results

| Prompt | Output | GPU-only tok/s | Heter tok/s | Heter/GPU | GPU wall s | Heter wall s |
|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 200 | 62.879 | 7.692 | 12.23% | 3.181 | 26.000 |
| 2000 | 400 | 86.176 | 24.233 | 28.12% | 4.642 | 16.506 |
| 2000 | 600 | 87.257 | 24.198 | 27.73% | 6.876 | 24.796 |
| 4000 | 200 | 64.824 | 20.780 | 32.06% | 3.085 | 9.625 |
| 4000 | 400 | 67.968 | 22.634 | 33.30% | 5.885 | 17.672 |
| 4000 | 600 | 71.749 | 23.133 | 32.24% | 8.362 | 25.937 |
| 8000 | 200 | 54.535 | 18.642 | 34.18% | 3.667 | 10.729 |
| 8000 | 400 | 59.561 | 19.834 | 33.30% | 6.716 | 20.168 |
| 8000 | 600 | 64.042 | 19.952 | 31.15% | 9.369 | 30.073 |
