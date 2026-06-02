# P2K/O200 Anomaly Recheck

- Time: `2026-05-28 13:07:49 UTC`
- Workload: `prompt=2000, output=200, batch=1`
- Repeats: `5`
- Heter OMP: `OMP_NUM_THREADS=64`
- Heter CPU affinity: `taskset -c 0-63`
- Prefix caching is not disabled; each repeat uses a fresh vLLM server and one measured request.
- Metric: `completion_tokens / request_wall_time`.

## Summary

- Previous matrix heter point: `7.692 tok/s`.
- Repeat mean: `22.817 tok/s`.
- Repeat median: `23.198 tok/s`.
- Repeat stdev: `0.811 tok/s`.
- GPU matrix reference: `62.879 tok/s`.
- Repeat mean / GPU reference: `36.29%`.
- Wall mean: `8.775 s`.

## Repeats

| Repeat | Heter tok/s | Wall s | Attempt |
|---:|---:|---:|---:|
| 1 | 23.450 | 8.529 | 1 |
| 2 | 23.367 | 8.559 | 1 |
| 3 | 21.506 | 9.300 | 1 |
| 4 | 22.563 | 8.864 | 1 |
| 5 | 23.198 | 8.622 | 1 |

## Files

- PDF: `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/gpu_vs_heter_throughput_20260528/p2k_o200_anomaly_recheck/p2k_o200_anomaly_recheck.pdf`
- JSON: `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/gpu_vs_heter_throughput_20260528/p2k_o200_anomaly_recheck/p2k_o200_anomaly_recheck_results.json`
- CSV: `/mnt/raid5/zb_data/HeterAgents/vLLM server/vllm/CPU_GPU_Heter_test/gpu_vs_heter_throughput_20260528/p2k_o200_anomaly_recheck/p2k_o200_anomaly_recheck_raw.csv`
